"""Pinned host staging for list → GPU copies (prefill + decode metadata)."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch


def fill_cpu_seq_lens(dest: torch.Tensor, base: torch.Tensor, step: int) -> None:
    """Write ``base + step + 1`` into a CPU seq_lens buffer (no D2H).

    FlashInfer ``plan()`` builds host_indptr from this tensor immediately.
    A non_blocking GPU→CPU copy of seq_lens races: after a prompt-len change
    the CPU buffer still holds the previous wave, and CUDA-graph decode
    collapses beams onto one SID.
    """
    n = int(dest.numel())
    if n <= 0:
        return
    torch.add(base.view(-1)[:n], int(step) + 1, out=dest.view(-1)[:n])


def _fill_pinned(pin: torch.Tensor, values: Sequence[int], dtype: torch.dtype) -> None:
    """Write ints into an existing pinned CPU slice without allocating a tensor."""
    n = int(pin.numel())
    if n <= 0:
        return
    try:
        pin.numpy()[:] = values
    except (TypeError, ValueError, RuntimeError):
        for i, v in enumerate(values):
            pin[i] = int(v)


class PinnedStage:
    """Grow-only pinned CPU buffers plus optional GPU mirrors.

    ``copy_list`` ping-pongs two pinned slots per name and waits for the prior
    H2D that sourced from a slot before CPU-refilling it. Overwriting a pin
    while a non_blocking H2D still reads it feeds stale ints to the GPU (fused
    expand then writes the next SID token into the wrong column, leaving a
    literal ``0`` / ``!`` in the middle of the SID).
    """

    def __init__(self) -> None:
        self._cpu: Dict[str, torch.Tensor] = {}
        self._gpu: Dict[str, torch.Tensor] = {}
        self._pin_epoch: Dict[str, int] = {}
        self._pin_ready: Dict[str, Optional[torch.cuda.Event]] = {}

    def cpu(self, name: str, n: int, dtype: torch.dtype) -> torch.Tensor:
        n = max(int(n), 0)
        t = self._cpu.get(name)
        if t is None or t.dtype != dtype or int(t.numel()) < max(n, 1):
            t = torch.empty(max(n, 1), dtype=dtype, pin_memory=True)
            self._cpu[name] = t
        return t[: max(n, 0)]

    def gpu(
        self, name: str, n: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        n = max(int(n), 0)
        t = self._gpu.get(name)
        if (
            t is None
            or t.dtype != dtype
            or t.device != device
            or int(t.numel()) < max(n, 1)
        ):
            t = torch.empty(max(n, 1), dtype=dtype, device=device)
            self._gpu[name] = t
        return t[: max(n, 0)]

    def copy_list(
        self,
        name: str,
        values: Sequence[int],
        device: torch.device,
        dtype: torch.dtype = torch.int64,
        dest: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write ``values`` through pinned memory. Returns ``(gpu_or_cpu, pinned)``."""
        n = len(values)
        if dest is not None and dest.device.type == "cpu":
            dest = dest.view(-1)[:n]
            if n:
                _fill_pinned(dest, values, dtype)
            return dest, dest
        epoch = int(self._pin_epoch.get(name, 0))
        self._pin_epoch[name] = epoch + 1
        pin_key = f"{name}#pin{epoch & 1}"
        pin = self.cpu(pin_key, n, dtype)
        # CPU must not refill this slot until the previous H2D that read it
        # has completed; otherwise the in-flight DMA observes the new values.
        prev = self._pin_ready.get(pin_key)
        if prev is not None:
            prev.synchronize()
        if n:
            _fill_pinned(pin, values, dtype)
        if dest is None:
            dest = self.gpu(name, n, dtype, device)
        else:
            dest = dest.view(-1)[:n]
        if n:
            dest.copy_(pin, non_blocking=True)
            if device.type == "cuda" and torch.cuda.is_available():
                ev = torch.cuda.Event()
                ev.record()
                self._pin_ready[pin_key] = ev
        return dest, pin

    def copy_rows(
        self,
        name: str,
        srcs: Sequence[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        dest: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Concatenate 1-D tensors into ``dest`` without ``torch.cat``."""
        n = int(sum(int(s.numel()) for s in srcs))
        if dest is None:
            dest = self.gpu(name, n, dtype, device)
        else:
            dest = dest.view(-1)[:n]
        off = 0
        for src in srcs:
            s = src.reshape(-1)
            if s.dtype != dtype:
                s = s.to(dtype=dtype)
            if s.device != dest.device:
                s = s.to(device=dest.device, non_blocking=True)
            k = int(s.numel())
            if k <= 0:
                continue
            sl = dest[off : off + k]
            if s.data_ptr() != sl.data_ptr():
                sl.copy_(s, non_blocking=True)
            off += k
        return dest

    def copy_lists(
        self,
        name: str,
        parts: Sequence[tuple[Sequence[int], torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype = torch.int64,
    ) -> None:
        """One pinned H2D for several int lists, then slice into GPU dests."""
        gpu_parts: list[tuple[Sequence[int], torch.Tensor]] = []
        for values, dest in parts:
            if dest.device.type == "cpu":
                d = dest.view(-1)[: len(values)]
                if values:
                    _fill_pinned(d, values, dtype)
            else:
                gpu_parts.append((values, dest))
        if not gpu_parts:
            return
        total = int(sum(len(v) for v, _ in gpu_parts))
        epoch = int(self._pin_epoch.get(name, 0))
        self._pin_epoch[name] = epoch + 1
        pin_key = f"{name}#pin{epoch & 1}"
        pin = self.cpu(pin_key, total, dtype)
        prev = self._pin_ready.get(pin_key)
        if prev is not None:
            prev.synchronize()
        off = 0
        for values, _dest in gpu_parts:
            n = len(values)
            if n:
                _fill_pinned(pin[off : off + n], values, dtype)
            off += n
        gpu = self.gpu(name, total, dtype, device)
        if total:
            gpu.copy_(pin, non_blocking=True)
            if device.type == "cuda" and torch.cuda.is_available():
                ev = torch.cuda.Event()
                ev.record()
                self._pin_ready[pin_key] = ev
        off = 0
        for values, dest in gpu_parts:
            n = len(values)
            d = dest.view(-1)[:n]
            if n and d.data_ptr() != gpu[off : off + n].data_ptr():
                d.copy_(gpu[off : off + n], non_blocking=True)
            off += n
