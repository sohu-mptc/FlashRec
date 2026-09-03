"""Token-level MHA KV pool, page_size=1. Slot 0 is the CUDA-graph dummy.

Free lists live on CPU (SGLang-style). ``alloc()`` is a Python pop + one
pinned async H2D of the chosen indices — never a pageable copy (which would
block the host behind everything already queued on the stream) and never a
GPU ``numel()`` / ``tolist()`` sync.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch

from flashrec.kernel.sglops import store_kv_cache


def _as_cpu_ids(indices) -> List[int]:
    if indices is None:
        return []
    if isinstance(indices, torch.Tensor):
        t = indices.detach().reshape(-1)
        if t.numel() == 0:
            return []
        if t.device.type != "cpu":
            t = t.to("cpu")
        return [int(x) for x in t.tolist()]
    return [int(x) for x in indices]


class _IdxStage:
    """Ring of pinned+GPU int64 buffers for alloc-index uploads.

    ``torch.tensor(list, device=cuda)`` is a pageable H2D that blocks the host
    behind everything queued on the stream (i.e. the previous decode graph).
    A pinned async copy does not. The ring gives reuse distance so a slot is
    not rewritten while its previous H2D may still be in flight.
    """

    SLOTS = 64

    def __init__(self, device: torch.device):
        self.device = device
        self._pin: List[Optional[torch.Tensor]] = [None] * self.SLOTS
        self._gpu: List[Optional[torch.Tensor]] = [None] * self.SLOTS
        self._i = 0

    def upload(self, ids: Sequence[int]) -> torch.Tensor:
        n = len(ids)
        if self.device.type != "cuda":
            return torch.tensor(ids, dtype=torch.int64, device=self.device)
        i = self._i
        self._i = (i + 1) % self.SLOTS
        pin = self._pin[i]
        if pin is None or int(pin.numel()) < n:
            cap = max(64, 1 << max(n - 1, 1).bit_length())
            pin = torch.empty(cap, dtype=torch.int64, pin_memory=True)
            self._pin[i] = pin
            self._gpu[i] = torch.empty(cap, dtype=torch.int64, device=self.device)
        gpu = self._gpu[i]
        pin[:n].numpy()[:] = ids
        gpu[:n].copy_(pin[:n], non_blocking=True)
        return gpu[:n]


class TokenToKVPool:
    """Physical KV: [num_tokens, num_kv_heads, head_dim] per layer.

    Index 0 is reserved as a dummy page (SGLang-style padding for CUDA Graph).
    """

    def __init__(
        self,
        num_layers: int,
        num_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.num_tokens = int(num_tokens)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.row_dim = num_kv_heads * head_dim
        shape = (self.num_tokens, num_kv_heads, head_dim)
        self.k = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        self.v = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        # padded slot 0 is never allocated to a live request
        self._free: List[int] = list(range(1, self.num_tokens))
        self.last_alloc_cpu: List[int] = []
        self._idx_stage = _IdxStage(device)

    def available_size(self) -> int:
        return len(self._free)

    def alloc(self, n: int) -> Optional[torch.Tensor]:
        n = int(n)
        if n <= 0:
            self.last_alloc_cpu = []
            return torch.empty(0, dtype=torch.int64, device=self.device)
        if n > len(self._free):
            self.last_alloc_cpu = []
            return None
        idx = self._free[-n:]
        del self._free[-n:]
        self.last_alloc_cpu = idx
        return self._idx_stage.upload(idx)

    def free(self, indices: Optional[Union[torch.Tensor, Sequence[int]]]) -> None:
        ids = [i for i in _as_cpu_ids(indices) if i > 0]
        if ids:
            self._free.extend(ids)

    def store(
        self, layer_id: int, k: torch.Tensor, v: torch.Tensor, loc: torch.Tensor
    ) -> None:
        loc = loc.view(-1).to(dtype=torch.int64)
        n = int(loc.numel())
        if n == 0:
            return
        k = k.reshape(n, self.num_kv_heads, self.head_dim)
        v = v.reshape(n, self.num_kv_heads, self.head_dim)
        if k.dtype != self.dtype:
            k = k.to(self.dtype)
            v = v.to(self.dtype)
        store_kv_cache(
            self.k[layer_id].view(-1, self.row_dim),
            self.v[layer_id].view(-1, self.row_dim),
            loc,
            k.reshape(n, self.row_dim),
            v.reshape(n, self.row_dim),
        )

    def k_cache(self, layer_id: int) -> torch.Tensor:
        return self.k[layer_id]

    def v_cache(self, layer_id: int) -> torch.Tensor:
        return self.v[layer_id]


class ReqToTokenPool:
    def __init__(self, max_reqs: int, max_seq_len: int, device: torch.device):
        self.max_reqs = int(max_reqs)
        self.max_seq_len = int(max_seq_len)
        self.device = device
        self.req_to_token = torch.zeros(
            (self.max_reqs, self.max_seq_len), dtype=torch.int32, device=device
        )
        # slot 0 is dummy (points at KV page 0)
        self._free: List[int] = list(range(1, self.max_reqs))
        self.last_alloc_cpu: List[int] = []
        self._idx_stage = _IdxStage(device)

    def available_size(self) -> int:
        return len(self._free)

    def alloc(self, n: int) -> Optional[torch.Tensor]:
        n = int(n)
        if n <= 0:
            self.last_alloc_cpu = []
            return torch.empty(0, dtype=torch.int64, device=self.device)
        if n > len(self._free):
            self.last_alloc_cpu = []
            return None
        idx = self._free[-n:]
        del self._free[-n:]
        self.last_alloc_cpu = idx
        # req slots (e.g. beam_pool_indices) live for the whole request, so
        # detach from the staging ring with an async D2D clone.
        return self._idx_stage.upload(idx).clone()

    def free(self, indices) -> None:
        ids = [i for i in _as_cpu_ids(indices) if i > 0]
        if ids:
            self._free.extend(ids)
