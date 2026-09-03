"""Triton gather of ragged page indices from req_to_token (SGLang-style)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _gather_kv_indices_kernel(
        req_to_token_ptr,
        req_pool_indices_ptr,
        seq_lens_ptr,
        kv_indptr_ptr,
        kv_indices_ptr,
        req_to_token_stride: tl.constexpr,
    ):
        BLOCK: tl.constexpr = 512
        pid = tl.program_id(axis=0)
        req_pool_index = tl.load(req_pool_indices_ptr + pid)
        kv_offset = tl.load(kv_indptr_ptr + pid)
        sl = tl.load(seq_lens_ptr + pid).to(tl.int32)
        nloop = tl.cdiv(sl, BLOCK)
        for i in range(nloop):
            offset = tl.arange(0, BLOCK).to(tl.int64) + i * BLOCK
            mask = offset < sl
            data = tl.load(
                req_to_token_ptr + req_pool_index * req_to_token_stride + offset,
                mask=mask,
            )
            tl.store(kv_indices_ptr + kv_offset + offset, data, mask=mask)


_IND_CACHE: Dict[torch.device, torch.Tensor] = {}


def _cached_indptr(n: int, device: torch.device) -> torch.Tensor:
    t = _IND_CACHE.get(device)
    need = max(int(n), 0) + 1
    if t is None or t.device != device or int(t.numel()) < need or t.is_inference():
        # Allocate outside inference mode: warmup runs under inference_mode, and a
        # buffer created there stays an inference tensor for life, so later
        # cumsum(out=...) from a normal-mode caller would raise.
        with torch.inference_mode(False):
            t = torch.zeros(max(need, 1), dtype=torch.int32, device=device)
        _IND_CACHE[device] = t
    # t[0] stays 0 forever (cumsum only writes t[1:]); re-zeroing it here would
    # be a scalar pageable H2D + stream sync on every decode step.
    return t[:need]


def gather_kv_indices(
    req_to_token: torch.Tensor,
    rows: torch.Tensor,
    seq_lens: Union[Sequence[int], torch.Tensor],
    out: Optional[torch.Tensor] = None,
    total: Optional[int] = None,
) -> torch.Tensor:
    """Return concatenated page indices for each row's ``[:seq_len]`` prefix."""
    device = req_to_token.device
    rows = rows.view(-1).to(device=device, dtype=torch.int64)
    n = int(rows.numel())
    if n == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device) if out is None else out[:0]
        )
    seq_list: Optional[List[int]] = None
    if isinstance(seq_lens, torch.Tensor):
        sl_src = seq_lens.view(-1)[:n]
        if sl_src.dtype != torch.int32:
            sl = sl_src.to(dtype=torch.int32)
        else:
            sl = sl_src.contiguous()
        if sl.device != device:
            sl = sl.to(device=device, non_blocking=True)
        if total is None:
            total = int(sl.sum().item())
    else:
        seq_list = [int(x) for x in seq_lens]
        if len(seq_list) != n:
            raise ValueError("rows and seq_lens length mismatch")
        if total is None:
            total = int(sum(seq_list))
        sl = torch.as_tensor(seq_list, dtype=torch.int32, device=device)
    total = int(total)
    if int(sl.numel()) != n:
        raise ValueError("rows and seq_lens length mismatch")
    if total <= 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device) if out is None else out[:0]
        )
    if out is not None:
        dest = out.view(-1)[:total]
        if dest.dtype != torch.int32:
            dest = dest.to(dtype=torch.int32)
    else:
        dest = None
    if _HAS_TRITON and device.type == "cuda":
        indptr = _cached_indptr(n, device)
        torch.cumsum(sl, dim=0, out=indptr[1:])
        if dest is None:
            dest = torch.empty(total, dtype=torch.int32, device=device)
        _gather_kv_indices_kernel[(n,)](
            req_to_token,
            rows,
            sl,
            indptr[:-1],
            dest,
            req_to_token.stride(0),
        )
        return dest
    if seq_list is None:
        seq_list = [int(x) for x in sl.detach().cpu().tolist()]
    max_sl = max(seq_list) if seq_list else 0
    table = req_to_token.index_select(0, rows.to(dtype=torch.int64))[:, :max_sl]
    mask = torch.arange(max_sl, device=device).unsqueeze(0) < sl.unsqueeze(1)
    gathered = table.to(dtype=torch.int32)[mask]
    if dest is not None:
        dest.copy_(gathered)
        return dest
    return gathered
