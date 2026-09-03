"""Fused Q/K RMSNorm + Neox RoPE + unscaled FP8 KV scatter."""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_TRITON_OK: Optional[bool] = None
_KERNEL = None


def _try_triton():
    global _TRITON_OK, _KERNEL
    if _TRITON_OK is not None:
        return _KERNEL if _TRITON_OK else None
    _TRITON_OK = False
    try:
        import triton
        import triton.language as tl
    except Exception as exc:
        logger.debug("qk_rope_kv triton unavailable: %s", exc)
        return None

    @triton.jit
    def kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        qw_ptr,
        kw_ptr,
        pos_ptr,
        loc_ptr,
        cs_ptr,
        k_cache_ptr,
        v_cache_ptr,
        stride_qt,
        stride_qh,
        stride_kt,
        stride_kh,
        stride_vt,
        stride_vh,
        stride_ckp,
        stride_ckh,
        stride_cvp,
        stride_cvh,
        n_q,
        n_kv,
        D,
        eps,
        BLOCK: tl.constexpr,
        HALF: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        lo = tl.arange(0, HALF)
        hi = lo + (D // 2)
        mask_lo = lo < (D // 2)
        mask_hi = hi < D
        pos = tl.load(pos_ptr + row)
        loc = tl.load(loc_ptr + row)
        cos = tl.load(cs_ptr + pos * D + lo, mask=mask_lo, other=0.0).to(tl.float32)
        sin = tl.load(cs_ptr + pos * D + hi, mask=mask_hi, other=0.0).to(tl.float32)
        qw1 = tl.load(qw_ptr + lo, mask=mask_lo, other=0.0).to(tl.float32)
        qw2 = tl.load(qw_ptr + hi, mask=mask_hi, other=0.0).to(tl.float32)
        kw1 = tl.load(kw_ptr + lo, mask=mask_lo, other=0.0).to(tl.float32)
        kw2 = tl.load(kw_ptr + hi, mask=mask_hi, other=0.0).to(tl.float32)

        h = 0
        while h < n_q:
            x = tl.load(
                q_ptr + row * stride_qt + h * stride_qh + offs, mask=mask, other=0.0
            ).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
            x1 = (
                tl.load(
                    q_ptr + row * stride_qt + h * stride_qh + lo,
                    mask=mask_lo,
                    other=0.0,
                ).to(tl.float32)
                * rstd
                * qw1
            )
            x2 = (
                tl.load(
                    q_ptr + row * stride_qt + h * stride_qh + hi,
                    mask=mask_hi,
                    other=0.0,
                ).to(tl.float32)
                * rstd
                * qw2
            )
            y1 = x1 * cos - x2 * sin
            y2 = x2 * cos + x1 * sin
            tl.store(
                q_ptr + row * stride_qt + h * stride_qh + lo,
                y1.to(q_ptr.dtype.element_ty),
                mask=mask_lo,
            )
            tl.store(
                q_ptr + row * stride_qt + h * stride_qh + hi,
                y2.to(q_ptr.dtype.element_ty),
                mask=mask_hi,
            )
            h += 1

        h = 0
        while h < n_kv:
            x = tl.load(
                k_ptr + row * stride_kt + h * stride_kh + offs, mask=mask, other=0.0
            ).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
            x1 = (
                tl.load(
                    k_ptr + row * stride_kt + h * stride_kh + lo,
                    mask=mask_lo,
                    other=0.0,
                ).to(tl.float32)
                * rstd
                * kw1
            )
            x2 = (
                tl.load(
                    k_ptr + row * stride_kt + h * stride_kh + hi,
                    mask=mask_hi,
                    other=0.0,
                ).to(tl.float32)
                * rstd
                * kw2
            )
            y1 = (x1 * cos - x2 * sin).to(tl.bfloat16)
            y2 = (x2 * cos + x1 * sin).to(tl.bfloat16)
            tl.store(
                k_cache_ptr + loc * stride_ckp + h * stride_ckh + lo,
                y1.to(tl.float8e4nv),
                mask=mask_lo,
            )
            tl.store(
                k_cache_ptr + loc * stride_ckp + h * stride_ckh + hi,
                y2.to(tl.float8e4nv),
                mask=mask_hi,
            )
            vv = tl.load(
                v_ptr + row * stride_vt + h * stride_vh + offs, mask=mask, other=0.0
            )
            tl.store(
                v_cache_ptr + loc * stride_cvp + h * stride_cvh + offs,
                vv.to(tl.float8e4nv),
                mask=mask,
            )
            h += 1

    _KERNEL = kernel
    _TRITON_OK = True
    return _KERNEL


# Opaque to torch.compile: the triton launch + fp8 cache views break inductor
# fake-tensor propagation; run eagerly inside compiled prefill.
@torch.compiler.disable
def fused_qk_norm_rope_store_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    loc: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> bool:
    """In-place Neox RoPE on Q; RMSNorm Q/K; scatter unscaled FP8 K/V.

    ``q/k/v`` are ``[T, H, D]``. Caches are ``[pages, H_kv, D]`` fp8.
    Returns False if the fused kernel cannot run (caller should fall back).
    """
    kernel = _try_triton()
    if kernel is None or (not q.is_cuda) or q.numel() == 0:
        return False
    if k_cache.dtype != torch.float8_e4m3fn or v_cache.dtype != torch.float8_e4m3fn:
        return False
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        return False
    t, n_q, d = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    n_kv = int(k.shape[1])
    if k.shape[0] != t or v.shape[0] != t or v.shape[1] != n_kv:
        return False
    if d > 128 or d % 2 != 0:
        return False
    if int(cos_sin_cache.shape[-1]) != d:
        return False
    q_src = q
    # The kernel indexes via explicit (row, head) strides; only the head dim
    # must be contiguous. Strided views (e.g. slices of a fused qkv buffer)
    # run in place with zero copies.
    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    pos = positions.view(-1)[:t].to(dtype=torch.int64).contiguous()
    loc64 = loc.view(-1)[:t].to(dtype=torch.int64).contiguous()
    qw = q_weight.reshape(-1).to(device=q.device, dtype=q.dtype).contiguous()
    kw = k_weight.reshape(-1).to(device=q.device, dtype=q.dtype).contiguous()
    cs = cos_sin_cache.to(device=q.device, dtype=torch.float32).contiguous()
    if int(qw.numel()) != d or int(kw.numel()) != d:
        return False
    try:
        kernel[(t,)](
            q,
            k,
            v,
            qw,
            kw,
            pos,
            loc64,
            cs,
            k_cache,
            v_cache,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            v_cache.stride(0),
            v_cache.stride(1),
            n_q,
            n_kv,
            d,
            float(eps),
            BLOCK=max(d, 32),
            HALF=max(d // 2, 16),
            num_warps=4,
        )
    except Exception as exc:
        logger.debug("qk_rope_kv triton launch failed: %s", exc)
        return False
    if q_src.data_ptr() != q.data_ptr():
        q_src.copy_(q)
    return True
