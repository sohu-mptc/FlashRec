"""Fused RMSNorm + per-token FP8 quant (Triton), matching sgl-kernel scales."""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_FP8_MAX = 448.0
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
        logger.debug("rms_fp8 triton unavailable: %s", exc)
        return None

    @triton.jit
    def _rms_quant_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        y_ptr,
        scale_ptr,
        hidden,
        eps,
        fp8_max,
        BLOCK: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < hidden
        x = tl.load(x_ptr + row * hidden + offs, mask=mask, other=0.0)
        y = x.to(tl.float32)
        if HAS_RESIDUAL:
            r = tl.load(residual_ptr + row * hidden + offs, mask=mask, other=0.0)
            y = y + r.to(tl.float32)
            tl.store(residual_ptr + row * hidden + offs, y.to(r.dtype), mask=mask)
        var = tl.sum(y * y, axis=0) / hidden
        rstd = tl.rsqrt(var + eps)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = y * rstd * w
        amax = tl.max(tl.abs(y), axis=0)
        amax = tl.maximum(amax, 1e-12)
        scale = amax / fp8_max
        q = y / scale
        q = tl.minimum(tl.maximum(q, -fp8_max), fp8_max)
        tl.store(y_ptr + row * hidden + offs, q.to(tl.float8e4nv), mask=mask)
        tl.store(scale_ptr + row, scale)

    _KERNEL = _rms_quant_kernel
    _TRITON_OK = True
    return _KERNEL


def _fallback_quant(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from flashrec.kernel.sglops import per_token_quant_fp8

    return per_token_quant_fp8(hidden)


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor],
) -> Optional[tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    kernel = _try_triton()
    if kernel is None or (not x.is_cuda) or x.numel() == 0:
        return None
    orig = x.shape
    hidden = int(weight.numel())
    x2 = x.reshape(-1, hidden).contiguous()
    w = weight.reshape(-1).to(device=x.device, dtype=x.dtype)
    rows = int(x2.shape[0])
    if residual is not None:
        r2 = residual.reshape(-1, hidden).contiguous()
        if r2.shape != x2.shape:
            return None
    else:
        r2 = None
    y = torch.empty(x2.shape, dtype=torch.float8_e4m3fn, device=x2.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x2.device)
    block = 1
    while block < hidden:
        block *= 2
    try:
        kernel[(rows,)](
            x2,
            r2 if r2 is not None else x2,
            w,
            y,
            scale,
            hidden,
            float(eps),
            _FP8_MAX,
            BLOCK=block,
            HAS_RESIDUAL=r2 is not None,
            num_warps=4 if block <= 1024 else 8,
        )
    except Exception as exc:
        logger.debug("rms_fp8 triton launch failed: %s", exc)
        return None
    res_out = r2.reshape(orig) if r2 is not None else None
    return y, scale, res_out


@torch.compiler.disable
def rmsnorm_per_token_quant_fp8(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    launched = _launch(x, weight, eps, residual=None)
    if launched is not None:
        y, scale, _ = launched
        return y, scale
    from flashrec.kernel.sglops import rmsnorm

    return _fallback_quant(rmsnorm(x, weight, eps))


@torch.compiler.disable
def fused_add_rmsnorm_per_token_quant_fp8(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    launched = _launch(x, weight, eps, residual=residual)
    if launched is not None and launched[2] is not None:
        y, scale, res = launched
        return y, scale, res
    from flashrec.kernel.sglops import fused_add_rmsnorm

    hidden, residual = fused_add_rmsnorm(x, residual, weight, eps)
    q, scale = _fallback_quant(hidden)
    return q, scale, residual
