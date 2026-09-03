"""Fused SwiGLU (silu-and-mul) + per-token FP8 quant."""

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
        logger.debug("silu_fp8 triton unavailable: %s", exc)
        return None

    @triton.jit
    def _silu_quant_kernel(
        x_ptr,
        y_ptr,
        scale_ptr,
        d,
        fp8_max,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        amax = 0.0
        col = 0
        while col < d:
            idx = col + offs
            mask = idx < d
            gate = tl.load(x_ptr + row * (2 * d) + idx, mask=mask, other=0.0).to(
                tl.float32
            )
            up = tl.load(x_ptr + row * (2 * d) + d + idx, mask=mask, other=0.0).to(
                tl.float32
            )
            silu = gate * (1.0 / (1.0 + tl.exp(-gate)))
            y = silu * up
            amax = tl.maximum(amax, tl.max(tl.where(mask, tl.abs(y), 0.0)))
            col += BLOCK
        amax = tl.maximum(amax, 1e-12)
        scale = amax / fp8_max
        tl.store(scale_ptr + row, scale)
        col = 0
        while col < d:
            idx = col + offs
            mask = idx < d
            gate = tl.load(x_ptr + row * (2 * d) + idx, mask=mask, other=0.0).to(
                tl.float32
            )
            up = tl.load(x_ptr + row * (2 * d) + d + idx, mask=mask, other=0.0).to(
                tl.float32
            )
            silu = gate * (1.0 / (1.0 + tl.exp(-gate)))
            q = (silu * up) / scale
            q = tl.minimum(tl.maximum(q, -fp8_max), fp8_max)
            tl.store(y_ptr + row * d + idx, q.to(tl.float8e4nv), mask=mask)
            col += BLOCK

    _KERNEL = _silu_quant_kernel
    _TRITON_OK = True
    return _KERNEL


def _fallback(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from flashrec.kernel.sglops import per_token_quant_fp8, silu_and_mul

    return per_token_quant_fp8(silu_and_mul(x))


@torch.compiler.disable
def silu_and_mul_per_token_quant_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``silu(x[..., :d]) * x[..., d:]`` then per-token e4m3. ``d = last_dim // 2``."""
    kernel = _try_triton()
    if kernel is None or (not x.is_cuda) or x.numel() == 0:
        return _fallback(x)
    orig = x.shape
    last = int(orig[-1])
    if last % 2 != 0:
        return _fallback(x)
    d = last // 2
    x2 = x.reshape(-1, last).contiguous()
    rows = int(x2.shape[0])
    y = torch.empty((rows, d), dtype=torch.float8_e4m3fn, device=x2.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x2.device)
    block = 1024
    while block < d and block < 2048:
        block *= 2
    block = min(block, 2048)
    try:
        kernel[(rows,)](
            x2,
            y,
            scale,
            d,
            _FP8_MAX,
            BLOCK=block,
            num_warps=8,
        )
    except Exception as exc:
        logger.debug("silu_fp8 triton launch failed: %s", exc)
        return _fallback(x)
    return y, scale
