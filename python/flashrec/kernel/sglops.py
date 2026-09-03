"""Optional sgl-kernel wrappers. Never import sglang."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_FP8_MAX = 448.0


def _try_import():
    try:
        import sgl_kernel

        return sgl_kernel
    except Exception as exc:
        logger.debug("sgl_kernel unavailable: %s", exc)
        return None


SGL = _try_import()

# sgl_kernel entry points go through pybind/DLPack; letting dynamo trace into
# them fails inductor fake-tensor propagation (aten.set_.source_Storage).
# Each public wrapper below is opaque to torch.compile.


@torch.compiler.disable
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if SGL is None or not x.is_cuda:
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        y = x_f * torch.rsqrt(var + eps)
        return (y * weight.float()).to(x.dtype)
    orig = x.shape
    hidden = int(weight.numel())
    x2 = x.reshape(-1, hidden).contiguous()
    w = weight.reshape(-1).to(device=x.device, dtype=x.dtype)
    out = SGL.rmsnorm(x2, w, eps)
    return out.view(orig)


@torch.compiler.disable
def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual += x`` then RMSNorm(residual). Mutates both when kernel is used."""
    if SGL is None or not x.is_cuda:
        residual = residual + x
        return rmsnorm(residual, weight, eps), residual
    hidden = int(weight.numel())
    x2 = x.reshape(-1, hidden).contiguous()
    r2 = residual.reshape(-1, hidden).contiguous()
    w = weight.reshape(-1).to(device=x.device, dtype=x.dtype)
    SGL.fused_add_rmsnorm(x2, r2, w, eps)
    return x2.reshape(x.shape), r2.reshape(residual.shape)


@torch.compiler.disable
def apply_rope_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
) -> bool:
    if SGL is None or not query.is_cuda:
        return False
    SGL.apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
    )
    return True


@torch.compiler.disable
def per_token_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    if SGL is not None and x2.is_cuda:
        q = torch.empty(x2.shape, dtype=torch.float8_e4m3fn, device=x2.device)
        s = torch.empty((x2.shape[0], 1), dtype=torch.float32, device=x2.device)
        SGL.sgl_per_token_quant_fp8(x2, q, s)
        return q, s
    x_f = x2.float()
    amax = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / _FP8_MAX).to(dtype=torch.float32)
    q = (x_f / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


@torch.compiler.disable
def fp8_scaled_mm(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """a: [M,K] fp8, b: [K,N] fp8, scale_a: [M,1], scale_b: [N]."""
    if SGL is not None and a_fp8.is_cuda:
        return SGL.fp8_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype, bias)
    out = torch._scaled_mm(
        a_fp8,
        b_fp8,
        scale_a=(
            scale_a.flatten()[:1]
            if scale_a.numel() == 1
            else torch.ones(1, dtype=torch.float32, device=a_fp8.device)
        ),
        scale_b=(
            scale_b.flatten()[:1]
            if scale_b.numel() == 1
            else torch.ones(1, dtype=torch.float32, device=a_fp8.device)
        ),
        out_dtype=out_dtype,
        use_fast_accum=True,
    )
    if isinstance(out, tuple):
        out = out[0]
    if scale_a.numel() > 1 or scale_b.numel() > 1:
        out = (
            out.float()
            * scale_a.to(dtype=torch.float32)
            * scale_b.view(1, -1).to(dtype=torch.float32)
        )
        out = out.to(out_dtype)
        if bias is not None:
            out = out + bias.to(dtype=out_dtype)
    elif bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out


@torch.compiler.disable
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU: ``silu(x[..., :d]) * x[..., d:]`` with ``d = last_dim // 2``."""
    d = int(x.shape[-1]) // 2
    if SGL is not None and x.is_cuda:
        out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
        SGL.silu_and_mul(x, out)
        return out
    return F.silu(x[..., :d]) * x[..., d:]


@torch.compiler.disable
def apply_rope_and_store_kv(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
) -> bool:
    """Fused RoPE + KV store when sgl-kernel supports it and dtypes match.

    FP8 KV plus bf16 activations cannot use the fused path (no k/v scale yet).
    Returns True if KV was written inside the rope kernel.
    """
    if SGL is None or not query.is_cuda:
        return False
    if key.dtype != k_cache.dtype or value.dtype != v_cache.dtype:
        return False
    rope = getattr(SGL, "apply_rope_with_cos_sin_cache_inplace", None)
    fused_cls = getattr(SGL, "FusedSetKVBufferArg", None)
    if fused_cls is None:
        try:
            from sgl_kernel.elementwise import FusedSetKVBufferArg as fused_cls
        except Exception:
            fused_cls = None
    if fused_cls is None or rope is None:
        return False
    try:
        loc64 = loc.view(-1).to(dtype=torch.int64)
        rope(
            positions=positions,
            query=query,
            key=key,
            head_size=int(head_size),
            cos_sin_cache=cos_sin_cache,
            fused_set_kv_buffer_arg=fused_cls(
                value=value,
                k_buffer=k_cache,
                v_buffer=v_cache,
                k_scale=None,
                v_scale=None,
                cache_loc=loc64,
            ),
        )
        return True
    except Exception as exc:
        logger.debug("fused rope+kv store fallback: %s", exc)
        return False


@torch.compiler.disable
def store_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if SGL is not None and k.is_cuda:
        try:
            SGL.set_kv_buffer_kernel(k_cache, v_cache, loc, k, v)
            return
        except Exception:
            pass
    k_cache[loc] = k
    v_cache[loc] = v
