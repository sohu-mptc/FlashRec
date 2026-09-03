"""Linear + FP8 GEMM via sgl-kernel (fp8_scaled_mm + per-token quant)."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashrec.kernel.sglops import fp8_scaled_mm, per_token_quant_fp8

logger = logging.getLogger(__name__)

_FP8_MAX = 448.0


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype in (
        torch.float8_e4m3fn,
        getattr(torch, "float8_e5m2", torch.float8_e4m3fn),
    )


def quantize_weight_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel absmax quantize to e4m3. Returns (fp8 [N,K], scale [N])."""
    w = weight.float()
    amax = w.abs().amax(dim=-1).clamp(min=1e-12)
    scale = (amax / _FP8_MAX).to(dtype=torch.float32)
    q = (w / scale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def _channel_scale(
    scale: torch.Tensor, out_features: int, device: torch.device
) -> torch.Tensor:
    s = scale.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if s.numel() == 1:
        return s.expand(out_features).contiguous()
    if s.numel() != out_features:
        return s.reshape(out_features).contiguous()
    return s.contiguous()


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
        self.bias = (
            nn.Parameter(torch.empty(out_features), requires_grad=False)
            if bias
            else None
        )
        self._fp8_weight: Optional[torch.Tensor] = None  # [K, N] for sgl_kernel
        self._fp8_scale: Optional[torch.Tensor] = None  # [N]
        self._use_fp8 = False

    def load(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        quantize_fp8: bool = False,
        weight_scale: torch.Tensor | None = None,
    ):
        device = weight.device
        # NVFP4: packed uint8 [N, K/2] + UE8M0 block scale. Official ckpt only;
        # FlashRec will not requantize the current W8A8 FP8 checkpoint.
        if weight.dtype == torch.uint8 and weight_scale is not None:
            raise NotImplementedError(
                "NVFP4 Linear needs an official NVFP4 checkpoint "
                "(packed uint8 + block scale); online FP8→NVFP4 is not supported"
            )
        if _is_fp8(weight.dtype):
            w_nk = weight  # [N, K]
            if weight_scale is None:
                scale = torch.ones(
                    self.out_features, dtype=torch.float32, device=device
                )
            else:
                scale = _channel_scale(weight_scale, self.out_features, device)
            self._fp8_weight = w_nk.t()  # [K, N] column-major for cutlass
            self._fp8_scale = scale
            self._use_fp8 = True
        elif quantize_fp8:
            q, scale = quantize_weight_fp8(weight)
            self._fp8_weight = q.t()
            self._fp8_scale = scale.to(device=device)
            self._use_fp8 = True
        else:
            self.weight = nn.Parameter(weight, requires_grad=False)
            self._use_fp8 = False
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._use_fp8
            and self._fp8_weight is not None
            and self._fp8_scale is not None
        ):
            try:
                return self._fp8_forward(x)
            except Exception as exc:
                logger.debug("fp8 gemm fallback: %s", exc)
                w = self._fp8_weight.t().to(dtype=x.dtype) * self._fp8_scale.to(
                    device=x.device, dtype=x.dtype
                ).unsqueeze(-1)
                return F.linear(x, w, self.bias)
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)

    def forward_fp8(
        self,
        q: torch.Tensor,
        a_scale: torch.Tensor,
        orig_shape: tuple,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """GEMM from a pre-quantized activation. ``q`` is [M, K] e4m3."""
        return self._fp8_mm(q, a_scale, orig_shape, out_dtype)

    def _fp8_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x2 = x.reshape(-1, orig[-1]).contiguous()
        q, a_scale = per_token_quant_fp8(x2)
        out_dtype = (
            x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        )
        return self._fp8_mm(q, a_scale, orig, out_dtype)

    def _fp8_mm(
        self,
        q: torch.Tensor,
        a_scale: torch.Tensor,
        orig_shape: tuple,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        b = self._fp8_weight
        b_scale = self._fp8_scale
        assert b is not None and b_scale is not None
        device = q.device
        if b.device != device:
            b = b.to(device=device)
            self._fp8_weight = b
        if b_scale.device != device:
            b_scale = b_scale.to(device=device)
            self._fp8_scale = b_scale
        bias = self.bias.to(dtype=out_dtype) if self.bias is not None else None
        out = fp8_scaled_mm(q, b, a_scale, b_scale, out_dtype, bias)
        return out.view(*orig_shape[:-1], self.out_features)
