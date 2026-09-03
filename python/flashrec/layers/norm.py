"""RMSNorm via sgl-kernel, with fused residual add."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from flashrec.kernel.sglops import fused_add_rmsnorm, rmsnorm


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.eps = float(eps)
        self.hidden_size = int(hidden_size)

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is None:
            return rmsnorm(x, self.weight, self.eps)
        y, residual = fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return y, residual

    def quant_fp8(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """RMSNorm then per-token e4m3. With residual: fused add+norm+quant."""
        from flashrec.kernel.rms_fp8 import (
            fused_add_rmsnorm_per_token_quant_fp8,
            rmsnorm_per_token_quant_fp8,
        )

        if residual is None:
            return rmsnorm_per_token_quant_fp8(x, self.weight, self.eps)
        return fused_add_rmsnorm_per_token_quant_fp8(x, residual, self.weight, self.eps)
