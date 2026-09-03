"""Restricted lm-head + log_softmax (no sampling)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


class RestrictedLMHead:
    def __init__(
        self,
        special_token_ids: Optional[List[int]],
        enabled: bool = True,
    ):
        self.ids = list(special_token_ids) if special_token_ids else None
        self.enabled = bool(enabled and self.ids)
        self._token_ids: Optional[torch.Tensor] = None
        self._weight: Optional[torch.Tensor] = None

    @property
    def token_ids(self) -> Optional[torch.Tensor]:
        return self._token_ids

    @property
    def num_tokens(self) -> Optional[int]:
        if self._token_ids is None:
            return None
        return int(self._token_ids.numel())

    @property
    def ready(self) -> bool:
        return bool(
            self.enabled and self._weight is not None and self._token_ids is not None
        )

    def bind(self, lm_weight: torch.Tensor) -> None:
        """Slice restricted rows once after weight load."""
        if lm_weight is None:
            return
        self._ensure(lm_weight.device, lm_weight)

    def _ensure(self, device: torch.device, lm_weight: torch.Tensor) -> None:
        if not self.enabled:
            return
        if self._token_ids is None or self._token_ids.device != device:
            vocab = int(lm_weight.shape[0])
            ids = self.ids or []
            if (not ids) or min(ids) < 0 or max(ids) >= vocab:
                raise ValueError(f"sid_token_range out of range for vocab {vocab}")
            self._token_ids = torch.tensor(ids, dtype=torch.long, device=device)
            self._weight = None
        if self._weight is None or self._weight.device != device:
            self._weight = torch.index_select(lm_weight, 0, self._token_ids)

    def _restricted_logprobs(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden.to(dtype=self._weight.dtype), self._weight)
        return F.log_softmax(logits.float(), dim=-1)

    def compute_into(
        self,
        hidden: torch.Tensor,
        lm_weight: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Write restricted logprobs into ``out`` [T, K] fp32.

        Requires a prior ``bind()``. Do not call ``_ensure`` / ``index_select``
        here: this path is captured inside a CUDA graph.
        """
        del lm_weight
        if self._weight is None or self._token_ids is None:
            raise RuntimeError("RestrictedLMHead.compute_into requires bind()")
        out.copy_(self._restricted_logprobs(hidden))
        return out

    def compute(
        self, hidden: torch.Tensor, lm_weight: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (logprobs [T, K or V], candidate_token_ids or None)."""
        device = hidden.device
        self._ensure(device, lm_weight)
        if self.enabled and self._weight is not None and self._token_ids is not None:
            return self._restricted_logprobs(hidden), self._token_ids
        logits = F.linear(hidden.to(dtype=lm_weight.dtype), lm_weight)
        return F.log_softmax(logits.float(), dim=-1), None
