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
        self._level_weights: Optional[List[torch.Tensor]] = None
        self._level_token_ids: Optional[List[torch.Tensor]] = None
        self._codebook_sizes: Optional[List[int]] = None

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

    @property
    def per_codebook_ready(self) -> bool:
        return bool(self._level_weights is not None and len(self._level_weights) > 0)

    @property
    def num_levels(self) -> int:
        return len(self._level_weights) if self._level_weights else 0

    @property
    def max_codebook_k(self) -> int:
        if not self._codebook_sizes:
            return 0
        return max(self._codebook_sizes)

    def codebook_k(self, level: int) -> int:
        if not self._codebook_sizes or level < 0 or level >= len(self._codebook_sizes):
            return 0
        return self._codebook_sizes[level]

    def bind(self, lm_weight: torch.Tensor) -> None:
        """Slice restricted rows once after weight load."""
        if lm_weight is None:
            return
        self._ensure(lm_weight.device, lm_weight)

    def bind_codebook(self, codebook_sizes: List[int]) -> None:
        """Pre-slice per-level weight views from the already-bound ``_weight``.

        Each level's weight is a zero-copy ``narrow()`` view.  Must be called
        after ``bind()``.
        """
        if self._weight is None or not self.enabled:
            return
        total = sum(codebook_sizes)
        if total != self._weight.shape[0]:
            return
        self._codebook_sizes = list(codebook_sizes)
        self._level_weights = []
        self._level_token_ids = []
        offset = 0
        ids = self.ids or []
        device = self._weight.device
        for size in codebook_sizes:
            self._level_weights.append(self._weight.narrow(0, offset, size))
            self._level_token_ids.append(
                torch.tensor(ids[offset : offset + size], dtype=torch.long, device=device)
            )
            offset += size

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
            ids = self.ids or []
            lo, hi = int(ids[0]), int(ids[-1])
            contiguous = (
                len(ids) > 1
                and hi - lo == len(ids) - 1
                and all(b - a == 1 for a, b in zip(ids, ids[1:]))
            )
            if contiguous:
                # SID ids form one contiguous range (sid_layout enforces it):
                # a zero-copy narrow() view avoids duplicating K x H weight
                # rows (~hundreds of MB at catalog scale) for process life.
                self._weight = lm_weight.narrow(0, lo, len(ids))
            else:
                self._weight = torch.index_select(lm_weight, 0, self._token_ids)

    def _restricted_logprobs(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden.to(dtype=self._weight.dtype), self._weight)
        return F.log_softmax(logits.float(), dim=-1)

    def compute_level(
        self, hidden: torch.Tensor, level: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-codebook logprobs: GEMM only against level's weight slice."""
        weights = self._level_weights
        ids = self._level_token_ids
        if (
            weights is None
            or ids is None
            or level < 0
            or level >= len(weights)
        ):
            # Past last codebook (max_tokens > SID depth): full restricted head.
            if self._weight is None or self._token_ids is None:
                raise RuntimeError(
                    "RestrictedLMHead.compute_level requires bind_codebook()"
                )
            return self._restricted_logprobs(hidden), self._token_ids
        w = weights[level]
        logits = F.linear(hidden.to(dtype=w.dtype), w)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        return logprobs, ids[level]

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
