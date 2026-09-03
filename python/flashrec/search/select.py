# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pure-tensor beam selection functions (no D2H sync, CUDA-graph-ready).

Contract: tensor-in / tensor-out with fixed output shapes for a given
(num_rows, num_candidates, beam_width) signature. The only scalar D2H the
caller needs is ``int(sel.num_survivors)`` / ``int(sel.num_finished)``
— two ``.item()`` calls versus the old per-step ``.tolist()`` over the full
candidate list.

Ported and adapted from sgl-project/sglang PR #31626 (beam_search/joint_select.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SelectResult:
    """Fixed-shape outputs of one beam expansion step.

    Survivor slots are valid in [0, num_survivors); finished slots in
    [0, num_finished); remaining slots hold zeros from the dump-slot
    scatter and must not be read.
    """

    next_tokens: torch.Tensor  # [beam_width] int64
    parent_idx: torch.Tensor  # [beam_width] int64, row index into the input frontier
    new_cum_logprobs: torch.Tensor  # [beam_width] float32
    num_survivors: torch.Tensor  # [] int64; < beam_width means the group must finish
    fin_tokens: torch.Tensor  # [num_candidates] int64
    fin_parent_idx: torch.Tensor  # [num_candidates] int64
    fin_cum_logprobs: torch.Tensor  # [num_candidates] float32
    num_finished: torch.Tensor  # [] int64


@dataclass
class FinalSelect:
    """Top-beam_width candidates of a length-terminated step (all finished)."""

    tokens: torch.Tensor  # [beam_width] int64
    parent_idx: torch.Tensor  # [beam_width] int64
    cum_logprobs: torch.Tensor  # [beam_width] float32


def _scatter_fixed(src: torch.Tensor, slot: torch.Tensor, size: int) -> torch.Tensor:
    """Fixed-shape compaction: element i lands at slot[i].

    Non-selected elements all target the dump slot ``size``, which is sliced
    away. Result shape is always [size].
    """
    buf = src.new_zeros(size + 1)
    buf.scatter_(0, slot, src)
    return buf[:size]


def _ranked_candidates(
    cum_logprobs: torch.Tensor,
    top_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    num_out: int,
    perturb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score all row × candidate extensions, return the top num_out sorted.

    ``perturb`` (same shape as ``top_logprobs``) is added for ranking only —
    Gumbel noise here yields sampling without replacement — while the returned
    ``cand_scores`` stay unperturbed so cumulative logprobs never absorb noise.

    Returns (cand_scores, parent_row, tokens), each shape [num_out].
    """
    num_candidates = top_logprobs.shape[1]
    scores = cum_logprobs.unsqueeze(1) + top_logprobs
    if perturb is None:
        cand_scores, cand_idx = scores.reshape(-1).topk(num_out, sorted=True)
    else:
        _, cand_idx = (scores + perturb).reshape(-1).topk(num_out, sorted=True)
        cand_scores = scores.reshape(-1).gather(0, cand_idx)
    parent = cand_idx // num_candidates
    tokens = top_tokens.reshape(-1).gather(0, cand_idx)
    return cand_scores, parent, tokens


def joint_select(
    cum_logprobs: torch.Tensor,  # [num_rows] float32, frontier cumulative logprobs
    top_logprobs: torch.Tensor,  # [num_rows, num_candidates] float32
    top_tokens: torch.Tensor,  # [num_rows, num_candidates] int64
    stop_token_ids: torch.Tensor,  # [num_stop] int64; empty tensor = no stop check
    beam_width: int,
    perturb: torch.Tensor | None = None,  # [num_rows, num_candidates] rank noise
) -> SelectResult:
    """One beam expansion step: pure tensor, no D2H, fixed output shapes.

    Walk the top num_candidates extensions in descending cumulative-logprob
    order. Stop-token candidates finish; non-stop candidates survive. A
    candidate is examined only while fewer than beam_width survivors precede
    it in score order.

    Args:
        cum_logprobs: [num_rows] cumulative logprobs of the current frontier.
        top_logprobs: [num_rows, num_candidates] per-row top-k step logprobs.
        top_tokens: [num_rows, num_candidates] corresponding token ids.
        stop_token_ids: token ids that end a beam; pass an empty tensor to
            skip stop checking (equivalent to ignore_eos).
        beam_width: number of survivors to keep.

    Returns:
        SelectResult with fixed shapes; read only up to num_survivors /
        num_finished to avoid garbage in unfilled slots.
    """
    num_candidates = top_logprobs.shape[1]
    k = beam_width

    cand_scores, parent, tokens = _ranked_candidates(
        cum_logprobs, top_logprobs, top_tokens, num_candidates, perturb=perturb
    )

    if stop_token_ids.numel() > 0:
        is_stop = torch.isin(tokens, stop_token_ids)
    else:
        is_stop = torch.zeros_like(tokens, dtype=torch.bool)

    non_stop = ~is_stop
    non_stop_rank = non_stop.long().cumsum(0)  # 1-based at non-stop positions
    survivor = non_stop & (non_stop_rank <= k)
    # Examined while fewer than k survivors strictly precede the candidate.
    examined = (non_stop_rank - non_stop.long()) < k
    finished = is_stop & examined
    fin_rank = finished.long().cumsum(0)

    surv_slot = torch.where(
        survivor, non_stop_rank - 1, torch.full_like(non_stop_rank, k)
    )
    fin_slot = torch.where(
        finished, fin_rank - 1, torch.full_like(fin_rank, num_candidates)
    )

    return SelectResult(
        next_tokens=_scatter_fixed(tokens, surv_slot, k),
        parent_idx=_scatter_fixed(parent, surv_slot, k),
        new_cum_logprobs=_scatter_fixed(cand_scores, surv_slot, k),
        num_survivors=survivor.long().sum(),
        fin_tokens=_scatter_fixed(tokens, fin_slot, num_candidates),
        fin_parent_idx=_scatter_fixed(parent, fin_slot, num_candidates),
        fin_cum_logprobs=_scatter_fixed(cand_scores, fin_slot, num_candidates),
        num_finished=finished.long().sum(),
    )


def select_final_topk(
    cum_logprobs: torch.Tensor,  # [num_rows] float32
    top_logprobs: torch.Tensor,  # [num_rows, num_candidates] float32
    top_tokens: torch.Tensor,  # [num_rows, num_candidates] int64
    beam_width: int,
    perturb: torch.Tensor | None = None,
) -> FinalSelect:
    """Length-terminated step: the best beam_width extensions all finish.

    No stop check needed; the caller decides this step is final deterministically
    (max_new_tokens reached).
    """
    cand_scores, parent, tokens = _ranked_candidates(
        cum_logprobs, top_logprobs, top_tokens, beam_width, perturb=perturb
    )
    return FinalSelect(tokens=tokens, parent_idx=parent, cum_logprobs=cand_scores)
