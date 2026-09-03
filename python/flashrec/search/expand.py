"""Beam expansion / pruning (top-k accumulate + optional EOS joint select)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from flashrec.search.select import joint_select, select_final_topk
from flashrec.search.state import BeamSearchList
from flashrec.search.trie import BeamValidPathTrie

__all__ = [
    "joint_select",
    "select_final_topk",
    "init_from_prefill",
    "init_from_prefill_batch",
    "expand_step",
    "apply_temperature",
    "gumbel_like",
]


def apply_temperature(logprobs: torch.Tensor, temperature: float) -> torch.Tensor:
    """Renormalize log-softmax rows to ``log_softmax(logits / T)``.

    Exact when ``logprobs`` covers the full candidate vocabulary: log_softmax
    output differs from raw logits by a per-row constant, which a second
    log_softmax removes. ``T <= 0`` and ``T == 1`` are identity.
    """
    if temperature is None or temperature <= 0.0 or abs(temperature - 1.0) < 1e-6:
        return logprobs
    return F.log_softmax(logprobs.float() / float(temperature), dim=-1)


def gumbel_like(x: torch.Tensor) -> torch.Tensor:
    """Standard Gumbel(0, 1) noise; added to logprobs before top-k it turns
    deterministic selection into sampling without replacement."""
    eps = torch.finfo(torch.float32).tiny
    u = torch.rand_like(x, dtype=torch.float32).clamp_min(eps)
    # u in [tiny, 1) keeps both logs finite: -log(u) in (5.9e-8, 88].
    return -torch.log(-torch.log(u))


def _map_candidate_tokens(
    idx: torch.Tensor, candidate_token_ids: Optional[torch.Tensor]
) -> torch.Tensor:
    if isinstance(candidate_token_ids, torch.Tensor):
        return candidate_token_ids[idx]
    return idx


def init_from_prefill(
    logprobs: torch.Tensor,
    beam_width: int,
    beam_candidates: int,
    max_new_tokens: int,
    prompt_len: int,
    valid_path: Optional[BeamValidPathTrie] = None,
    candidate_token_ids: Optional[torch.Tensor] = None,
    temperature: float = 0.0,
) -> BeamSearchList:
    """Initialize a BeamSearchList from a single prefill logprob row ``[V]``.

    ``temperature > 0`` first rescales the full row to ``softmax(logits / T)``
    (exact — the row spans the whole candidate vocabulary), then samples the
    initial ``beam_width`` beams without replacement (Gumbel top-k) from the
    top ``beam_candidates`` pool instead of taking them deterministically.
    """
    if logprobs.dim() != 1:
        raise ValueError(f"expected 1D prefill logprobs, got {tuple(logprobs.shape)}")
    device = logprobs.device
    logprobs = apply_temperature(logprobs, temperature)
    topk = min(int(beam_candidates), int(logprobs.shape[0]), int(logprobs.numel()))
    if topk < beam_width:
        raise ValueError(f"topk={topk} < beam_width={beam_width}")
    vals, idx = logprobs.topk(topk, dim=0, sorted=True)
    tokens = _map_candidate_tokens(idx, candidate_token_ids)
    vals = vals.to(dtype=torch.float32)
    tokens = tokens.to(dtype=torch.int64)
    if valid_path is not None and valid_path.active:
        scores = valid_path.mask_candidates(
            None, tokens.unsqueeze(0), vals.unsqueeze(0), cur_len=0
        ).squeeze(0)
        vals, order = torch.topk(scores, k=topk, dim=0, largest=True, sorted=True)
        tokens = tokens[order]
    if temperature > 0.0:
        _, order = torch.topk(
            vals + gumbel_like(vals), k=beam_width, largest=True, sorted=True
        )
        sel_vals = vals.gather(0, order).contiguous()
        sel_tokens = tokens.gather(0, order).contiguous()
    else:
        sel_vals = vals[:beam_width].contiguous()
        sel_tokens = tokens[:beam_width].contiguous()

    bl = BeamSearchList()
    bl.completed = []
    bl.incomplete = bl.ensure_empty_stubs(beam_width)
    bl.prompt_len = int(prompt_len)
    bl.cum_logprobs = sel_vals
    bl.last_tokens = sel_tokens
    bl.init_token_ids(max_new_tokens, device=device)
    if bl.token_ids is not None and bl.token_ids.shape[0] >= beam_width:
        bl.token_ids[:beam_width, 0] = sel_tokens
        bl.cur_len = 1
        bl.dense_authoritative = True
    if valid_path is not None and valid_path.active and valid_path.mode == "trie":
        bl.node_ids = valid_path.bootstrap_node_ids(
            sel_tokens.view(beam_width, 1), beam_width, 1, device
        )
    return bl


def init_from_prefill_batch(
    logprobs: torch.Tensor,
    beam_width: int,
    beam_candidates: int,
    max_new_tokens: int,
    prompt_lens: Sequence[int],
    valid_path: Optional[BeamValidPathTrie] = None,
    candidate_token_ids: Optional[torch.Tensor] = None,
    temperature: float = 0.0,
) -> list[BeamSearchList]:
    """Batched ``init_from_prefill`` for a wave of B homogeneous prefills.

    ``logprobs`` is ``[B, V]``. One topk / mask / (gumbel) pass over the whole
    wave replaces B per-request passes; token buffers are one ``[B*bw, T]``
    allocation sliced into per-request views. Selection math is identical to
    the single-request path (same ops on a batch dim).
    """
    if logprobs.dim() != 2:
        raise ValueError(f"expected 2D wave logprobs, got {tuple(logprobs.shape)}")
    B = int(logprobs.shape[0])
    if B != len(prompt_lens):
        raise ValueError(f"logprobs rows {B} != prompt_lens {len(prompt_lens)}")
    device = logprobs.device
    logprobs = apply_temperature(logprobs, temperature)
    topk = min(int(beam_candidates), int(logprobs.shape[1]))
    if topk < beam_width:
        raise ValueError(f"topk={topk} < beam_width={beam_width}")
    vals, idx = logprobs.topk(topk, dim=1, sorted=True)
    tokens = _map_candidate_tokens(idx, candidate_token_ids)
    vals = vals.to(dtype=torch.float32)
    tokens = tokens.to(dtype=torch.int64)
    if valid_path is not None and valid_path.active:
        scores = valid_path.mask_candidates(None, tokens, vals, cur_len=0)
        vals, order = torch.topk(scores, k=topk, dim=1, largest=True, sorted=True)
        tokens = tokens.gather(1, order)
    if temperature > 0.0:
        _, order = torch.topk(
            vals + gumbel_like(vals), k=beam_width, dim=1, largest=True, sorted=True
        )
        sel_vals = vals.gather(1, order).contiguous()
        sel_tokens = tokens.gather(1, order).contiguous()
    else:
        sel_vals = vals[:, :beam_width].contiguous()
        sel_tokens = tokens[:, :beam_width].contiguous()

    max_new_tokens = max(int(max_new_tokens), 1)
    buf = torch.zeros(
        (B * beam_width, max_new_tokens), dtype=torch.int64, device=device
    )
    buf[:, 0] = sel_tokens.reshape(-1)
    node_ids_all = None
    if valid_path is not None and valid_path.active and valid_path.mode == "trie":
        node_ids_all = valid_path.bootstrap_node_ids(
            sel_tokens.reshape(B * beam_width, 1), B * beam_width, 1, device
        )
    out: list[BeamSearchList] = []
    for i in range(B):
        bl = BeamSearchList()
        bl.completed = []
        bl.incomplete = bl.ensure_empty_stubs(beam_width)
        bl.prompt_len = int(prompt_lens[i])
        bl.cum_logprobs = sel_vals[i]
        bl.last_tokens = sel_tokens[i]
        bl.token_ids = buf[i * beam_width : (i + 1) * beam_width]
        bl.cur_len = 1
        bl.dense_authoritative = True
        if node_ids_all is not None:
            bl.node_ids = node_ids_all[i * beam_width : (i + 1) * beam_width]
        out.append(bl)
    return out


def expand_step(
    beam_list: BeamSearchList,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    beam_width: int,
    valid_path: Optional[BeamValidPathTrie] = None,
    stop_token_ids: Optional[Sequence[int]] = None,
    ignore_eos: bool = True,
    will_finish: bool = False,
    temperature: float = 0.0,
) -> Optional[torch.Tensor]:
    """One decode expand. Returns relative parent indices, or None if finished.

    ``top_tokens`` / ``top_logprobs`` are ``[beam_width, topk]``.

    ``temperature > 0`` ranks candidates by Gumbel-perturbed scores (sampling
    the survivors without replacement) while cumulative logprobs stay exact.
    Temperature *scaling* of ``top_logprobs`` is the caller's job — rows here
    are already truncated to top-k, so renormalizing over them would be wrong.
    """
    if beam_list.cum_logprobs is None:
        raise RuntimeError("beam_list.cum_logprobs is not initialized")
    device = top_tokens.device
    cum = beam_list.cum_logprobs.to(device=device, dtype=torch.float32)
    lp = top_logprobs.to(dtype=torch.float32)
    tt = top_tokens.to(dtype=torch.int64)
    scores = cum.unsqueeze(1) + lp
    cur_len = beam_list.generated_len()
    if valid_path is not None and valid_path.active:
        scores = valid_path.mask_candidates(
            beam_list.token_ids,
            tt,
            scores,
            cur_len,
            node_ids=beam_list.node_ids,
        )

    perturb = gumbel_like(scores) if temperature > 0.0 else None

    stop_t = None
    if stop_token_ids and not ignore_eos:
        stop_t = torch.as_tensor(list(stop_token_ids), dtype=torch.int64, device=device)

    step_lp = scores - cum.unsqueeze(1)
    if will_finish:
        sel = select_final_topk(cum, step_lp, tt, beam_width, perturb=perturb)
        beam_list.expand_token_ids(sel.parent_idx, sel.tokens)
        beam_list.cum_logprobs = sel.cum_logprobs
        beam_list.last_tokens = sel.tokens
        if (
            valid_path is not None
            and valid_path.active
            and beam_list.node_ids is not None
        ):
            beam_list.node_ids = valid_path.advance_node_ids(
                beam_list.node_ids, sel.parent_idx, sel.tokens
            )
        return None

    use_joint = stop_t is not None and stop_t.numel() > 0 and not ignore_eos
    if use_joint:
        step_lp = scores - cum.unsqueeze(1)
        res = joint_select(cum, step_lp, tt, stop_t, beam_width, perturb=perturb)
        n_surv = int(res.num_survivors.item())
        n_fin = int(res.num_finished.item())
        if n_fin > 0:
            fin_p = res.fin_parent_idx[:n_fin]
            fin_tok = res.fin_tokens[:n_fin]
            fin_v = res.fin_cum_logprobs[:n_fin]
            # Materialize finished via a temporary expand on a copy of prefixes.
            tmp = BeamSearchList()
            tmp.token_ids = (
                beam_list.token_ids.clone() if beam_list.token_ids is not None else None
            )
            tmp.cur_len = beam_list.cur_len
            if tmp.token_ids is not None:
                tmp.expand_token_ids(fin_p, fin_tok)
                beam_list.completed.extend(tmp.sequences_from_token_ids(fin_v))
        if n_surv < beam_width:
            if n_surv <= 0:
                beam_list.incomplete = []
                return None
            parents = res.parent_idx[:n_surv]
            toks = res.next_tokens[:n_surv]
            vals = res.new_cum_logprobs[:n_surv]
            beam_list.expand_token_ids(parents, toks)
            beam_list.cum_logprobs = vals
            beam_list.last_tokens = toks
            beam_list.incomplete = beam_list.ensure_empty_stubs(n_surv)
            if valid_path is not None and beam_list.node_ids is not None:
                beam_list.node_ids = valid_path.advance_node_ids(
                    beam_list.node_ids, parents, toks
                )
            return parents.to(dtype=torch.int64)

        parents = res.parent_idx
        toks = res.next_tokens
        vals = res.new_cum_logprobs
        beam_list.expand_token_ids(parents, toks)
        beam_list.cum_logprobs = vals
        beam_list.last_tokens = toks
        beam_list.incomplete = beam_list.ensure_empty_stubs(beam_width)
        if valid_path is not None and beam_list.node_ids is not None:
            beam_list.node_ids = valid_path.advance_node_ids(
                beam_list.node_ids, parents, toks
            )
        return parents.to(dtype=torch.int64)

    # GenRec dense fast path: no EOS, keep beam_width survivors.
    flat_scores = scores.reshape(-1)
    k = min(int(beam_width), int(flat_scores.numel()))
    topk = int(tt.shape[1])
    if perturb is not None:
        _, idx = torch.topk(
            flat_scores + perturb.reshape(-1), k=k, largest=True, sorted=True
        )
        vals = flat_scores.gather(0, idx)
    else:
        vals, idx = torch.topk(flat_scores, k=k, largest=True, sorted=True)
    parents = idx // topk
    toks = tt.reshape(-1).gather(0, idx)
    beam_list.expand_token_ids(parents, toks)
    beam_list.cum_logprobs = vals
    beam_list.last_tokens = toks
    beam_list.incomplete = beam_list.ensure_empty_stubs(k)
    if valid_path is not None and valid_path.active and beam_list.node_ids is not None:
        beam_list.node_ids = valid_path.advance_node_ids(
            beam_list.node_ids, parents, toks
        )
    return parents.to(dtype=torch.int64)
