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
    "repeat_best_legal",
    "apply_temperature",
    "gumbel_like",
    "trie_candidate_width",
]

# Codebooks at or under this width are masked in full. A ``2 * beam`` window
# drops trie-legal children, and the following top-k then pads the beam with
# those illegal ids (catalog_miss). OneRec's 8192-wide levels stay narrow:
# their legal mass already lands inside ``2 * beam``.
_FULL_CODEBOOK_MASK_MAX = 4096


def trie_candidate_width(beam_width: int, vocab: int, constrain: bool) -> int:
    """How many logits to keep before the trie mask.

    ``constrain`` is the trie being active. Small codebooks are returned
    whole so every legal child is visible to the mask.
    """
    vocab = max(int(vocab), 1)
    bw = max(int(beam_width), 1)
    narrow = min(max(bw * 2, bw), vocab)
    if constrain and vocab <= _FULL_CODEBOOK_MASK_MAX:
        return vocab
    return narrow


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


# Per-(device, ids) cache for the EOS stop-token tensor: rebuilding it from a
# Python list is an H2D copy on every decode step of every request.
_STOP_T_CACHE: dict = {}


def _stop_tokens_tensor(
    stop_token_ids: Sequence[int], device: torch.device
) -> torch.Tensor:
    key = (str(device), tuple(int(t) for t in stop_token_ids))
    t = _STOP_T_CACHE.get(key)
    if t is None:
        t = torch.as_tensor(list(key[1]), dtype=torch.int64, device=device)
        _STOP_T_CACHE[key] = t
    return t


def repeat_best_legal(
    vals: torch.Tensor,
    rows: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Copy survivor 0 into slots whose score is not finite.

    A beam that has fewer legal trie children than its width used to keep the
    rejected token id in those slots (parent 0's unconstrained top token, or
    token 0). Every such slot decoded to the same illegal SID. Survivor 0 is
    the best legal child when any exist; scores are left unchanged.
    """
    finite = torch.isfinite(vals)
    dup = (~finite) & finite[..., :1]
    filled = []
    for row in rows:
        filled.append(torch.where(dup, row[..., :1].expand_as(row), row))
    return filled


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
    select_width: Optional[int] = None,
    coverage_tau: float = 0.0,
    coverage_n_min: Optional[int] = None,
) -> BeamSearchList:
    """Initialize a BeamSearchList from a single prefill logprob row ``[V]``.

    ``temperature > 0`` first rescales the full row to ``softmax(logits / T)``
    (exact — the row spans the whole candidate vocabulary), then samples the
    initial ``beam_width`` beams without replacement (Gumbel top-k) from the
    top ``beam_candidates`` pool instead of taking them deterministically.

    ``select_width`` (default ``beam_width``) caps the first-level survivors.
    With ``coverage_tau > 0`` and a trie ``leaf_count`` table, the survivors
    are the smallest ranked prefix covering ``tau`` of the reachable leaf
    mass, clamped to ``[coverage_n_min, select_width]``. The floor (not a cap:
    a hard first-level cap of n0=64 truncated n=128/512 beams below their own
    width and lost the correct first-level codes outright) defaults to 16.
    """
    if logprobs.dim() != 1:
        raise ValueError(f"expected 1D prefill logprobs, got {tuple(logprobs.shape)}")
    device = logprobs.device
    sw = int(select_width) if select_width is not None else int(beam_width)
    sw = max(min(sw, int(beam_width)), 1)
    logprobs = apply_temperature(logprobs, temperature)
    topk = trie_candidate_width(
        beam_width,
        int(logprobs.shape[0]),
        valid_path is not None and valid_path.active,
    )
    topk = min(topk, int(logprobs.shape[0]), int(logprobs.numel()))
    # Beam may exceed the SID candidate window (e.g. MiniOneRec n=1000 on a
    # 560-wide codebook); clamp rather than failing the request.
    sw = min(sw, topk)
    if sw < 1:
        raise ValueError(f"empty candidate window: topk={topk}")
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
            vals + gumbel_like(vals), k=sw, largest=True, sorted=True
        )
        sel_vals = vals.gather(0, order).contiguous()
        sel_tokens = tokens.gather(0, order).contiguous()
    else:
        take = sw
        if (
            coverage_tau > 0.0
            and valid_path is not None
            and valid_path.active
            and valid_path.mode == "trie"
            and valid_path.ensure_unary_tables(device)
            and valid_path._leaf_count is not None
        ):
            from flashrec.search.budget import coverage_select_count

            nodes = valid_path.bootstrap_node_ids(
                tokens.view(topk, 1), topk, 1, device
            )
            lc = valid_path._leaf_count.to(device=device)[
                nodes.clamp(0, valid_path._leaf_count.shape[0] - 1)
            ]
            floor = int(coverage_n_min) if coverage_n_min is not None else 16
            take = coverage_select_count(
                lc[:topk],
                tau=float(coverage_tau),
                n_min=min(max(floor, 1), sw),
                n_max=sw,
            )
        sel_vals = vals[:take].contiguous()
        sel_tokens = tokens[:take].contiguous()
        sw = take

    (sel_tokens,) = repeat_best_legal(sel_vals, (sel_tokens,))

    bl = BeamSearchList()
    bl.completed = []
    bl.incomplete = bl.ensure_empty_stubs(sw)
    bl.prompt_len = int(prompt_len)
    bl.cum_logprobs = sel_vals
    bl.last_tokens = sel_tokens
    bl.init_token_ids(max_new_tokens, device=device)
    if bl.token_ids is not None and bl.token_ids.shape[0] >= sw:
        bl.token_ids[:sw, 0] = sel_tokens
        bl.cur_len = 1
        bl.dense_authoritative = True
    if valid_path is not None and valid_path.active and valid_path.mode == "trie":
        bl.node_ids = valid_path.bootstrap_node_ids(
            sel_tokens.view(sw, 1), sw, 1, device
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
    select_width: Optional[int] = None,
    coverage_tau: float = 0.0,
    coverage_n_min: Optional[int] = None,
) -> list[BeamSearchList]:
    """Batched ``init_from_prefill`` for a wave of B homogeneous prefills.

    ``logprobs`` is ``[B, V]``. One topk / mask / (gumbel) pass over the whole
    wave replaces B per-request passes; token buffers are one ``[B*bw, T]``
    allocation sliced into per-request views. Selection math is identical to
    the single-request path (same ops on a batch dim), including the
    coverage-tau first-level prefix: it is computed for all B requests in one
    batched cumsum and one host sync, and requests keep ragged views of the
    rectangular buffers. Serving initializes through this path, so skipping
    tau here (as before) silently disabled the first-level budget in
    production.
    """
    if logprobs.dim() != 2:
        raise ValueError(f"expected 2D wave logprobs, got {tuple(logprobs.shape)}")
    B = int(logprobs.shape[0])
    if B != len(prompt_lens):
        raise ValueError(f"logprobs rows {B} != prompt_lens {len(prompt_lens)}")
    device = logprobs.device
    sw = int(select_width) if select_width is not None else int(beam_width)
    sw = max(min(sw, int(beam_width)), 1)
    logprobs = apply_temperature(logprobs, temperature)
    topk = trie_candidate_width(
        beam_width,
        int(logprobs.shape[1]),
        valid_path is not None and valid_path.active,
    )
    topk = min(topk, int(logprobs.shape[1]))
    sw = min(sw, topk)
    if sw < 1:
        raise ValueError(f"empty candidate window: topk={topk}")
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
            vals + gumbel_like(vals), k=sw, dim=1, largest=True, sorted=True
        )
        sel_vals = vals.gather(1, order).contiguous()
        sel_tokens = tokens.gather(1, order).contiguous()
    else:
        sel_vals = vals[:, :sw].contiguous()
        sel_tokens = tokens[:, :sw].contiguous()

    (sel_tokens,) = repeat_best_legal(sel_vals, (sel_tokens,))

    node_ids_all = None
    if valid_path is not None and valid_path.active and valid_path.mode == "trie":
        node_ids_all = valid_path.bootstrap_node_ids(
            sel_tokens.reshape(B * sw, 1), B * sw, 1, device
        )

    takes = [sw] * B
    if (
        temperature <= 0.0
        and coverage_tau > 0.0
        and node_ids_all is not None
        and valid_path.ensure_unary_tables(device)
        and valid_path._leaf_count is not None
    ):
        lc_table = valid_path._leaf_count.to(device=device)
        n_leaf = int(lc_table.shape[0])
        nodes = node_ids_all.view(B, sw)
        lc = lc_table[nodes.clamp(0, max(n_leaf - 1, 0))].to(dtype=torch.float32)
        lc = torch.where((nodes >= 0) & (nodes < n_leaf), lc, torch.zeros_like(lc))
        total = lc.sum(dim=1, keepdim=True)
        csum = lc.cumsum(dim=1)
        kvec = (csum < float(coverage_tau) * total).sum(dim=1).to(torch.int64) + 1
        kvec = torch.where(total.squeeze(1) > 0, kvec, torch.full_like(kvec, sw))
        floor = int(coverage_n_min) if coverage_n_min is not None else 16
        kvec = kvec.clamp(min(max(floor, 1), sw), sw)
        # Round up to a multiple of 32 so requests that shrank alike stay in
        # one lock-step decode group (see expand_group_tbb).
        kvec = torch.minimum((kvec + 31) // 32 * 32, torch.full_like(kvec, sw))
        takes = [int(x) for x in kvec.cpu().tolist()]

    max_new_tokens = max(int(max_new_tokens), 1)
    buf = torch.zeros(
        (B * sw, max_new_tokens), dtype=torch.int64, device=device
    )
    buf[:, 0] = sel_tokens.reshape(-1)
    out: list[BeamSearchList] = []
    for i in range(B):
        k = takes[i]
        bl = BeamSearchList()
        bl.completed = []
        bl.incomplete = bl.ensure_empty_stubs(k)
        bl.prompt_len = int(prompt_lens[i])
        bl.cum_logprobs = sel_vals[i, :k]
        bl.last_tokens = sel_tokens[i, :k]
        bl.token_ids = buf[i * sw : i * sw + k]
        bl.cur_len = 1
        bl.dense_authoritative = True
        if node_ids_all is not None:
            bl.node_ids = node_ids_all[i * sw : i * sw + k]
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
    skip_mask: bool = False,
) -> Optional[torch.Tensor]:
    """One decode expand. Returns relative parent indices, or None if finished.

    ``top_tokens`` / ``top_logprobs`` are ``[beam_width, topk]``.

    ``temperature > 0`` ranks candidates by Gumbel-perturbed scores (sampling
    the survivors without replacement) while cumulative logprobs stay exact.
    Temperature *scaling* of ``top_logprobs`` is the caller's job — rows here
    are already truncated to top-k, so renormalizing over them would be wrong.

    ``skip_mask=True`` assumes ``top_tokens`` were scored only over legal
    neighbors (sparse path) so re-running ``mask_candidates`` would be a no-op
    gather over the full codebook vocab.
    """
    if beam_list.cum_logprobs is None:
        raise RuntimeError("beam_list.cum_logprobs is not initialized")
    device = top_tokens.device
    cum = beam_list.cum_logprobs.to(device=device, dtype=torch.float32).reshape(-1)
    lp = top_logprobs.to(dtype=torch.float32)
    tt = top_tokens.to(dtype=torch.int64)
    # Trie-budget / unary shrink live_width below the graph-padded beam, so
    # logprob rows can outnumber survivors. Score only the live prefix.
    n_live = min(int(cum.numel()), int(lp.shape[0]), int(tt.shape[0]))
    if n_live <= 0:
        return None
    if int(cum.numel()) != n_live:
        cum = cum[:n_live]
    if int(lp.shape[0]) != n_live:
        lp = lp[:n_live]
        tt = tt[:n_live]
    scores = cum.unsqueeze(1) + lp
    cur_len = beam_list.generated_len()
    if (
        not skip_mask
        and valid_path is not None
        and valid_path.active
    ):
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
        stop_t = _stop_tokens_tensor(stop_token_ids, device)

    if will_finish:
        # step_lp only exists on the finish/EOS branches; the no-EOS fast
        # path below never reads it, so computing it unconditionally would
        # burn one [n, topk] kernel + allocation per decode step.
        step_lp = scores - cum.unsqueeze(1)
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
    parents, toks = repeat_best_legal(vals, (parents, toks))
    beam_list.expand_token_ids(parents, toks)
    beam_list.cum_logprobs = vals
    beam_list.last_tokens = toks
    beam_list.incomplete = beam_list.ensure_empty_stubs(k)
    if valid_path is not None and valid_path.active and beam_list.node_ids is not None:
        beam_list.node_ids = valid_path.advance_node_ids(
            beam_list.node_ids, parents, toks
        )
    return parents.to(dtype=torch.int64)
