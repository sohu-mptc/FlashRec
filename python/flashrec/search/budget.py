"""Trie-budget beam: coverage prefix selection, width schedule, leaf flush."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from flashrec.core import FinishReason
from flashrec.search.state import BeamSearchList

__all__ = [
    "coverage_select_count",
    "expand_group_tbb",
    "target_select_k",
    "try_leaf_flush",
]


def coverage_select_count(
    leaf_counts: torch.Tensor,
    *,
    tau: float,
    n_min: int,
    n_max: int,
) -> int:
    """Smallest prefix of score-sorted ``leaf_counts`` covering ``tau`` of mass."""
    n = int(leaf_counts.numel())
    if n <= 0:
        return 0
    n_max = max(min(int(n_max), n), 1)
    n_min = max(min(int(n_min), n_max), 1)
    if tau <= 0.0:
        return n_max
    lc = leaf_counts.to(dtype=torch.float32).clamp(min=0)
    total = float(lc.sum().item())
    if total <= 0.0:
        return n_max
    csum = lc.cumsum(0)
    hit = (csum >= float(tau) * total).nonzero(as_tuple=False)
    if hit.numel() == 0:
        return n_max
    n_hit = int(hit[0].item()) + 1
    return max(n_min, min(n_hit, n_max))


def target_select_k(
    beam_width: int,
    generated_len: int,
    remain: int,
    n0: int,
    n_max: int,
) -> int:
    """Upper bound on survivors for the next expand.

    This is a cap, not a schedule: adaptive shrinking is the coverage-tau
    prefix (model-ranked), applied by the caller. The old form returned ``n0``
    on the first decode step, which silently truncated wide beams (n=128/512)
    to 64 model-unranked survivors and capped recall at the first level; ``n0``
    is now the coverage *floor* instead (see ``expand_group_tbb`` callers and
    ``init_from_prefill``).
    """
    bw = max(int(beam_width), 1)
    cap = int(n_max) if int(n_max) > 0 else bw
    cap = max(min(cap, bw), 1)
    if int(remain) <= 1:
        return bw
    return cap


def expand_group_tbb(
    bls: Sequence[BeamSearchList],
    scores,
    *,
    live: int,
    select_k: int,
    tau: float,
    n_min: int,
    leaf_count: Optional[torch.Tensor],
) -> Optional[List[torch.Tensor]]:
    """One batched select + expand for a lock-step group of beam lists.

    ``scores`` is a ``TrieBudgetScores`` over the group's packed ``[N * live,
    D]`` rows (rows sorted descending, dead tail at ``-inf``), as produced by
    one ``score_neighbors`` call over the concatenated frontier. Every list
    must hold ``live`` beams at the same ``cur_len`` with token tables of one
    width. The caller checks that; this function assumes it.

    Selection is the same math as the per-request ``expand_step`` fast path —
    global top ``select_k`` of ``cum + logprob`` per request — but batched:
    one ``topk`` / gather chain for the whole group instead of ~10 kernels per
    request. Next-node ids come from the children kernel A already gathered,
    so there is no ``advance_node_ids`` table lookup.

    ``tau > 0`` with a ``leaf_count`` table additionally applies the per-step
    coverage budget: each request keeps the smallest ranked prefix whose
    children cover ``tau`` of the surviving leaf mass, clamped to
    ``[n_min, select_k]``. This costs one host sync of an ``[N]`` int tensor
    for the whole group (the surviving widths shape next step's tensors).
    With ``tau <= 0`` the width is ``select_k`` and nothing syncs.

    Returns per-request relative parent indices for the KV remap, or ``None``
    when the group cannot be batched (caller falls back to the per-request
    loop).
    """
    n = len(bls)
    if n == 0 or live <= 0:
        return None
    d = int(scores.logprobs.shape[1])
    if int(scores.logprobs.shape[0]) != n * live:
        return None
    col = int(bls[0].cur_len)
    width = int(bls[0].token_ids.shape[1])
    if col >= width:
        return None

    cum = torch.stack([bl.cum_logprobs for bl in bls])  # [N, live] fp32
    lp = scores.logprobs.view(n, live, d)
    # Rows are sorted descending, so a global top-k per request never needs
    # more than k candidates from any one row: cap the flat plane at
    # min(d, select_k) columns instead of the request's beam_candidates.
    cols = max(min(d, int(select_k)), 1)
    flat = (cum.unsqueeze(2) + lp[:, :, :cols]).reshape(n, live * cols)
    k = min(int(select_k), live * cols)
    if k <= 0:
        return None
    vals, idx = torch.topk(flat, k=k, dim=1, largest=True, sorted=True)
    parents = idx // cols
    # Map flat-over-cols indices back onto the full [live * d] plane so tokens
    # and children gather without materializing sliced copies.
    idx_full = parents * d + (idx - parents * cols)
    tokens = scores.tokens.view(n, live * d).gather(1, idx_full)
    children = scores.children.view(n, live * d).gather(1, idx_full)

    if tau > 0.0 and leaf_count is not None and int(leaf_count.numel()) > 0:
        n_leaf = int(leaf_count.numel())
        safe = children.clamp(0, n_leaf - 1)
        w = leaf_count.to(device=children.device, dtype=torch.float32)[safe]
        w = torch.where((children >= 0) & (children < n_leaf), w, torch.zeros_like(w))
        total = w.sum(dim=1, keepdim=True)
        csum = w.cumsum(dim=1)
        # Smallest prefix with csum >= tau * total == count(csum < want) + 1;
        # matches coverage_select_count / the tbb_select kernel.
        kvec = (csum < float(tau) * total).sum(dim=1).to(dtype=torch.int64) + 1
        kvec = torch.where(
            total.squeeze(1) > 0, kvec, torch.full_like(kvec, k)
        )
        lo = min(max(int(n_min), 1), k)
        kvec = kvec.clamp(lo, k)
        # Round surviving widths up to a multiple of 32: keeps a few extra
        # beams (never fewer), and requests that shrank alike stay in one
        # lock-step group instead of fragmenting into per-width groups.
        kvec = torch.minimum(
            (kvec + 31) // 32 * 32, torch.full_like(kvec, k)
        )
        ks = [int(x) for x in kvec.cpu().tolist()]
    else:
        ks = [k] * n

    tables = torch.stack([bl.token_ids for bl in bls])  # [N, live?, L]
    gathered = tables.gather(
        1, parents.unsqueeze(-1).expand(n, k, width)
    )
    gathered[:, :, col] = tokens

    out_parents: List[torch.Tensor] = []
    for i, bl in enumerate(bls):
        ki = ks[i]
        bl.token_ids = gathered[i, :ki]
        bl.cur_len = col + 1
        bl.dense_authoritative = True
        bl.cum_logprobs = vals[i, :ki]
        bl.last_tokens = tokens[i, :ki]
        bl.node_ids = children[i, :ki]
        bl.incomplete = bl.ensure_empty_stubs(ki)
        out_parents.append(parents[i, :ki])
    return out_parents


def try_leaf_flush(
    req,
    valid_path,
    limit: int,
    flush_leaves: int,
) -> bool:
    """Enumerate remaining SID suffixes when every live node is a small tree.

    Fires only when the enumeration cannot change the beam's membership:
    every emitted path is a complete length-``remain`` suffix (no token-0
    padding, no truncated non-leaves — those decoded to invalid SIDs), and the
    total number of completions fits inside ``beam_width`` (the old
    256-per-node / ``4 * beam_width`` mid-walk caps kept whichever leaves the
    DFS reached first and dropped model-reachable items). Ranking among
    same-parent leaves still ties on the parent score — which is why this stays
    opt-in (``trie_flush_leaves > 0``).
    """
    if req is None or bool(getattr(req, "finished", False)):
        return False
    if int(flush_leaves) <= 0:
        return False
    if float(getattr(req, "temperature", 0.0) or 0.0) > 0.0:
        return False
    if valid_path is None or not valid_path.active or valid_path.mode != "trie":
        return False
    bl: Optional[BeamSearchList] = getattr(req, "beam_list", None)
    if bl is None or bl.node_ids is None or bl.token_ids is None:
        return False
    remain = int(limit) - int(bl.generated_len())
    if remain < 2:
        return False
    if not valid_path.ensure_unary_tables(bl.node_ids.device):
        return False
    lc = valid_path._leaf_count
    nodes = bl.node_ids.reshape(-1).to(dtype=torch.int64)
    n_nodes = int(lc.shape[0])
    valid = (nodes >= 0) & (nodes < n_nodes)
    # One sync for both gates: every node in range, and no node with more than
    # flush_leaves leaves below it. Two separate .item() calls stalled twice per
    # request per step even when the flush never fires.
    safe = nodes.clamp(0, max(n_nodes - 1, 0))
    counts = lc.to(device=nodes.device)[safe]
    ok = valid.all() & (counts <= int(flush_leaves)).all()
    if not bool(ok.item()):
        return False
    # The CSR edge tables live on GPU once _place_unary_tables has run, and
    # pulling them back here cost 3 x 12MB of pageable DtoH per flush at a 1.5M
    # SID catalog. They are immutable after _build_unary_tables, so prefer the
    # host lists it kept; fall back to a copy for tries built another way.
    rowptr = valid_path._edge_rowptr_host
    if rowptr is None:
        rowptr = valid_path._edge_rowptr.detach().cpu().tolist()
    etok = valid_path._edge_token_host
    if etok is None:
        etok = valid_path._edge_token.detach().cpu().tolist()
    echild = valid_path._edge_child_host
    if echild is None:
        echild = valid_path._edge_child.detach().cpu().tolist()
    node_list = nodes.detach().cpu().tolist()
    bw = int(getattr(req, "beam_width", len(node_list)) or len(node_list))
    cum = bl.cum_logprobs
    parent_scores = (
        cum.detach().cpu().tolist()
        if cum is not None
        else [0.0] * len(node_list)
    )
    prefix = bl.token_ids[:, : bl.cur_len].detach().cpu().tolist()
    paths: List[Tuple[float, List[int]]] = []
    for i, nid in enumerate(node_list):
        score = float(parent_scores[i]) if i < len(parent_scores) else 0.0
        base = list(prefix[i]) if i < len(prefix) else []
        for extra in _enumerate_paths(int(nid), remain, rowptr, etok, echild, n_nodes):
            paths.append((score, base + extra))
        if len(paths) > bw:
            # More completions than the beam can hold: truncating here would
            # drop reachable items in DFS order, so run the model instead.
            return False
    if not paths:
        return False
    paths.sort(key=lambda x: -x[0])
    device = bl.token_ids.device
    width = max(len(p[1]) for p in paths)
    # Build the padded plane on the host and ship it in one HtoD. The per-row
    # torch.tensor(...) form issued two pageable copies per surviving path
    # (1475 of them in a 12-prompt cascade trace) on the critical path.
    rows = [t + [0] * (width - len(t)) for _, t in paths]
    buf = torch.tensor(rows, dtype=torch.int64, device=device)
    scores = torch.tensor(
        [sc for sc, _ in paths], dtype=torch.float32, device=device
    )
    bl.token_ids = buf
    bl.cur_len = width
    bl.cum_logprobs = scores
    bl.last_tokens = buf[:, -1].contiguous()
    bl.node_ids = torch.full(
        (len(paths),), -1, dtype=torch.int64, device=device
    )
    bl.dense_authoritative = True
    bl.incomplete = bl.ensure_empty_stubs(len(paths))
    req.finished = True
    req.finish_reason = FinishReason(
        type="length", length=int(getattr(req, "max_new_tokens", limit) or limit)
    )
    return True


def _enumerate_paths(
    nid: int,
    remain: int,
    rowptr: Sequence[int],
    etok: Sequence[int],
    echild: Sequence[int],
    n_nodes: int,
) -> List[List[int]]:
    """Complete length-``remain`` suffixes below ``nid``; dead ends yield none.

    The old form returned ``[[]]`` for dead ends and invalid children, which
    surfaced short paths the caller padded with token 0 — decoded, those were
    unique invalid SIDs (measured invalid_rate 0.23-0.39 at n=128/512).
    """
    if remain <= 0:
        return [[]]
    if nid < 0 or nid >= len(rowptr) - 1:
        return []
    start, end = int(rowptr[nid]), int(rowptr[nid + 1])
    if start >= end:
        return []
    out: List[List[int]] = []
    for e in range(start, end):
        tok = int(etok[e]) if e < len(etok) else 0
        child = int(echild[e]) if e < len(echild) else -1
        for tail in _enumerate_paths(
            child, remain - 1, rowptr, etok, echild, n_nodes
        ):
            out.append([tok] + tail)
    return out
