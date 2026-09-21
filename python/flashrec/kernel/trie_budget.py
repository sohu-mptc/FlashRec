"""Trie-Budget Beam: fused sparse scoring + budgeted beam selection.

Two device kernels replacing the Python decode chain that
``scheduler._try_sparse_expand`` runs per request per step:

``trie_budget_score``
    CSR neighbor gather + log-softmax over legal children + per-row descending
    sort, reading ``lm_head`` weight rows in place. The PyTorch path here
    materializes ``W[tokens]`` as ``[R, D, H]``; the kernel does not, which is
    what makes wide out-degrees affordable.

``trie_budget_select``
    Sorted-list merge to the global top-k, coverage-tau budget over child
    ``leaf_count``, token-table expand, and next-node write in one block per
    request. The surviving width comes back in a device tensor, so the decode
    step needs no ``.item()``.

As elsewhere in this package the PyTorch fallbacks are the source of
correctness; the CUDA module is optional and JIT-loaded.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "TrieBudgetScores",
    "TrieBudgetSelection",
    "trie_budget_score",
    "trie_budget_select",
    "try_load_trie_budget",
]

_NEG_INF = float("-inf")
_MODULE = None
_MODULE_TRIED = False

# Mirrors kTbbMaxDeg / kTbbSelectBlock in csrc/trie_budget.cuh. Above these the
# CUDA path is skipped and the PyTorch reference runs instead.
MAX_DEG = 2048
MAX_BEAM = 1024


class TrieBudgetScores(NamedTuple):
    """Per-row candidates, sorted descending by logprob within each row."""

    logprobs: torch.Tensor  # [R, D] fp32, -inf past the row's out-degree
    tokens: torch.Tensor  # [R, D] int64
    children: torch.Tensor  # [R, D] int64 next node id (invalid_node when unused)
    deg: torch.Tensor  # [R] int32 true out-degree


class TrieBudgetSelection(NamedTuple):
    """Survivors per request. Only the first ``k[n]`` columns are meaningful."""

    vals: torch.Tensor  # [N, BW] fp32 new cumulative logprobs
    parents: torch.Tensor  # [N, BW] int64 parent row, relative to the request
    tokens: torch.Tensor  # [N, BW] int64
    k: torch.Tensor  # [N] int32 surviving width (device-side, no sync)
    token_ids: Optional[torch.Tensor]  # [N, BW, L] int64
    node_ids: torch.Tensor  # [N, BW] int64


def _jit_include_dirs():
    from pathlib import Path

    inc = Path(__file__).resolve().parent / "include"
    return [str(inc)] if inc.is_dir() else []


def _try_module():
    global _MODULE, _MODULE_TRIED
    if _MODULE_TRIED:
        return _MODULE
    _MODULE_TRIED = True
    if not torch.cuda.is_available():
        logger.info("trie_budget JIT fallback (no CUDA)")
        return None
    try:
        from pathlib import Path

        from tvm_ffi.cpp import load_inline

        cuh = Path(__file__).resolve().parent / "csrc" / "trie_budget.cuh"
        if not cuh.is_file():
            logger.info("trie_budget JIT fallback (missing %s)", cuh.name)
            return None
        src = f'#include "{cuh}"\n'
        wrappers = [
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trie_budget_score, (trie_budget_score));",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trie_budget_select, (trie_budget_select));",
        ]
        includes = _jit_include_dirs()
        cflags = ["-std=c++20", "-O3", "--expt-relaxed-constexpr", "--use_fast_math"]
        cflags.extend(f"-I{p}" for p in includes)
        kwargs = dict(
            cuda_sources=[src + "\n".join(wrappers)],
            extra_cuda_cflags=cflags,
        )
        try:
            _MODULE = load_inline(
                "beamrec_trie_budget", extra_include_paths=includes, **kwargs
            )
        except TypeError:
            _MODULE = load_inline("beamrec_trie_budget", **kwargs)
        logger.info("flashrec fused trie_budget JIT loaded")
    except Exception as exc:
        logger.warning("trie_budget JIT fallback (PyTorch): %s", exc)
        _MODULE = None
    return _MODULE


def _c(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


def trie_budget_score(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    node_ids: torch.Tensor,
    edge_rowptr: torch.Tensor,
    edge_token: torch.Tensor,
    edge_child: torch.Tensor,
    *,
    token_base: int,
    invalid_node: int,
    max_deg: int,
    out: Optional[TrieBudgetScores] = None,
) -> TrieBudgetScores:
    """Score each beam row's legal trie children and sort them descending.

    ``hidden`` is ``[R, H]``, ``weight`` the restricted lm_head ``[vocab, H]``
    whose row ``i`` is token ``token_base + i``. ``max_deg`` is the padded plane
    width: pass the trie's ``_max_outdeg`` (a static property, so no
    ``deg.max().item()`` sync is needed to size it).
    """
    r = int(node_ids.numel())
    d = max(int(max_deg), 1)
    device = hidden.device
    if out is None:
        lp = torch.empty((r, d), dtype=torch.float32, device=device)
        tok = torch.empty((r, d), dtype=torch.int64, device=device)
        child = torch.empty((r, d), dtype=torch.int64, device=device)
        deg = torch.empty((r,), dtype=torch.int32, device=device)
        out = TrieBudgetScores(lp, tok, child, deg)

    mod = _try_module()
    if mod is not None and hidden.is_cuda and d <= MAX_DEG:
        try:
            mod.trie_budget_score(
                out.logprobs,
                out.tokens,
                out.children,
                out.deg,
                _c(weight),
                _c(hidden),
                _c(node_ids.reshape(-1).to(dtype=torch.int64)),
                _c(edge_rowptr.to(dtype=torch.int32)),
                _c(edge_token.to(dtype=torch.int64)),
                _c(edge_child.to(dtype=torch.int64)),
                int(token_base),
                int(invalid_node),
            )
            return out
        except Exception as exc:
            logger.debug("trie_budget_score CUDA failed, PyTorch fallback: %s", exc)

    nodes = node_ids.reshape(-1).to(dtype=torch.int64)
    n_nodes = int(edge_rowptr.numel()) - 1
    valid = (nodes >= 0) & (nodes < n_nodes)
    safe = nodes.clamp(0, max(n_nodes - 1, 0))
    rp = edge_rowptr.to(dtype=torch.int64)
    starts = rp[:-1][safe]
    row_deg = (rp[1:][safe] - starts).clamp(min=0)
    row_deg = torch.where(valid, row_deg, torch.zeros_like(row_deg)).clamp(max=d)
    offs = torch.arange(d, device=device, dtype=torch.int64)
    mask = offs.unsqueeze(0) < row_deg.unsqueeze(1)
    e = max(int(edge_token.numel()), 1)
    idx = (starts.unsqueeze(1) + offs.unsqueeze(0)).clamp(0, e - 1)
    tokens = edge_token.to(dtype=torch.int64)[idx]
    children = edge_child.to(dtype=torch.int64)[idx]

    vsz = int(weight.shape[0])
    rel = (tokens - int(token_base)).clamp(0, max(vsz - 1, 0))
    in_range = (tokens - int(token_base) >= 0) & (tokens - int(token_base) < vsz)
    legal = mask & in_range
    # Reference only: the [R, D, H] gather the CUDA path exists to avoid.
    # Reduce in fp32 like the kernel (and like F.linear) — a bf16 reduction over
    # H=2048 loses enough precision to reorder near-tied candidates.
    w = weight[rel].float()
    logits = (hidden.float().unsqueeze(1) * w).sum(dim=-1)
    logits = logits.masked_fill(~legal, _NEG_INF)
    lp = F.log_softmax(logits, dim=-1)
    lp = lp.masked_fill(~legal, _NEG_INF)
    # Sort descending; -inf columns sink to the tail, matching the kernel.
    order = torch.argsort(lp, dim=-1, descending=True, stable=True)
    lp = lp.gather(1, order)
    tokens = tokens.gather(1, order)
    children = children.gather(1, order)
    dead = lp == _NEG_INF
    tokens = tokens.masked_fill(dead, 0)
    children = children.masked_fill(dead, int(invalid_node))

    out.logprobs.copy_(lp)
    out.tokens.copy_(tokens)
    out.children.copy_(children)
    out.deg.copy_(legal.sum(dim=-1).to(dtype=torch.int32))
    return out


def trie_budget_select(
    scores: TrieBudgetScores,
    cum: torch.Tensor,
    row_offset: torch.Tensor,
    token_ids: Optional[torch.Tensor],
    col: torch.Tensor,
    leaf_count: Optional[torch.Tensor],
    *,
    beam_width: int,
    tau: float,
    n_min: int,
    n_max: int,
    invalid_node: int,
    out: Optional[TrieBudgetSelection] = None,
) -> TrieBudgetSelection:
    """Merge sorted rows to the top-k, apply the coverage-tau budget, expand.

    ``row_offset`` is an ``[N + 1]`` int32 prefix sum of each request's live
    beam rows into the packed ``[R, ...]`` planes, so requests may have
    different live widths. ``cum`` is ``[R]``. ``token_ids`` is ``[N, BW, L]``
    and must not alias the output table.

    ``tau > 0`` with a ``leaf_count`` table shrinks the surviving width to the
    smallest ranked prefix covering ``tau`` of the children's leaf mass,
    clamped to ``[n_min, n_max]``. The result is in ``.k`` on device.
    """
    n = int(row_offset.numel()) - 1
    bw = max(int(beam_width), 1)
    device = cum.device
    width = int(token_ids.shape[2]) if token_ids is not None else 1
    if out is None:
        out = TrieBudgetSelection(
            vals=torch.empty((n, bw), dtype=torch.float32, device=device),
            parents=torch.empty((n, bw), dtype=torch.int64, device=device),
            tokens=torch.empty((n, bw), dtype=torch.int64, device=device),
            k=torch.empty((n,), dtype=torch.int32, device=device),
            token_ids=(
                torch.empty((n, bw, width), dtype=torch.int64, device=device)
                if token_ids is not None
                else None
            ),
            node_ids=torch.empty((n, bw), dtype=torch.int64, device=device),
        )

    use_lc = leaf_count is not None and float(tau) > 0.0
    mod = _try_module()
    if mod is not None and cum.is_cuda and bw <= MAX_BEAM:
        lc = (
            leaf_count.to(dtype=torch.int32)
            if use_lc
            else torch.zeros(1, dtype=torch.int32, device=device)
        )
        tin = (
            token_ids
            if token_ids is not None
            else torch.zeros((n, bw, 1), dtype=torch.int64, device=device)
        )
        tout = out.token_ids if out.token_ids is not None else tin
        try:
            mod.trie_budget_select(
                out.vals,
                out.parents,
                out.tokens,
                out.k,
                _c(tin),
                tout,
                out.node_ids,
                _c(cum.to(dtype=torch.float32)),
                scores.logprobs,
                scores.tokens,
                scores.children,
                scores.deg,
                _c(lc),
                _c(row_offset.to(dtype=torch.int32)),
                _c(col.to(dtype=torch.int32)),
                float(tau),
                int(n_min),
                int(n_max),
                int(invalid_node),
                int(token_ids is not None),
                int(use_lc),
            )
            return out
        except Exception as exc:
            logger.debug("trie_budget_select CUDA failed, PyTorch fallback: %s", exc)

    return _select_torch(
        scores,
        cum,
        row_offset,
        token_ids,
        col,
        leaf_count if use_lc else None,
        bw=bw,
        tau=float(tau),
        n_min=int(n_min),
        n_max=int(n_max),
        invalid_node=int(invalid_node),
        out=out,
    )


def _select_torch(
    scores: TrieBudgetScores,
    cum: torch.Tensor,
    row_offset: torch.Tensor,
    token_ids: Optional[torch.Tensor],
    col: torch.Tensor,
    leaf_count: Optional[torch.Tensor],
    *,
    bw: int,
    tau: float,
    n_min: int,
    n_max: int,
    invalid_node: int,
    out: TrieBudgetSelection,
) -> TrieBudgetSelection:
    """Reference selection. Loops requests on the host (the kernel does not)."""
    offs = row_offset.detach().cpu().tolist()
    cols = col.detach().cpu().reshape(-1).tolist()
    kmax = min(int(n_max) if int(n_max) > 0 else bw, bw)
    d = int(scores.logprobs.shape[1])
    out.vals.fill_(_NEG_INF)
    out.parents.zero_()
    out.tokens.zero_()
    out.node_ids.fill_(int(invalid_node))
    for i in range(len(offs) - 1):
        r0, r1 = int(offs[i]), int(offs[i + 1])
        rows = r1 - r0
        if rows <= 0:
            out.k[i] = 0
            continue
        lp = scores.logprobs[r0:r1]
        flat = (cum[r0:r1].to(dtype=torch.float32).unsqueeze(1) + lp).reshape(-1)
        finite = int((lp > _NEG_INF).sum().item())
        k = min(kmax, finite)
        if k <= 0:
            out.k[i] = 0
            continue
        vals, idx = torch.topk(flat, k=k, largest=True, sorted=True)
        par = idx // d
        tok = scores.tokens[r0:r1].reshape(-1).gather(0, idx)
        ch = scores.children[r0:r1].reshape(-1).gather(0, idx)
        if tau > 0.0 and leaf_count is not None:
            n_leaf = int(leaf_count.numel())
            safe = ch.clamp(0, max(n_leaf - 1, 0))
            w = leaf_count.to(device=ch.device, dtype=torch.float32)[safe]
            w = torch.where((ch >= 0) & (ch < n_leaf), w, torch.zeros_like(w))
            total = float(w.sum().item())
            if total > 0.0:
                hit = (w.cumsum(0) >= tau * total).nonzero(as_tuple=False)
                if hit.numel() > 0:
                    k = min(max(int(hit[0].item()) + 1, max(n_min, 1)), k)
        out.k[i] = k
        out.vals[i, :k] = vals[:k]
        out.parents[i, :k] = par[:k]
        out.tokens[i, :k] = tok[:k]
        out.node_ids[i, :k] = ch[:k]
        if token_ids is not None and out.token_ids is not None:
            c = min(max(int(cols[i]), 0), int(token_ids.shape[2]) - 1)
            table = token_ids[i]
            gathered = table.clone()
            gathered[:k] = table[par[:k].clamp(0, rows - 1)]
            gathered[:k, c] = tok[:k]
            out.token_ids[i].copy_(gathered)
    return out


def try_load_trie_budget() -> bool:
    if not torch.cuda.is_available():
        return False
    return _try_module() is not None
