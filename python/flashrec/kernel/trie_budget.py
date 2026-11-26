"""Fused sparse lm_head scoring for trie decode.

``trie_budget_score`` gathers CSR neighbors, log-softmaxes over the legal
children, and sorts each row descending, reading ``lm_head`` weight rows in
place. The PyTorch path materializes ``W[tokens]`` as ``[R, D, H]``; the
kernel does not, which is what makes wide out-degrees affordable.

The PyTorch fallback is the source of correctness; the CUDA module is
optional and JIT-loaded.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "TrieBudgetScores",
    "trie_budget_score",
    "try_load_trie_budget",
]

_NEG_INF = float("-inf")
_MODULE = None
_MODULE_TRIED = False

# Mirrors kTbbMaxDeg in csrc/trie_budget.cuh. Above this the CUDA path is
# skipped and the PyTorch reference runs instead.
MAX_DEG = 2048


class TrieBudgetScores(NamedTuple):
    """Per-row candidates, sorted descending by logprob within each row."""

    logprobs: torch.Tensor  # [R, D] fp32, -inf past the row's out-degree
    tokens: torch.Tensor  # [R, D] int64
    children: torch.Tensor  # [R, D] int64 next node id (invalid_node when unused)
    deg: torch.Tensor  # [R] int32 true out-degree


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


def try_load_trie_budget() -> bool:
    if not torch.cuda.is_available():
        return False
    return _try_module() is not None
