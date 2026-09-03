"""GenRec / beam-search trie helpers.

CUDA fused kernels are optional. The PyTorch fallbacks are the source of
correctness; ``try_load_beam_trie`` warms a self-contained extension when
tvm-ffi / CUDA is available. JIT includes live under ``kernel/include/``
(vendored TensorMatcher / LaunchKernel headers). Never imports sglang.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional, Sequence, Union

import torch

logger = logging.getLogger(__name__)

_NEG_INF = float("-inf")
_MODULE = None
_MODULE_TRIED = False


def _as_allow_u8(allow_table: torch.Tensor) -> torch.Tensor:
    if allow_table.dtype == torch.bool:
        return allow_table.view(torch.uint8)
    if allow_table.dtype == torch.uint8:
        return allow_table
    raise TypeError(f"allow_table dtype must be bool/uint8, got {allow_table.dtype}")


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
        logger.info("beam_trie JIT fallback (no CUDA)")
        return None
    try:
        from pathlib import Path

        from tvm_ffi.cpp import load_inline

        cuh = Path(__file__).resolve().parent / "csrc" / "beam_trie.cuh"
        if not cuh.is_file():
            logger.info("beam_trie JIT fallback (missing %s)", cuh.name)
            return None
        src = f'#include "{cuh}"\n'
        wrappers = [
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trie_mask_candidates, (trie_mask_candidates));",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trie_advance_nodes, (trie_advance_nodes));",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(beam_expand_token_ids, (beam_expand_token_ids));",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(genrec_mask_topk_expand, (genrec_mask_topk_expand));",
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
                "beamrec_beam_trie_col", extra_include_paths=includes, **kwargs
            )
        except TypeError:
            _MODULE = load_inline("beamrec_beam_trie_col", **kwargs)
        logger.info("flashrec fused beam_trie JIT loaded")
    except Exception as exc:
        logger.warning("beam_trie JIT fallback (PyTorch): %s", exc)
        _MODULE = None
    return _MODULE


def trie_mask_candidates(
    scores: torch.Tensor,
    cand_tokens: torch.Tensor,
    node_ids: torch.Tensor,
    allow_table: torch.Tensor,
    token_base: int,
    out: Optional[torch.Tensor] = None,
    neg_inf: float = _NEG_INF,
) -> torch.Tensor:
    mod = _try_module()
    if mod is not None and scores.is_cuda:
        if out is None:
            out = torch.empty_like(scores)
        mod.trie_mask_candidates(
            out.contiguous(),
            scores.contiguous(),
            cand_tokens.contiguous(),
            node_ids.contiguous(),
            _as_allow_u8(allow_table).contiguous(),
            int(token_base),
            float(neg_inf),
        )
        return out
    base = int(token_base)
    vsz = int(allow_table.shape[1])
    rel = cand_tokens.to(dtype=torch.int64) - base
    in_range = (rel >= 0) & (rel < vsz)
    rel_c = rel.clamp(0, max(vsz - 1, 0))
    unconstrained = node_ids < 0
    n_nodes = int(allow_table.shape[0])
    safe = node_ids.clamp(min=0, max=max(n_nodes - 1, 0))
    ok = allow_table[safe.unsqueeze(1), rel_c]
    if ok.dtype != torch.bool:
        ok = ok.bool()
    ok = (ok & in_range) | unconstrained.unsqueeze(1)
    return scores.masked_fill(~ok, neg_inf)


def trie_advance_nodes(
    node_ids: torch.Tensor,
    parents: torch.Tensor,
    tokens: torch.Tensor,
    next_node: torch.Tensor,
    token_base: int,
    invalid_node: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mod = _try_module()
    if mod is not None and node_ids.is_cuda:
        parents = (
            parents.to(device=node_ids.device, dtype=torch.int64).view(-1).contiguous()
        )
        tokens = (
            tokens.to(device=node_ids.device, dtype=torch.int64).view(-1).contiguous()
        )
        if out is None:
            out = torch.empty_like(parents)
        mod.trie_advance_nodes(
            out,
            node_ids.contiguous(),
            parents,
            tokens,
            next_node.contiguous(),
            int(token_base),
            int(invalid_node),
        )
        return out
    vsz = int(next_node.shape[1])
    parents = parents.to(device=node_ids.device, dtype=torch.int64).view(-1)
    toks = tokens.to(device=node_ids.device, dtype=torch.int64).view(-1)
    max_p = max(int(node_ids.shape[0]) - 1, 0)
    old = node_ids[parents.clamp(min=0, max=max_p)]
    rel = toks - int(token_base)
    in_range = (rel >= 0) & (rel < vsz)
    rel_c = rel.clamp(0, max(vsz - 1, 0))
    unconstrained = old < 0
    n_nodes = int(next_node.shape[0])
    safe = old.clamp(min=0, max=max(n_nodes - 1, 0))
    nxt = next_node[safe, rel_c]
    nxt = torch.where(in_range, nxt, torch.full_like(nxt, int(invalid_node)))
    return torch.where(unconstrained, old, nxt)


def trie_advance_nodes_batched(
    node_ids: torch.Tensor,
    parents: torch.Tensor,
    tokens: torch.Tensor,
    next_node: torch.Tensor,
    token_base: int,
    invalid_node: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if node_ids.ndim == 1:
        return trie_advance_nodes(
            node_ids, parents, tokens, next_node, token_base, invalid_node, out
        )
    n, bw = int(node_ids.shape[0]), int(node_ids.shape[1])
    vsz = int(next_node.shape[1])
    par = parents.to(device=node_ids.device, dtype=torch.int64).clamp(0, max(bw - 1, 0))
    toks = tokens.to(device=node_ids.device, dtype=torch.int64)
    batch_idx = torch.arange(n, device=node_ids.device).unsqueeze(1).expand(n, bw)
    old = node_ids[batch_idx, par]
    rel = toks - int(token_base)
    in_range = (rel >= 0) & (rel < vsz)
    rel_c = rel.clamp(0, max(vsz - 1, 0))
    unconstrained = old < 0
    n_nodes = int(next_node.shape[0])
    safe = old.clamp(min=0, max=max(n_nodes - 1, 0))
    nxt = next_node[safe, rel_c]
    nxt = torch.where(in_range, nxt, torch.full_like(nxt, int(invalid_node)))
    return torch.where(unconstrained, old, nxt)


def beam_expand_token_ids(
    token_ids: torch.Tensor,
    parents: torch.Tensor,
    new_tokens: torch.Tensor,
    col: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mod = _try_module()
    if mod is not None and token_ids.is_cuda and token_ids.dtype == torch.int64:
        bw, width = token_ids.shape
        parents = (
            parents.to(device=token_ids.device, dtype=torch.int64).view(-1).contiguous()
        )
        new_tokens = (
            new_tokens.to(device=token_ids.device, dtype=torch.int64)
            .view(-1)
            .contiguous()
        )
        if out is None:
            out = torch.empty_like(token_ids)
        mod.beam_expand_token_ids(
            out.contiguous(), token_ids.contiguous(), parents, new_tokens, int(col)
        )
        return out
    device = token_ids.device
    parents = torch.as_tensor(parents, dtype=torch.int64, device=device).view(-1)
    toks = torch.as_tensor(new_tokens, dtype=token_ids.dtype, device=device).view(-1)
    max_parent = max(token_ids.shape[0] - 1, 0)
    gathered = token_ids[parents.clamp(min=0, max=max_parent)]
    col = int(col)
    if col >= gathered.shape[1]:
        gathered = torch.nn.functional.pad(gathered, (0, 1))
    gathered[:, col] = toks
    return gathered


class GenrecFusedResult(NamedTuple):
    vals: torch.Tensor
    parents: torch.Tensor
    tokens: torch.Tensor
    indices: torch.Tensor
    token_ids: Optional[torch.Tensor]
    node_ids: Optional[torch.Tensor]


def _canonical_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


class GenrecFusedWorkspace:
    def __init__(self) -> None:
        self._bufs: dict = {}

    def get(
        self, name: str, shape: tuple, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        shape = tuple(int(x) for x in shape)
        device = _canonical_device(device)
        t = self._bufs.get(name)
        tail = shape[1:]
        same_tail = (
            t is not None
            and t.dtype == dtype
            and t.device == device
            and t.ndim == len(shape)
            and tuple(int(x) for x in t.shape[1:]) == tail
        )
        if t is None or not same_tail or int(t.size(0)) < shape[0]:
            grown = list(shape)
            if same_tail:
                grown[0] = max(shape[0], int(t.size(0)))
            t = torch.empty(*grown, dtype=dtype, device=device)
            self._bufs[name] = t
        return t[: shape[0]]

    def get_filled(
        self,
        name: str,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        value,
    ) -> torch.Tensor:
        """Like ``get``, but fill only when the backing buffer is (re)allocated."""
        prev = self._bufs.get(name)
        view = self.get(name, shape, dtype, device)
        full = self._bufs[name]
        if (
            prev is None
            or prev.data_ptr() != full.data_ptr()
            or prev.shape != full.shape
        ):
            full.fill_(value)
        return view

    def pair(
        self, name: str, shape: tuple, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.get(f"{name}_a", shape, dtype, device),
            self.get(f"{name}_b", shape, dtype, device),
        )


def tensors_alias(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True when ``a`` and ``b`` are the same storage, shape, and layout."""
    return (
        a.dtype == b.dtype
        and a.device == b.device
        and tuple(a.shape) == tuple(b.shape)
        and a.data_ptr() == b.data_ptr()
        and a.stride() == b.stride()
    )


def row_aliases_stack(row: Optional[torch.Tensor], stacked_row: torch.Tensor) -> bool:
    if row is None:
        return False
    return tensors_alias(row, stacked_row)


def rows_alias_stack(
    rows: Sequence[Optional[torch.Tensor]], stacked: torch.Tensor
) -> bool:
    n = int(stacked.shape[0])
    if len(rows) != n:
        return False
    return all(row_aliases_stack(rows[i], stacked[i]) for i in range(n))


def pick_pingpong_stack(
    ws: GenrecFusedWorkspace,
    name: str,
    rows: Sequence[Optional[torch.Tensor]],
    shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return the ping-pong src stack, packing only when rows are not already in it.

    If every ``rows[i]`` is a view of ``a[i]`` (or ``b[i]``), return that stack
    with no D2D copy. Otherwise copy each row into ``a``.
    """
    a, b = ws.pair(name, shape, dtype, device)
    if rows_alias_stack(rows, a):
        return a
    if rows_alias_stack(rows, b):
        return b
    n = int(shape[0])
    for i in range(n):
        src = rows[i]
        if src is None:
            raise ValueError(f"{name}: missing row {i}")
        a[i].copy_(src)
    return a


def _pingpong_io(
    ws: GenrecFusedWorkspace,
    name: str,
    user: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel must not gather from the buffer it writes. Read ``user`` in place
    when it is foreign storage; if ``user`` is already a ping/pong plane, write
    the other plane (no snapshot ``copy_``).
    """
    shape = tuple(int(x) for x in user.shape)
    a, b = ws.pair(name, shape, dtype, device)
    if tensors_alias(user, a):
        return a, b
    if tensors_alias(user, b):
        return b, a
    dst = b if user.data_ptr() == a.data_ptr() else a
    return user, dst


def _maybe_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


def _col_i32(
    col: Union[int, torch.Tensor],
    n: int,
    device: torch.device,
    ws: GenrecFusedWorkspace,
) -> torch.Tensor:
    """Broadcast int or copy [N]/[1] tensor into a persistent int32[N] buffer."""
    out = ws.get("col", (n,), torch.int32, device)
    if isinstance(col, torch.Tensor):
        src = col.to(device=device, dtype=torch.int32).reshape(-1)
        if int(src.numel()) == 1:
            out.copy_(src.expand(n))
        elif int(src.numel()) >= n:
            out.copy_(src[:n])
        else:
            raise ValueError(f"col length {int(src.numel())} < n={n}")
    else:
        out.fill_(int(col))
    return out


def _expand_token_ids_per_row(
    token_ids: torch.Tensor,
    parents: torch.Tensor,
    new_tokens: torch.Tensor,
    col: torch.Tensor,
) -> torch.Tensor:
    n, bw, width = token_ids.shape
    k = int(parents.shape[-1])
    par = parents.to(dtype=torch.int64).clamp(0, max(bw - 1, 0))
    batch_idx = torch.arange(n, device=token_ids.device).unsqueeze(1).expand(n, k)
    gathered = token_ids[batch_idx, par]
    col_t = col.to(device=token_ids.device, dtype=torch.int64).reshape(-1)[:n]
    idx = col_t.clamp(0, max(width - 1, 0)).view(n, 1, 1).expand(n, k, 1)
    gathered.scatter_(2, idx, new_tokens.to(dtype=token_ids.dtype).unsqueeze(-1))
    return gathered


def _genrec_cuda(
    cum: torch.Tensor,
    top_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    *,
    select_k: int,
    node_ids: Optional[torch.Tensor],
    allow_table: Optional[torch.Tensor],
    token_base: int,
    token_ids: Optional[torch.Tensor],
    col: Union[int, torch.Tensor],
    next_node: Optional[torch.Tensor],
    invalid_node: int,
    do_expand: Optional[torch.Tensor],
    apply_mask: bool,
    apply_expand: bool,
    apply_advance: bool,
    inplace: bool,  # CUDA always writes a distinct ping-pong plane; see fallback.
    workspace: Optional[GenrecFusedWorkspace],
) -> Optional[GenrecFusedResult]:
    mod = _try_module()
    if mod is None or not hasattr(mod, "genrec_mask_topk_expand") or not cum.is_cuda:
        return None
    n, bw = int(cum.shape[0]), int(cum.shape[1])
    c = int(top_logprobs.shape[2])
    k = int(select_k)
    apply_mask_i = int(
        bool(apply_mask and node_ids is not None and allow_table is not None)
    )
    apply_expand_i = int(bool(apply_expand and token_ids is not None))
    apply_advance_i = int(
        bool(
            apply_advance and apply_mask_i and apply_expand_i and next_node is not None
        )
    )
    device = cum.device
    ws = workspace if workspace is not None else GenrecFusedWorkspace()
    out_vals = ws.get("vals", (n, k), torch.float32, device)
    out_parents = ws.get("parents", (n, k), torch.int64, device)
    out_tokens = ws.get("sel_tokens", (n, k), torch.int64, device)
    out_indices = ws.get("indices", (n, k), torch.int64, device)
    scratch = ws.get("scratch", (n, bw * c), torch.float32, device)
    if apply_mask_i:
        node_in = _maybe_contiguous(
            node_ids
            if node_ids.dtype == torch.int64
            else node_ids.to(dtype=torch.int64)
        )
        allow_u8 = _maybe_contiguous(_as_allow_u8(allow_table))
    else:
        node_in = ws.get("dummy_nodes", (n, bw), torch.int64, device)
        allow_u8 = ws.get("dummy_allow", (1, 1), torch.uint8, device)
    if apply_expand_i:
        tok_user = _maybe_contiguous(
            token_ids
            if token_ids.dtype == torch.int64
            else token_ids.to(dtype=torch.int64)
        )
        tok_in, tok_out = _pingpong_io(ws, "tok", tok_user, torch.int64, device)
        if do_expand is None:
            do_exp = ws.get_filled("do_exp", (n,), torch.uint8, device, 1)
        else:
            do_exp = _maybe_contiguous(
                do_expand.to(device=device, dtype=torch.uint8).view(-1)
            )
        col_t = _col_i32(col, n, device, ws)
    else:
        tok_in = ws.get("dummy_tok", (n, bw, 1), torch.int64, device)
        tok_out = tok_in
        do_exp = ws.get_filled("do_exp_zero", (n,), torch.uint8, device, 0)
        col_t = ws.get_filled("col", (n,), torch.int32, device, 0)
    if apply_advance_i:
        nxt = _maybe_contiguous(
            next_node
            if next_node.dtype == torch.int64
            else next_node.to(dtype=torch.int64)
        )
        node_in, node_out = _pingpong_io(ws, "node", node_in, torch.int64, device)
    else:
        nxt = ws.get("dummy_next", (1, 1), torch.int64, device)
        node_out = ws.get("dummy_node_out", tuple(node_in.shape), torch.int64, device)
    try:
        mod.genrec_mask_topk_expand(
            out_vals,
            out_parents,
            out_tokens,
            out_indices,
            scratch,
            _maybe_contiguous(cum),
            _maybe_contiguous(top_logprobs),
            _maybe_contiguous(top_tokens),
            node_in,
            allow_u8,
            tok_in,
            tok_out,
            nxt,
            node_out,
            do_exp,
            col_t,
            int(token_base),
            int(invalid_node),
            apply_mask_i,
            apply_expand_i,
            apply_advance_i,
        )
    except Exception as exc:
        logger.debug("genrec CUDA kernel failed, PyTorch fallback: %s", exc)
        return None
    return GenrecFusedResult(
        vals=out_vals,
        parents=out_parents,
        tokens=out_tokens,
        indices=out_indices,
        token_ids=tok_out if apply_expand_i else None,
        node_ids=node_out if apply_advance_i else None,
    )


def genrec_mask_topk_expand(
    cum: torch.Tensor,
    top_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    *,
    select_k: int,
    node_ids: Optional[torch.Tensor] = None,
    allow_table: Optional[torch.Tensor] = None,
    token_base: int = 0,
    token_ids: Optional[torch.Tensor] = None,
    col: Union[int, torch.Tensor] = 0,
    next_node: Optional[torch.Tensor] = None,
    invalid_node: int = 0,
    do_expand: Optional[torch.Tensor] = None,
    apply_mask: bool = True,
    apply_expand: bool = True,
    apply_advance: bool = True,
    inplace: bool = False,
    workspace: Optional[GenrecFusedWorkspace] = None,
) -> GenrecFusedResult:
    """Mask + top-K + optional expand/advance. CUDA fused when JIT is available.

    CUDA writes token/node planes to a workspace ping-pong buffer (``in`` and
    ``out`` never alias). Callers that keep the returned ``token_ids`` /
    ``node_ids`` and pass them back on the next step avoid the D2D snapshot
    ``copy_`` of the whole [N,BW,L] table. ``inplace`` is honored only by the
    PyTorch fallback.
    """
    fused = _genrec_cuda(
        cum,
        top_logprobs,
        top_tokens,
        select_k=select_k,
        node_ids=node_ids,
        allow_table=allow_table,
        token_base=token_base,
        token_ids=token_ids,
        col=col,
        next_node=next_node,
        invalid_node=invalid_node,
        do_expand=do_expand,
        apply_mask=apply_mask,
        apply_expand=apply_expand,
        apply_advance=apply_advance,
        inplace=inplace,
        workspace=workspace,
    )
    if fused is not None:
        return fused
    n, bw = int(cum.shape[0]), int(cum.shape[1])
    c = int(top_logprobs.shape[2])
    k = int(select_k)
    scores = cum.unsqueeze(-1) + top_logprobs
    if apply_mask and node_ids is not None and allow_table is not None:
        flat_scores = scores.reshape(n * bw, c)
        flat_toks = top_tokens.reshape(n * bw, c)
        flat_nodes = node_ids.reshape(n * bw)
        flat_scores = trie_mask_candidates(
            flat_scores, flat_toks, flat_nodes, allow_table, int(token_base)
        )
        scores = flat_scores.view(n, bw, c)
    flat = scores.reshape(n, bw * c)
    vals, indices = torch.topk(
        flat, k=min(k, bw * c), dim=-1, largest=True, sorted=True
    )
    parents = indices // c
    tokens = top_tokens.reshape(n, bw * c).gather(1, indices)
    tok_out = token_ids
    node_out = node_ids
    col_t = None
    if apply_expand and token_ids is not None:
        col_t = (
            col
            if isinstance(col, torch.Tensor)
            else torch.full((n,), int(col), dtype=torch.int64, device=token_ids.device)
        )
        tok_out = _expand_token_ids_per_row(token_ids, parents, tokens, col_t)
        if inplace:
            token_ids.copy_(tok_out)
            tok_out = token_ids
    if apply_advance and apply_mask and next_node is not None and node_ids is not None:
        node_out = trie_advance_nodes_batched(
            node_ids, parents, tokens, next_node, int(token_base), int(invalid_node)
        )
        if inplace:
            node_ids.copy_(node_out)
            node_out = node_ids
    return GenrecFusedResult(
        vals=vals,
        parents=parents,
        tokens=tokens,
        indices=indices,
        token_ids=tok_out if apply_expand else None,
        node_ids=node_out if apply_advance else None,
    )


def call_genrec_cuda(
    out_vals: torch.Tensor,
    out_parents: torch.Tensor,
    out_tokens: torch.Tensor,
    out_indices: torch.Tensor,
    scratch: torch.Tensor,
    cum: torch.Tensor,
    top_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    node_in: torch.Tensor,
    allow_u8: torch.Tensor,
    tok_in: torch.Tensor,
    tok_out: torch.Tensor,
    next_node: torch.Tensor,
    node_out: torch.Tensor,
    do_exp: torch.Tensor,
    token_base: int,
    invalid_node: int,
    col_t: torch.Tensor,
    apply_mask: int = 1,
    apply_expand: int = 1,
    apply_advance: int = 1,
) -> bool:
    """Launch the fused CUDA kernel into caller-owned buffers (CUDA-graph safe)."""
    mod = _try_module()
    if mod is None or not hasattr(mod, "genrec_mask_topk_expand"):
        return False
    # C++ host: tensors then col_n, then token_base/invalid/apply_*.
    mod.genrec_mask_topk_expand(
        out_vals,
        out_parents,
        out_tokens,
        out_indices,
        scratch,
        cum,
        top_logprobs,
        top_tokens,
        node_in,
        allow_u8,
        tok_in,
        tok_out,
        next_node,
        node_out,
        do_exp,
        col_t,
        int(token_base),
        int(invalid_node),
        int(apply_mask),
        int(apply_expand),
        int(apply_advance),
    )
    return True


def try_load_beam_trie() -> bool:
    if not torch.cuda.is_available():
        return False
    return _try_module() is not None
