"""Standard row-pointer CSR queries and mask -> Top-K/expand kernels.

The BW=K=50 fast path ports trie-op-learn's warp compaction + CUB sort.
Inputs retain the existing per-row sorting and log-softmax semantics.
"""

import logging
import math
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _check_tables(rowptr, columns, children):
    if (
        rowptr.dtype != torch.int32
        or columns.dtype != torch.int64
        or children.dtype != torch.int64
    ):
        raise TypeError("CSR requires int32 rowptr and int64 columns/children")
    if (
        rowptr.ndim != 1
        or columns.ndim != 1
        or children.ndim != 1
        or rowptr.numel() < 2
        or columns.numel() != children.numel()
    ):
        raise ValueError("invalid CSR array shapes")
    if any(
        not t.is_contiguous() or t.device != rowptr.device
        for t in (rowptr, columns, children)
    ):
        raise ValueError("CSR arrays must be contiguous on the same device")


@lru_cache(maxsize=1)
def module():
    from tvm_ffi.cpp import load_inline

    root = Path(__file__).resolve().parent
    include = str(root / "include")
    source = root / "csrc" / "csr_trie.cuh"
    # Include file contents in the JIT key; changing a header must rebuild it.
    src = source.read_text()
    wrappers = "\n".join(
        f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({name}, ({name}));"
        for name in ("csr_lookup", "csr_mask", "csr_finish")
    )
    result = load_inline(
        "flashrec_csr_trie",
        cuda_sources=[src + "\n" + wrappers],
        extra_include_paths=[include],
        extra_cuda_cflags=[
            "-std=c++17",
            "-O3",
            "--expt-relaxed-constexpr",
            f"-I{include}",
        ],
    )
    logger.info("standard CSR mask/Top-K/expand JIT loaded")
    return result


def lookup(rowptr, columns, children, invalid, nodes, tokens):
    _check_tables(rowptr, columns, children)
    if nodes.dtype != torch.int64 or tokens.dtype != torch.int64:
        raise TypeError("CSR nodes/tokens must be int64")
    if nodes.device != rowptr.device or tokens.device != rowptr.device:
        raise ValueError("CSR queries must use the table device")
    nodes, tokens = torch.broadcast_tensors(nodes, tokens)
    nodes, tokens = nodes.contiguous(), tokens.contiguous()
    if nodes.is_cuda:
        found = torch.empty_like(nodes, dtype=torch.bool)
        out = torch.empty_like(nodes)
        if nodes.numel():
            module().csr_lookup(
                rowptr, columns, children, invalid, nodes, tokens, found, out
            )
        return found, out
    safe = nodes.clamp(0, rowptr.numel() - 2)
    lo, hi = rowptr[safe].long(), rowptr[safe + 1].long()
    end = hi.clone()
    if columns.numel() == 0:
        return nodes < 0, torch.where(nodes < 0, nodes, invalid)
    for _ in range(math.ceil(math.log2(columns.numel() + 1)) + 1):
        active = lo < hi
        mid = (lo + hi) // 2
        smaller = columns[mid.clamp(max=columns.numel() - 1)] < tokens
        lo = torch.where(active & smaller, mid + 1, lo)
        hi = torch.where(active & ~smaller, mid, hi)
    pos = lo.clamp(max=columns.numel() - 1)
    found = (lo < end) & (columns[pos] == tokens)
    out = torch.where(found, children[pos], invalid)
    return found | (nodes < 0), torch.where(nodes < 0, nodes, out)


def run(
    rowptr,
    columns,
    children,
    invalid,
    scores,
    cum,
    lp,
    candidates,
    nodes,
    tin,
    cols,
    expand,
    values,
    parents,
    tokens,
    indices,
    tout,
    nout,
):
    """Capture-safe; all buffers owned by the existing graph/workspace.

    BW=K=50 uses exactly two kernels. Other beam widths retain torch.topk
    between the same CSR mask and expand kernels, with no dense allocation.
    """
    _check_tables(rowptr, columns, children)
    n, bw, c = lp.shape
    k, width = values.shape[1], tin.shape[-1]
    for tensor, shape, dtype in (
        (scores, (n, bw * c), torch.float32),
        (cum, (n, bw), torch.float32),
        (lp, (n, bw, c), torch.float32),
        (candidates, (n, bw, c), torch.int64),
        (nodes, (n, bw), torch.int64),
        (tin, (n, bw, width), torch.int64),
        (cols, (n,), torch.int32),
        (expand, (n,), torch.uint8),
        (values, (n, k), torch.float32),
        (parents, (n, k), torch.int64),
        (tokens, (n, k), torch.int64),
        (indices, (n, k), torch.int64),
        (tout, (n, bw, width), torch.int64),
        (nout, (n, bw), torch.int64),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != rowptr.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"CSR operator expected {shape} {dtype} contiguous on {rowptr.device}"
            )
    if min(n, bw, c, k, width) < 1 or k > bw or k > bw * c:
        raise ValueError("invalid CSR operator dimensions")
    mod = module()
    mod.csr_mask(rowptr, columns, children, invalid, cum, lp, candidates, nodes, scores)
    sort50 = nodes.shape[1] == 50 and values.shape[1] == 50
    if not sort50:
        torch.topk(scores, values.shape[1], dim=-1, sorted=True, out=(values, indices))
        torch.div(indices, candidates.shape[-1], rounding_mode="floor", out=parents)
        torch.gather(candidates.flatten(1), 1, indices, out=tokens)
    mod.csr_finish(
        rowptr,
        columns,
        children,
        invalid,
        scores,
        candidates,
        nodes,
        tin,
        cols,
        expand,
        values,
        parents,
        tokens,
        indices,
        tout,
        nout,
        sort50,
    )
    return True
