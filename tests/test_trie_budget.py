"""Trie-Budget Beam kernels vs their PyTorch references.

Both entry points in ``flashrec.kernel.trie_budget`` fall back to PyTorch when
the JIT module is unavailable, so the reference is the correctness oracle: we
run each shape twice, once with the module and once with ``_MODULE`` forced to
``None``, and compare.

``trie_budget_score`` sorts each row by logprob, and near-tied candidates can
legitimately swap between a warp-reduced fp32 dot product and a batched
PyTorch matmul. Rather than demanding identical token order we assert the
invariants that make the row usable downstream: descending logprobs, a
log-softmax that sums to 1 over the legal children, and swaps confined to
float-noise ties.
"""

from __future__ import annotations

import pytest
import torch

from flashrec.kernel import trie_budget as tb

NEG_INF = float("-inf")
TOKEN_BASE = 1000

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")


def _random_csr(n_nodes, max_deg, vocab, device, rng):
    """Random CSR trie: ``rowptr`` / ``edge_token`` / ``edge_child``."""
    deg = torch.randint(0, max_deg + 1, (n_nodes,), generator=rng)
    rowptr = torch.zeros(n_nodes + 1, dtype=torch.int32)
    rowptr[1:] = deg.cumsum(0).to(torch.int32)
    n_edges = int(rowptr[-1])
    token = torch.randint(0, vocab, (n_edges,), dtype=torch.int64, generator=rng)
    # -1 stands for "no child": the kernel must treat it as an invalid node.
    child = torch.randint(-1, n_nodes, (n_edges,), dtype=torch.int64, generator=rng)
    return (
        rowptr.to(device),
        (token + TOKEN_BASE).to(device),
        child.to(device),
        max(int(deg.max()), 1),
    )


def _both_ways(fn, *args, **kwargs):
    """Run ``fn`` on CUDA and again on the PyTorch reference."""
    cuda = fn(*args, **kwargs)
    saved = tb._MODULE
    tb._MODULE = None
    try:
        ref = fn(*args, **kwargs)
    finally:
        tb._MODULE = saved
    return cuda, ref


@pytest.mark.parametrize(
    "n_rows,hidden,n_nodes,max_deg,vocab,dtype",
    [
        (64, 512, 256, 8, 4096, torch.float32),
        (64, 512, 256, 8, 4096, torch.bfloat16),
        (128, 2048, 1024, 32, 8192, torch.bfloat16),
        (50, 2048, 512, 64, 8192, torch.float16),
        (200, 1024, 4096, 200, 8192, torch.bfloat16),
        (8, 256, 64, 1, 1024, torch.float32),
    ],
)
def test_score_matches_reference(n_rows, hidden, n_nodes, max_deg, vocab, dtype):
    device = torch.device("cuda", 0)
    rng = torch.Generator().manual_seed(0)
    rowptr, edge_token, edge_child, deg_max = _random_csr(
        n_nodes, max_deg, vocab, device, rng
    )
    weight = (torch.randn(vocab, hidden, device=device) * 0.05).to(dtype)
    hid = (torch.randn(n_rows, hidden, device=device) * 0.5).to(dtype)
    # -1 rows are unconstrained beams: degree 0, every column dead.
    nodes = torch.randint(-1, n_nodes, (n_rows,), dtype=torch.int64, device=device)

    cuda, ref = _both_ways(
        tb.trie_budget_score,
        hid,
        weight,
        nodes,
        rowptr,
        edge_token,
        edge_child,
        token_base=TOKEN_BASE,
        invalid_node=n_nodes,
        max_deg=deg_max,
    )

    live = ref.logprobs > NEG_INF
    assert torch.equal(cuda.deg, ref.deg)
    assert bool((cuda.logprobs[~live] == NEG_INF).all())
    tol = 2e-4 if dtype == torch.float32 else 3e-3
    if live.any():
        assert (cuda.logprobs[live] - ref.logprobs[live]).abs().max().item() < tol

    # Rows are sorted descending with the dead tail at -inf.
    assert bool((cuda.logprobs[:, :-1] >= cuda.logprobs[:, 1:]).all())
    # log_softmax over the legal children: logsumexp of a live row is 0.
    lse = torch.logsumexp(cuda.logprobs.masked_fill(~live, NEG_INF), dim=-1)
    if bool((ref.deg > 0).any()):
        assert lse[ref.deg > 0].abs().max().item() < 1e-4

    # Any token that differs must sit next to a neighbor at the same score.
    for row, col in ((cuda.tokens != ref.tokens) & live).nonzero().tolist():
        gaps = [
            abs(ref.logprobs[row, col].item() - ref.logprobs[row, other].item())
            for other in (col - 1, col + 1)
            if 0 <= other < ref.logprobs.shape[1]
        ]
        assert gaps and min(gaps) < 1e-5, f"real reorder at row {row} col {col}"
