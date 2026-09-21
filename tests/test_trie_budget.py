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


@pytest.mark.parametrize(
    "widths,max_deg,beam_width,tau,n_min,n_max,length,n_leaf",
    [
        ([50, 50, 50], 12, 50, 0.0, 1, 0, 4, 4096),
        ([50, 50, 50], 12, 50, 0.9, 16, 50, 4, 4096),
        ([64, 32, 17, 5], 64, 64, 0.9, 8, 64, 5, 8192),  # ragged live widths
        ([200], 32, 200, 0.5, 16, 200, 4, 4096),
        ([1, 2, 3], 8, 16, 0.9, 1, 16, 4, 512),
        ([512, 512], 16, 512, 0.95, 32, 512, 4, 65536),
    ],
)
def test_select_matches_reference(
    widths, max_deg, beam_width, tau, n_min, n_max, length, n_leaf
):
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    n_rows = sum(widths)
    n_req = len(widths)

    # Kernel B's input contract: rows sorted descending, -inf past the degree.
    logprobs = (
        torch.randn(n_rows, max_deg, device=device)
        .sort(dim=-1, descending=True)
        .values
    )
    deg = torch.randint(1, max_deg + 1, (n_rows,), dtype=torch.int32, device=device)
    offs = torch.arange(max_deg, device=device)
    logprobs = logprobs.masked_fill(
        offs.unsqueeze(0) >= deg.unsqueeze(1).to(torch.int64), NEG_INF
    ).contiguous()
    scores = tb.TrieBudgetScores(
        logprobs,
        torch.randint(0, 9999, (n_rows, max_deg), dtype=torch.int64, device=device),
        torch.randint(-1, n_leaf, (n_rows, max_deg), dtype=torch.int64, device=device),
        deg,
    )
    cum = torch.randn(n_rows, device=device)
    row_offset = torch.zeros(n_req + 1, dtype=torch.int32, device=device)
    row_offset[1:] = torch.tensor(widths, dtype=torch.int32).cumsum(0).to(device)
    token_ids = torch.randint(
        0, 9999, (n_req, beam_width, length), dtype=torch.int64, device=device
    )
    col = torch.full((n_req,), length - 1, dtype=torch.int32, device=device)
    leaf_count = torch.randint(1, 500, (n_leaf,), dtype=torch.int32, device=device)

    cuda, ref = _both_ways(
        tb.trie_budget_select,
        scores,
        cum,
        row_offset,
        token_ids,
        col,
        leaf_count,
        beam_width=beam_width,
        tau=tau,
        n_min=n_min,
        n_max=n_max,
        invalid_node=-7,
    )

    assert torch.equal(cuda.k, ref.k)
    for i in range(n_req):
        k = int(ref.k[i].item())
        assert 0 < k <= min(beam_width, widths[i] * max_deg)
        assert (cuda.vals[i, :k] - ref.vals[i, :k]).abs().max().item() < 1e-5
        assert torch.equal(cuda.tokens[i, :k], ref.tokens[i, :k])
        assert torch.equal(cuda.parents[i, :k], ref.parents[i, :k])
        assert torch.equal(cuda.node_ids[i, :k], ref.node_ids[i, :k])
        assert torch.equal(cuda.token_ids[i], ref.token_ids[i])
        # Survivors stay ranked, and each one's new score is parent + step.
        assert bool((cuda.vals[i, :k][:-1] >= cuda.vals[i, :k][1:]).all())
        assert bool((cuda.parents[i, :k] < widths[i]).all())


def test_select_tau_shrinks_below_beam_width():
    """Coverage-tau is the point of the kernel: it must give back k < bw."""
    device = torch.device("cuda", 0)
    torch.manual_seed(1)
    rows, max_deg, beam_width = 8, 16, 64
    logprobs = (
        torch.randn(rows, max_deg, device=device)
        .sort(dim=-1, descending=True)
        .values.contiguous()
    )
    scores = tb.TrieBudgetScores(
        logprobs,
        torch.randint(0, 999, (rows, max_deg), dtype=torch.int64, device=device),
        torch.arange(rows * max_deg, dtype=torch.int64, device=device).view(
            rows, max_deg
        ),
        torch.full((rows,), max_deg, dtype=torch.int32, device=device),
    )
    row_offset = torch.tensor([0, rows], dtype=torch.int32, device=device)
    # One dominant leaf: the tau prefix should stop almost immediately, then get
    # clamped up to n_min.
    leaf_count = torch.ones(rows * max_deg, dtype=torch.int32, device=device)
    leaf_count[scores.children[0, 0]] = 10_000

    out = tb.trie_budget_select(
        scores,
        torch.zeros(rows, device=device),
        row_offset,
        None,
        torch.zeros(1, dtype=torch.int32, device=device),
        leaf_count,
        beam_width=beam_width,
        tau=0.5,
        n_min=4,
        n_max=beam_width,
        invalid_node=-1,
    )
    assert int(out.k[0].item()) == 4
