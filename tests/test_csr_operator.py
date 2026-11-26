"""Independent CPU oracle for standard CSR selection, expansion and replay."""

import os
from unittest.mock import patch

import pytest
import torch

from flashrec.kernel.csr_trie import run
from flashrec.search.trie import build_trie_from_sequences


def prepare(bw=50, c=100, case="random"):
    g = torch.Generator().manual_seed(42)
    seqs = [[a, 64 + b, 128 + (a + b) % 64] for a in range(64) for b in range(8)]
    seqs += [[a, 64 + b, 128 + (a + b + 1) % 64] for a in range(32) for b in range(8)]
    with patch.dict(os.environ, {"FLASHREC_TRIE_BACKEND": "csr"}):
        trie = build_trie_from_sequences(seqs)
        trie._ensure_gpu_cache(torch.device("cuda"))
    n, width = 2, 3
    lp = torch.randn(n, bw, c, generator=g).sort(-1, descending=True).values
    toks = torch.randint(-2, 195, (n, bw, c), generator=g)
    nodes = torch.randint(0, len(trie._prefix_to_node), (n, bw), generator=g)
    nodes[0, 0], nodes[1, 0] = -7, trie._invalid_node
    cum = torch.randn(n, bw, generator=g)
    history = torch.randint(0, 195, (n, bw, width), generator=g)
    if case == "dead":
        nodes.fill_(trie._invalid_node)
    elif case == "short":
        nodes.fill_(trie._invalid_node)
        nodes[:, 0] = -1
        lp[:, 0, 3:] = -torch.inf
    elif case == "ties":
        cum.zero_()
        lp.zero_()
    elif case == "infinities":
        lp[:, 0, 0] = torch.inf
        lp[:, 0, -1] = -torch.inf
    cols = torch.tensor([1, 2], dtype=torch.int32)
    expand = torch.ones(n, dtype=torch.uint8)
    if case == "skip":
        expand[1] = 0
    arrays = [x.cuda() for x in (cum, lp, toks, nodes, history, cols, expand)]
    out = [
        torch.empty(n, bw, device="cuda", dtype=dtype)
        for dtype in (torch.float32, torch.int64, torch.int64, torch.int64)
    ]
    out += [torch.full_like(arrays[4], -999), torch.full_like(arrays[3], -999)]
    scratch = torch.empty(n, bw * c, device="cuda")
    return trie, arrays, out, scratch


def oracle(trie, arrays):
    cum, lp, toks, nodes, history, cols, expand = [x.cpu() for x in arrays]
    n, bw, c = lp.shape
    inv = {i: p for p, i in trie._prefix_to_node.items()}
    values, parents, tokens, indices = [], [], [], []
    histories = torch.full_like(history, -999)
    children = torch.full_like(nodes, -999)
    for request in range(n):
        scores = cum[request, :, None] + lp[request]
        for b in range(bw):
            node = int(nodes[request, b])
            allowed = trie.children.get(inv.get(node, ("invalid",)), set())
            for col in range(c):
                if node >= 0 and int(toks[request, b, col]) not in allowed:
                    scores[b, col] = -torch.inf
        flat = scores.flatten().tolist()
        order = sorted(range(len(flat)), key=lambda i: (-flat[i], i))[:bw]
        vals = torch.tensor([flat[i] for i in order])
        best = order[0] if torch.isfinite(vals[0]) else 0
        selected = [
            idx if torch.isfinite(vals[k]) else best for k, idx in enumerate(order)
        ]
        par = torch.tensor([i // c for i in selected])
        tok = torch.tensor([int(toks[request].flatten()[i]) for i in selected])
        values.append(vals)
        parents.append(par)
        tokens.append(tok)
        indices.append(torch.tensor(selected))
        if expand[request]:
            histories[request] = history[request, par]
            histories[request, :, int(cols[request])] = tok
            for k in range(bw):
                old = int(nodes[request, par[k]])
                prefix = inv.get(old, ("invalid",)) + (int(tok[k]),)
                child = (
                    trie._prefix_to_node.get(prefix, -1)
                    if int(tok[k])
                    in trie.children.get(inv.get(old, ("invalid",)), set())
                    else trie._invalid_node
                )
                children[request, k] = old if old < 0 else child
    return [torch.stack(x) for x in (values, parents, tokens, indices)] + [
        histories,
        children,
    ]


def execute(trie, arrays, out, scratch):
    return run(
        trie._csr_rowptr,
        trie._csr_columns,
        trie._csr_child,
        trie._invalid_node,
        scratch,
        *arrays,
        *out,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("c", [31, 32, 33, 100, 512])
@pytest.mark.parametrize(
    "case", ["random", "dead", "short", "ties", "infinities", "skip"]
)
def test_csr_fused_matches_independent_cpu(c, case):
    trie, arrays, out, scratch = prepare(c=c, case=case)
    execute(trie, arrays, out, scratch)
    for got, expected in zip(out, oracle(trie, arrays)):
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_existing_graph_capture_replays_changed_inputs():
    trie, arrays, out, scratch = prepare(c=512)
    execute(trie, arrays, out, scratch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute(trie, arrays, out, scratch)
    for shift in (0.25, -0.7, 1.3):
        arrays[0].add_(shift)
        arrays[2].copy_(arrays[2].roll(1, dims=1))
        arrays[3].copy_(arrays[3].roll(1, dims=1))
        graph.replay()
        for got, expected in zip(out, oracle(trie, arrays)):
            torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bw", [1, 8, 128, 512, 1000])
def test_other_beam_widths_keep_csr_and_match_cpu(bw):
    trie, arrays, out, scratch = prepare(bw=bw, c=33)
    execute(trie, arrays, out, scratch)
    for got, expected in zip(out, oracle(trie, arrays)):
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)
