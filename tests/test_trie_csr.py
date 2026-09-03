"""CSR trie cache must match the dense allow/next tables exactly.

The CSR layout activates automatically for large tries (dense cells over
``FLASHREC_TRIE_DENSE_MAX_CELLS``); here we force it by shrinking the module
threshold and compare against a dense twin built from the same sequences.
"""

import random

import pytest
import torch

from flashrec.search import trie as trie_mod
from flashrec.search.trie import build_trie_from_sequences

BASE = 1000
SIZES = [8, 8, 8]


def _random_sequences(rng, n):
    seqs = []
    for _ in range(n):
        a = rng.randrange(SIZES[0])
        b = rng.randrange(SIZES[1])
        c = rng.randrange(SIZES[2])
        seqs.append([BASE + a, BASE + SIZES[0] + b, BASE + SIZES[0] + SIZES[1] + c])
    return seqs


def _build_pair(seqs, device):
    dense = build_trie_from_sequences(seqs)
    dense._ensure_gpu_cache(device)
    saved = trie_mod._DENSE_MAX_CELLS
    trie_mod._DENSE_MAX_CELLS = 0
    try:
        csr = build_trie_from_sequences(seqs)
        csr._ensure_gpu_cache(device)
    finally:
        trie_mod._DENSE_MAX_CELLS = saved
    return dense, csr


@pytest.fixture
def trie_pair():
    rng = random.Random(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqs = _random_sequences(rng, 120)
    dense, csr = _build_pair(seqs, device)
    assert dense._allow_table is not None
    assert csr._allow_table is None
    assert csr._csr_keys is not None
    return rng, device, seqs, dense, csr


def _random_nodes(rng, dense, device, bw):
    n_nodes = len(dense._prefix_to_node)
    vals = []
    for _ in range(bw):
        r = rng.random()
        if r < 0.1:
            vals.append(-1)  # unconstrained
        elif r < 0.2:
            vals.append(dense._invalid_node)
        else:
            vals.append(rng.randrange(n_nodes))
    return torch.tensor(vals, dtype=torch.int64, device=device)


class TestTrieCsrParity:
    def test_mask_candidates_matches_dense(self, trie_pair):
        rng, device, _seqs, dense, csr = trie_pair
        vsz = sum(SIZES)
        for _ in range(20):
            bw, cand = 16, 24
            nodes = _random_nodes(rng, dense, device, bw)
            tokens = torch.randint(BASE - 3, BASE + vsz + 3, (bw, cand), device=device)
            scores = torch.randn(bw, cand, device=device, dtype=torch.float32)
            out_d = dense.mask_candidates(
                None, tokens, scores.clone(), cur_len=1, node_ids=nodes
            )
            out_c = csr.mask_candidates(
                None, tokens, scores.clone(), cur_len=1, node_ids=nodes
            )
            torch.testing.assert_close(out_d, out_c, equal_nan=True)

    def test_advance_matches_dense(self, trie_pair):
        rng, device, _seqs, dense, csr = trie_pair
        vsz = sum(SIZES)
        for _ in range(20):
            bw = 16
            nodes = _random_nodes(rng, dense, device, bw)
            parents = torch.randint(0, bw, (bw,), device=device)
            tokens = torch.randint(BASE - 3, BASE + vsz + 3, (bw,), device=device)
            out_d = dense.advance_node_ids(nodes, parents, tokens)
            out_c = csr.advance_node_ids(nodes, parents, tokens)
            assert torch.equal(out_d, out_c), f"advance mismatch:\n{out_d}\n{out_c}"

    def test_bootstrap_matches_dense(self, trie_pair):
        rng, device, seqs, dense, csr = trie_pair
        for cur_len in (1, 2, 3):
            bw = 16
            rows = []
            for _ in range(bw):
                if rng.random() < 0.7:
                    seq = rng.choice(seqs)
                    rows.append(seq[:cur_len])
                else:
                    rows.append(
                        [rng.randrange(BASE, BASE + sum(SIZES)) for _ in range(cur_len)]
                    )
            prefixes = torch.tensor(rows, dtype=torch.int64, device=device)
            out_d = dense.bootstrap_node_ids(prefixes, bw, cur_len, device)
            out_c = csr.bootstrap_node_ids(prefixes, bw, cur_len, device)
            assert torch.equal(
                out_d, out_c
            ), f"bootstrap mismatch at len {cur_len}:\n{out_d}\n{out_c}"
