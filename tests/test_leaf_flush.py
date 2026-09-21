"""``try_leaf_flush`` enumerates the same suffixes with or without the host cache.

The flush walk used to pull the CSR edge tables back from GPU on every firing
(three ``.cpu().tolist()`` calls, ~12MB each at a 1.5M-SID catalog, 1.2ms each
measured). ``_build_unary_tables`` now keeps the Python lists it already built,
so the walk reads those instead. These tests pin the behaviour that makes the
substitution safe: the cached lists must equal the tensors they replace, and the
flush must produce identical output whether it reads the cache or falls back to
the device copy.
"""

from __future__ import annotations

import torch

from flashrec.search.budget import try_leaf_flush
from flashrec.search.state import BeamSearchList
from flashrec.search.trie import BeamValidPathTrie


def _sid_trie(seqs):
    """ValidPath in trie mode over the given complete SID sequences."""
    children = {}
    for seq in seqs:
        for i in range(len(seq)):
            children.setdefault(tuple(seq[:i]), set()).add(seq[i])
    children.setdefault(tuple(seqs[0]), set())
    vp = BeamValidPathTrie(children=children, max_depth=len(seqs[0]), mode="trie")
    vp.finalize_index(token_base=0, vocab_size=max(max(s) for s in seqs) + 1)
    return vp


class _Req:
    def __init__(self, beam_list, beam_width, max_new_tokens):
        self.beam_list = beam_list
        self.beam_width = beam_width
        self.max_new_tokens = max_new_tokens
        self.finished = False
        self.finish_reason = None
        self.temperature = 0.0


def _make_case(device):
    """A frontier one token deep whose live nodes each have <=2 leaves below."""
    seqs = [[1, 4, 7], [1, 4, 8], [2, 5, 9], [2, 6, 10]]
    vp = _sid_trie(seqs)
    node_of = vp._prefix_to_node
    bl = BeamSearchList()
    bl.token_ids = torch.tensor([[1], [2]], dtype=torch.int64, device=device)
    bl.cur_len = 1
    bl.cum_logprobs = torch.tensor([-0.5, -1.5], dtype=torch.float32, device=device)
    bl.node_ids = torch.tensor(
        [node_of[(1,)], node_of[(2,)]], dtype=torch.int64, device=device
    )
    bl.dense_authoritative = True
    return vp, _Req(bl, beam_width=4, max_new_tokens=3)


def test_host_edge_lists_match_tensors():
    vp, _ = _make_case(torch.device("cpu"))
    assert vp._edge_rowptr_host == vp._edge_rowptr.tolist()
    assert vp._edge_token_host == vp._edge_token.tolist()
    assert vp._edge_child_host == vp._edge_child.tolist()


def test_flush_output_matches_without_host_cache():
    """Cached lists and the .cpu() fallback must flush to the same beams."""
    dev = torch.device("cpu")
    results = []
    for use_cache in (True, False):
        vp, req = _make_case(dev)
        if not use_cache:
            vp._edge_rowptr_host = None
            vp._edge_token_host = None
            vp._edge_child_host = None
        assert try_leaf_flush(req, vp, limit=3, flush_leaves=2)
        bl = req.beam_list
        results.append(
            (bl.token_ids.tolist(), bl.cum_logprobs.tolist(), bl.cur_len)
        )
    assert results[0] == results[1]


def test_flush_enumerates_expected_suffixes():
    dev = torch.device("cpu")
    vp, req = _make_case(dev)
    assert try_leaf_flush(req, vp, limit=3, flush_leaves=2)
    bl = req.beam_list
    assert req.finished
    assert bl.cur_len == 3
    rows = {tuple(r) for r in bl.token_ids.tolist()}
    assert rows == {(1, 4, 7), (1, 4, 8), (2, 5, 9), (2, 6, 10)}
    # Rows stay sorted by parent score, so the -0.5 prefix leads.
    assert bl.cum_logprobs.tolist()[:2] == [-0.5, -0.5]
    assert bl.node_ids.tolist() == [-1] * 4
    assert bl.last_tokens.tolist() == [r[-1] for r in bl.token_ids.tolist()]


def test_flush_declines_when_subtree_too_wide():
    """A node with more leaves than flush_leaves must not be enumerated."""
    dev = torch.device("cpu")
    vp, req = _make_case(dev)
    assert not try_leaf_flush(req, vp, limit=3, flush_leaves=1)
    assert not req.finished
