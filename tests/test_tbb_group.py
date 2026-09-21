"""Batched group TBB expand and the one-sync unary/leaf gates.

``expand_group_tbb`` must reproduce the per-request ``expand_step`` fast path
(global top-k of ``cum + logprob``) exactly when ``tau == 0``, and apply the
coverage budget per request when ``tau > 0``. Everything here runs on CPU: the
function is plain torch over the (already scored) candidate planes.
"""

from __future__ import annotations

import torch

from flashrec.kernel.trie_budget import TrieBudgetScores
from flashrec.search.budget import expand_group_tbb
from flashrec.search.state import BeamSearchList

NEG_INF = float("-inf")


def _sorted_scores(n_rows: int, d: int, deg: torch.Tensor, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    lp = torch.randn(n_rows, d, generator=g).sort(dim=-1, descending=True).values
    offs = torch.arange(d)
    dead = offs.unsqueeze(0) >= deg.unsqueeze(1).to(torch.int64)
    lp = lp.masked_fill(dead, NEG_INF)
    tokens = torch.randint(100, 999, (n_rows, d), dtype=torch.int64, generator=g)
    children = torch.randint(0, 50, (n_rows, d), dtype=torch.int64, generator=g)
    tokens = tokens.masked_fill(dead, 0)
    children = children.masked_fill(dead, -1)
    return TrieBudgetScores(lp, tokens, children, deg.to(torch.int32))


def _beam_list(live: int, width: int, cur_len: int, seed: int = 0) -> BeamSearchList:
    g = torch.Generator().manual_seed(seed)
    bl = BeamSearchList()
    bl.token_ids = torch.randint(
        100, 999, (live, width), dtype=torch.int64, generator=g
    )
    bl.cur_len = cur_len
    bl.dense_authoritative = True
    bl.cum_logprobs = torch.randn(live, generator=g)
    bl.last_tokens = bl.token_ids[:, cur_len - 1].clone()
    bl.node_ids = torch.randint(0, 50, (live,), dtype=torch.int64, generator=g)
    bl.incomplete = bl.ensure_empty_stubs(live)
    return bl


def _reference_expand(bl_cum, bl_tokens, scores_row0, scores, r0, live, k, col):
    """Per-request expand_step fast-path math on one request's rows."""
    lp = scores.logprobs[r0 : r0 + live]
    d = lp.shape[1]
    flat = (bl_cum.unsqueeze(1) + lp).reshape(-1)
    vals, idx = torch.topk(flat, k=k, largest=True, sorted=True)
    parents = idx // d
    toks = scores.tokens[r0 : r0 + live].reshape(-1)[idx]
    children = scores.children[r0 : r0 + live].reshape(-1)[idx]
    table = bl_tokens[parents]
    table[:, col] = toks
    return vals, parents, toks, children, table


class TestExpandGroupParity:
    def test_matches_per_request_topk(self):
        live, d, width, col = 4, 6, 8, 2
        n = 3
        deg = torch.tensor([6, 3, 1, 6] * n, dtype=torch.int32)
        scores = _sorted_scores(n * live, d, deg, seed=3)
        bls = [_beam_list(live, width, col, seed=i) for i in range(n)]
        cums = [bl.cum_logprobs.clone() for bl in bls]
        tabs = [bl.token_ids.clone() for bl in bls]

        select_k = 5
        parents = expand_group_tbb(
            bls,
            scores,
            live=live,
            select_k=select_k,
            tau=0.0,
            n_min=1,
            leaf_count=None,
        )
        assert parents is not None
        for i, bl in enumerate(bls):
            vals, par, toks, children, table = _reference_expand(
                cums[i], tabs[i], None, scores, i * live, live, select_k, col
            )
            assert torch.equal(parents[i], par)
            assert torch.allclose(bl.cum_logprobs, vals)
            assert torch.equal(bl.last_tokens, toks)
            assert torch.equal(bl.node_ids, children)
            assert torch.equal(bl.token_ids, table)
            assert bl.cur_len == col + 1
            assert len(bl.incomplete) == select_k

    def test_select_k_wider_than_live(self):
        """Post-n0 growth: 2 live rows expanding to 8 survivors."""
        live, d, width, col = 2, 6, 4, 1
        deg = torch.full((live,), 6, dtype=torch.int32)
        scores = _sorted_scores(live, d, deg, seed=5)
        bl = _beam_list(live, width, col, seed=9)
        cum = bl.cum_logprobs.clone()
        tab = bl.token_ids.clone()
        parents = expand_group_tbb(
            [bl], scores, live=live, select_k=8, tau=0.0, n_min=1, leaf_count=None
        )
        vals, par, toks, children, table = _reference_expand(
            cum, tab, None, scores, 0, live, 8, col
        )
        assert torch.equal(parents[0], par)
        assert torch.allclose(bl.cum_logprobs, vals)
        assert torch.equal(bl.token_ids, table)

    def test_dead_tail_padded_like_topk(self):
        """Frontier narrower than select_k keeps -inf padded survivors."""
        live, d, width, col = 2, 4, 4, 1
        deg = torch.tensor([1, 1], dtype=torch.int32)
        scores = _sorted_scores(live, d, deg, seed=7)
        bl = _beam_list(live, width, col, seed=2)
        parents = expand_group_tbb(
            [bl], scores, live=live, select_k=4, tau=0.0, n_min=1, leaf_count=None
        )
        assert parents is not None
        assert bl.cum_logprobs.shape[0] == 4
        assert bool((bl.cum_logprobs[2:] == NEG_INF).all())


class TestCoverageBudget:
    def test_tau_shrinks_and_rounds(self):
        live, d, width, col = 4, 8, 6, 1
        deg = torch.full((live,), 8, dtype=torch.int32)
        scores = _sorted_scores(live, d, deg, seed=11)
        bl = _beam_list(live, width, col, seed=4)
        # Every child maps to leaf mass 1 except the top pick, which dominates:
        # the tau=0.5 prefix is 1, clamped up to n_min=4, rounded to 32, capped
        # at select_k.
        leaf = torch.ones(50, dtype=torch.int32)
        flat = (bl.cum_logprobs.unsqueeze(1) + scores.logprobs).reshape(-1)
        top = int(torch.argmax(flat))
        dom_child = int(scores.children.reshape(-1)[top])
        leaf[dom_child] = 100_000
        parents = expand_group_tbb(
            [bl],
            scores,
            live=live,
            select_k=24,
            tau=0.5,
            n_min=4,
            leaf_count=leaf,
        )
        assert parents is not None
        # 1 -> n_min 4 -> rounded to 32 -> capped at select_k 24.
        assert bl.cum_logprobs.shape[0] == 24 or bl.cum_logprobs.shape[0] == 4
        # With rounding to 32 capped at k=24 the width must be 24.
        assert bl.cum_logprobs.shape[0] == 24

    def test_tau_keeps_n_min_floor(self):
        live, d, width, col = 2, 4, 6, 1
        deg = torch.full((live,), 4, dtype=torch.int32)
        scores = _sorted_scores(live, d, deg, seed=13)
        bl = _beam_list(live, width, col, seed=6)
        leaf = torch.ones(50, dtype=torch.int32)
        parents = expand_group_tbb(
            [bl],
            scores,
            live=live,
            select_k=8,
            tau=1.0,
            n_min=2,
            leaf_count=leaf,
        )
        assert parents is not None
        # tau=1.0 needs the full prefix: width stays at select_k.
        assert bl.cum_logprobs.shape[0] == 8


class TestUnarySuffixTable:
    def test_suffix_matches_walk(self):
        from flashrec.search.trie import BeamValidPathTrie
        from flashrec.search.unary import can_unary_finish

        trie = BeamValidPathTrie(mode="trie")
        trie.children[()] = {10, 11}
        trie.children[(10,)] = {20}
        trie.children[(11,)] = {20, 21}
        trie.children[(10, 20)] = {30}
        trie.children[(11, 20)] = {30}
        trie.children[(11, 21)] = {30}
        trie.children[(10, 20, 30)] = set()
        trie.children[(11, 20, 30)] = set()
        trie.children[(11, 21, 30)] = set()
        trie.max_depth = 3
        trie.all_tokens = {10, 11, 20, 21, 30}
        trie.finalize_index(token_base=10, vocab_size=21)

        sfx = trie._unary_suffix_len
        assert sfx is not None
        for prefix, nid in trie._prefix_to_node.items():
            node = torch.tensor([nid], dtype=torch.int64)
            for remain in range(1, trie.max_depth + 1):
                walk = can_unary_finish(
                    node, trie._outdeg, trie._unary_child, remain
                )
                table = bool(int(sfx[nid]) >= remain)
                assert walk == table, (prefix, remain)

    def test_gated_try_unary_finish(self):
        from flashrec.search.trie import BeamValidPathTrie
        from flashrec.search.unary import try_unary_finish_request
        from flashrec.core import BeamRequest

        trie = BeamValidPathTrie(mode="trie")
        trie.children[()] = {10}
        trie.children[(10,)] = {20}
        trie.children[(10, 20)] = set()
        trie.max_depth = 2
        trie.all_tokens = {10, 20}
        trie.finalize_index(token_base=10, vocab_size=11)

        def _req():
            req = BeamRequest(input_ids=[1], beam_width=1, max_new_tokens=2)
            bl = BeamSearchList()
            bl.token_ids = torch.tensor([[10, 0]], dtype=torch.int64)
            bl.cur_len = 1
            bl.dense_authoritative = True
            bl.cum_logprobs = torch.zeros(1)
            bl.last_tokens = torch.tensor([10], dtype=torch.int64)
            bl.node_ids = torch.tensor(
                [trie._prefix_to_node[(10,)]], dtype=torch.int64
            )
            bl.incomplete = bl.ensure_empty_stubs(1)
            req.beam_list = bl
            return req

        # gate=False short-circuits without touching the request.
        req = _req()
        assert not try_unary_finish_request(req, trie, 2, gate=False)
        assert not req.finished
        # gate=True applies without re-running the device walk.
        req = _req()
        assert try_unary_finish_request(req, trie, 2, gate=True)
        assert req.finished
        assert req.beam_list.token_ids[0].tolist() == [10, 20]
        # gate=None still decides correctly on its own (suffix table path).
        req = _req()
        assert try_unary_finish_request(req, trie, 2)
        assert req.finished


class TestFirstLevelNotCapped:
    """The n0=64 first-level hard cap truncated n=128/512 recall; n0 is a floor."""

    def test_target_select_k_no_first_step_cap(self):
        from flashrec.search.budget import target_select_k

        # generated_len=1 (first decode step) used to return n0=64.
        assert target_select_k(512, 1, 3, 64, 0) == 512
        assert target_select_k(128, 1, 3, 64, 0) == 128
        assert target_select_k(512, 2, 1, 64, 0) == 512  # final fill
        assert target_select_k(512, 2, 3, 64, 256) == 256  # explicit n_max cap

    def test_batch_init_applies_coverage_with_floor(self):
        from flashrec.search.expand import init_from_prefill_batch
        from flashrec.search.trie import BeamValidPathTrie

        # 40 first-level codes, one of which dominates the leaf mass.
        n_first = 40
        trie = BeamValidPathTrie(mode="trie")
        trie.children[()] = set(range(100, 100 + n_first))
        for t in range(100, 100 + n_first):
            fanout = 30 if t == 100 else 1
            trie.children[(t,)] = set(range(200, 200 + fanout))
            for s in range(200, 200 + fanout):
                trie.children[(t, s)] = set()
        trie.max_depth = 2
        trie.all_tokens = set(range(100, 100 + n_first)) | set(range(200, 230))
        trie.finalize_index(token_base=100, vocab_size=130)

        torch.manual_seed(0)
        B, V = 3, 130
        logprobs = torch.log_softmax(torch.randn(B, V), dim=-1)
        cand_ids = torch.arange(100, 100 + V, dtype=torch.int64)
        bw = 32
        bls = init_from_prefill_batch(
            logprobs,
            beam_width=bw,
            beam_candidates=2 * bw,
            max_new_tokens=2,
            prompt_lens=[4] * B,
            valid_path=trie,
            candidate_token_ids=cand_ids,
            coverage_tau=0.9,
            coverage_n_min=8,
        )
        for bl in bls:
            w = int(bl.cum_logprobs.shape[0])
            # Coverage decides within [floor, bw]; never the old hard cap of
            # something below the floor, never more than bw.
            assert 8 <= w <= bw
            assert int(bl.last_tokens.shape[0]) == w
            assert int(bl.token_ids.shape[0]) == w
            assert int(bl.node_ids.shape[0]) == w
            assert bl.cur_len == 1

    def test_batch_init_full_width_when_tau_zero(self):
        from flashrec.search.expand import init_from_prefill_batch

        torch.manual_seed(1)
        B, V, bw = 2, 64, 48
        logprobs = torch.log_softmax(torch.randn(B, V), dim=-1)
        bls = init_from_prefill_batch(
            logprobs,
            beam_width=bw,
            beam_candidates=2 * bw,
            max_new_tokens=2,
            prompt_lens=[4] * B,
            coverage_tau=0.0,
        )
        for bl in bls:
            assert int(bl.cum_logprobs.shape[0]) == bw


class TestLeafFlushCompleteness:
    def _trie(self, seqs):
        from flashrec.search.trie import BeamValidPathTrie

        children = {}
        for seq in seqs:
            for i in range(len(seq)):
                children.setdefault(tuple(seq[:i]), set()).add(seq[i])
            children.setdefault(tuple(seq), set())
        vp = BeamValidPathTrie(
            children=children, max_depth=max(len(s) for s in seqs), mode="trie"
        )
        vp.finalize_index(
            token_base=0, vocab_size=max(max(s) for s in seqs) + 1
        )
        return vp

    def _req(self, vp, prefixes, bw, max_new):
        from flashrec.core import BeamRequest

        bl = BeamSearchList()
        bl.token_ids = torch.tensor(prefixes, dtype=torch.int64)
        bl.cur_len = len(prefixes[0])
        bl.cum_logprobs = torch.tensor(
            [-0.1 * (i + 1) for i in range(len(prefixes))]
        )
        bl.node_ids = torch.tensor(
            [vp._prefix_to_node[tuple(p)] for p in prefixes], dtype=torch.int64
        )
        bl.dense_authoritative = True
        req = BeamRequest(input_ids=[1], beam_width=bw, max_new_tokens=max_new)
        req.beam_list = bl
        return req

    def test_declines_when_completions_exceed_beam(self):
        """Old caps kept DFS-first leaves and dropped reachable items."""
        from flashrec.search.budget import try_leaf_flush

        seqs = [[1, s, t] for s in (4, 5) for t in (7, 8, 9)]
        vp = self._trie(seqs)
        req = self._req(vp, [[1]], bw=4, max_new=3)  # 6 completions > bw 4
        assert not try_leaf_flush(req, vp, limit=3, flush_leaves=8)
        assert not req.finished

    def test_no_short_paths_no_token_zero_padding(self):
        """Dead ends must not surface as token-0-padded invalid SIDs."""
        from flashrec.search.budget import try_leaf_flush

        # (1, 4) completes at depth 3; (1, 5) dead-ends at depth 2.
        seqs = [[1, 4, 7], [1, 4, 8]]
        vp = self._trie(seqs)
        vp.children[(1,)].add(5)
        vp.children[(1, 5)] = set()
        vp._index_ready = False
        vp.finalize_index(token_base=0, vocab_size=9)
        req = self._req(vp, [[1]], bw=4, max_new=3)
        assert try_leaf_flush(req, vp, limit=3, flush_leaves=8)
        rows = {tuple(r) for r in req.beam_list.token_ids.tolist()}
        assert rows == {(1, 4, 7), (1, 4, 8)}
