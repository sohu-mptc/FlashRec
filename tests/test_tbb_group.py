"""Batched group TBB expand and the one-sync unary/leaf gates.

``expand_group_tbb`` must reproduce the per-request ``expand_step`` fast path
(global top-k of ``cum + logprob``) exactly when ``tau == 0``, and apply the
coverage budget per request when ``tau > 0``. Everything here runs on CPU: the
function is plain torch over the (already scored) candidate planes.
"""

from __future__ import annotations

import copy

import torch

from flashrec.kernel.trie_budget import TrieBudgetScores
from flashrec.search.budget import coverage_select_count, expand_group_tbb
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
        parents = expand_group_tbb([bl], scores, live=live, select_k=8)
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
        parents = expand_group_tbb([bl], scores, live=live, select_k=4)
        assert parents is not None
        assert bl.cum_logprobs.shape[0] == 4
        assert bool((bl.cum_logprobs[2:] == NEG_INF).all())


class TestCoverageBudget:
    """``tau > 0`` must reproduce the legacy per-step prefix, per request.

    ``coverage_select_count`` is the reference the vectorized block inside
    ``expand_group_tbb`` is derived from, so a disagreement between the two is
    the bug that matters: it would silently reintroduce the beam widths the
    86e5129 removal was meant to bury. The two extra rules the block adds on
    top (round up to a multiple of 32, keep everything when every child is a
    single leaf) are part of the legacy behaviour, so the oracle applies them
    too rather than treating them as incidental.
    """

    N_LEAF = 50

    def _setup(self, seed, n=3, live=4, d=6, col=2):
        deg = torch.full((n * live,), d, dtype=torch.int32)
        scores = _sorted_scores(n * live, d, deg, seed=seed)
        bls = [_beam_list(live, 8, col, seed=i) for i in range(n)]
        leaf_count = torch.ones(self.N_LEAF, dtype=torch.int32)
        return scores, bls, leaf_count

    def _oracle(self, children_row, leaf_count, tau, n_min, k):
        """Plain-Python replay of the block's k for one request."""
        weights = [
            float(leaf_count[c]) if 0 <= c < self.N_LEAF else 0.0
            for c in children_row.tolist()
        ]
        total = sum(weights)
        if total <= 0.0:
            return k
        take = coverage_select_count(
            torch.tensor(weights, dtype=torch.float32),
            tau=tau,
            n_min=n_min,
            n_max=k,
        )
        # OneRec's last code: every child is a single item above a forced
        # sid_end. Equal unit mass makes the prefix ~tau * k and the unary
        # finish that follows cannot grow the beam back, so keep select_k.
        if max(weights) <= 1.0:
            take = k
        return min((take + 31) // 32 * 32, k)

    def test_widths_match_oracle(self):
        """Replay the block's k selection against the Python oracle.

        Run the same inputs twice: ``tau=0`` first, because that plain top-k
        pass defines the ranked row order the coverage prefix slices, then
        ``tau>0``. Comparing widths across the two runs is what makes the
        oracle non-circular — the beam lists are cloned, so the second call
        re-selects from the original frontier.
        """
        # k must exceed 32 for the block to truncate at all: every surviving
        # width is rounded up to a multiple of 32, so a k of 8 or 32 lands back
        # on k no matter where the prefix stopped.
        scores, bls, leaf_count = self._setup(seed=13, live=16, d=8)
        # One dominant subtree so the prefix genuinely truncates somewhere.
        leaf_count[scores.children[0, 0]] = 10_000
        leaf_count[scores.children[2, 1]] = 5_000
        live, k, tau, n_min = 16, 96, 0.9, 1

        bls0 = copy.deepcopy(bls)
        expand_group_tbb(bls0, scores, live=live, select_k=k)
        ordered = [bl.node_ids.clone() for bl in bls0]

        bls1 = copy.deepcopy(bls)
        expand_group_tbb(
            bls1,
            scores,
            live=live,
            select_k=k,
            tau=tau,
            n_min=n_min,
            leaf_count=leaf_count,
        )
        widths = [int(bl.cum_logprobs.shape[0]) for bl in bls1]
        assert min(widths) < k, "fixture no longer exercises truncation"
        for i, bl in enumerate(bls1):
            want = self._oracle(ordered[i], leaf_count, tau, n_min, k)
            assert int(bl.cum_logprobs.shape[0]) == want, (
                f"request {i}: block kept {bl.cum_logprobs.shape[0]}, "
                f"oracle said {want}"
            )
            # Survivors must be the *prefix* of the ranked order, not a
            # re-ranked subset: pruning is supposed to drop the tail only.
            assert torch.equal(bl.node_ids, ordered[i][:want])

    def test_tau_zero_keeps_full_select_k(self):
        """The default-off path must never shrink, whatever leaf_count says."""
        scores, bls, leaf_count = self._setup(seed=17)
        leaf_count[scores.children[0, 0]] = 1_000_000
        parents = expand_group_tbb(
            bls,
            scores,
            live=4,
            select_k=8,
            tau=0.0,
            n_min=8,
            leaf_count=leaf_count,
        )
        assert parents is not None
        assert [int(p.numel()) for p in parents] == [8, 8, 8]

    def test_floor_clamps_up(self):
        """A one-subtree frontier hits the prefix at 1; n_min must lift it.

        n_min has to be above 32 to be observable here: the block rounds every
        surviving width up to a multiple of 32, so any floor at or below 32
        lands on 32 either way. That is the legacy behaviour, so the test
        pins it at a size where the floor actually decides.
        """
        weights = torch.cat(
            [torch.tensor([1_000_000.0]), torch.ones(127, dtype=torch.float32)]
        )
        assert coverage_select_count(weights, tau=0.9, n_min=1, n_max=128) == 1
        assert coverage_select_count(weights, tau=0.9, n_min=96, n_max=128) == 96

        scores, bls, leaf_count = self._setup(seed=19, n=1, live=32, d=8)
        leaf_count[:] = 1
        leaf_count[scores.children[0, 0]] = 1_000_000
        parents = expand_group_tbb(
            bls,
            scores,
            live=32,
            select_k=128,
            tau=0.9,
            n_min=96,
            leaf_count=leaf_count,
        )
        assert parents is not None
        assert int(parents[0].numel()) == 96


class TestFusedCoverageKeep:
    """``_apply_fused`` must obey the same prefix, and must not touch anything
    when the flag is off.

    The fused CUDA decode path bypasses ``_expand_groups`` entirely, so it was
    silently exempt from the restored rule. These two properties are what the
    same-commit A/B rests on: the default path returns the full width untouched
    (so the ``full_beam`` arm really is HEAD), and the flag on actually trims.
    """

    class _Stub:
        """Enough of BeamRecEngine for the one method under test."""

        def __init__(self, *, enabled: bool, leaf_count=None):
            self.config = type(
                "Cfg",
                (),
                {
                    "enable_trie_budget": enabled,
                    "trie_budget_tau": 0.9,
                    "trie_budget_n0": 64,
                    "trie_budget_n_min": 16,
                    "trie_budget_n_max": 0,
                },
            )()
            self.valid_path = type(
                "VP",
                (),
                {
                    "active": True,
                    "mode": "trie",
                    "_leaf_count": leaf_count,
                    "ensure_unary_tables": lambda self, device: True,
                },
            )()

        _budget_tau = None
        _budget_floor = None

    @staticmethod
    def _scheduler(enabled, leaf_count=None):
        from flashrec.scheduler.scheduler import BeamRecEngine

        stub = TestFusedCoverageKeep._Stub(enabled=enabled, leaf_count=leaf_count)
        # Bind the real budget helpers so the flag plumbing is under test too,
        # not a reimplementation of it.
        stub._budget_tau = lambda: BeamRecEngine._budget_tau(stub)
        stub._budget_floor = lambda bw: BeamRecEngine._budget_floor(stub, bw)
        return stub

    def test_flag_off_keeps_full_width(self):
        n = 100
        leaf_count = torch.ones(n, dtype=torch.int64)
        stub = self._scheduler(enabled=False, leaf_count=leaf_count)
        node_ids = torch.arange(n, dtype=torch.int64)
        req = type("R", (), {"beam_width": 128})()
        from flashrec.scheduler.scheduler import BeamRecEngine

        assert (
            BeamRecEngine._fused_coverage_keep(stub, req, node_ids, remain=5)
            == node_ids.numel()
        )

    def test_flag_on_trims_to_coverage_prefix(self):
        # floor = max(n0=64, n_min=16) = 64, so a trim is only observable at
        # more than 64 rows: coverage_select_count clamps n_min to n_max, and
        # on fewer rows the floor alone pins the width back to k.
        n = 100
        leaf_count = torch.ones(n, dtype=torch.int64)
        stub = self._scheduler(enabled=True, leaf_count=leaf_count)
        node_ids = torch.arange(n, dtype=torch.int64)
        req = type("R", (), {"beam_width": 128})()
        from flashrec.scheduler.scheduler import BeamRecEngine

        keep = BeamRecEngine._fused_coverage_keep(stub, req, node_ids, remain=5)
        assert keep == 90  # 0.9 prefix of 100 equal-weight nodes
        assert keep < node_ids.numel()

    def test_floor_clamps_up(self):
        # A single dominant node means the prefix alone would keep 1; the
        # coverage floor is what stops the beam collapsing to nothing.
        n = 100
        leaf_count = torch.tensor([50_000, *([1] * (n - 1))], dtype=torch.int64)
        stub = self._scheduler(enabled=True, leaf_count=leaf_count)
        node_ids = torch.arange(n, dtype=torch.int64)
        req = type("R", (), {"beam_width": 128})()
        from flashrec.scheduler.scheduler import BeamRecEngine

        keep = BeamRecEngine._fused_coverage_keep(stub, req, node_ids, remain=5)
        assert keep == 64

    def test_final_expand_never_trims(self):
        n = 100
        leaf_count = torch.tensor([50_000, *([1] * (n - 1))], dtype=torch.int64)
        stub = self._scheduler(enabled=True, leaf_count=leaf_count)
        node_ids = torch.arange(n, dtype=torch.int64)
        req = type("R", (), {"beam_width": 128})()
        from flashrec.scheduler.scheduler import BeamRecEngine

        # remain <= 1 is the last expand, which must fill the beam.
        assert (
            BeamRecEngine._fused_coverage_keep(stub, req, node_ids, remain=1)
            == node_ids.numel()
        )


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
                walk = can_unary_finish(node, trie._outdeg, trie._unary_child, remain)
                table = bool(int(sfx[nid]) >= remain)
                assert walk == table, (prefix, remain)

    def test_gated_try_unary_finish(self):
        from flashrec.core import BeamRequest
        from flashrec.search.trie import BeamValidPathTrie
        from flashrec.search.unary import try_unary_finish_request

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
            bl.node_ids = torch.tensor([trie._prefix_to_node[(10,)]], dtype=torch.int64)
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
        vp.finalize_index(token_base=0, vocab_size=max(max(s) for s in seqs) + 1)
        return vp

    def _req(self, vp, prefixes, bw, max_new):
        from flashrec.core import BeamRequest

        bl = BeamSearchList()
        bl.token_ids = torch.tensor(prefixes, dtype=torch.int64)
        bl.cur_len = len(prefixes[0])
        bl.cum_logprobs = torch.tensor([-0.1 * (i + 1) for i in range(len(prefixes))])
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
