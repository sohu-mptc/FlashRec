"""Restricted LM-head: gather-then-softmax, compute_into, optional CUDA graph."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashrec.logits import RestrictedLMHead


def _gather_then_log_softmax(hidden, lm_weight, ids):
    logits = F.linear(hidden.to(dtype=lm_weight.dtype), lm_weight)
    gathered = logits[:, ids]
    return F.log_softmax(gathered.float(), dim=-1)


class TestRestrictedLMHead:
    def test_compute_matches_full_vocab_gather_then_softmax(self):
        torch.manual_seed(0)
        hidden = torch.randn(3, 8)
        lm_weight = torch.randn(20, 8)
        ids = [1, 4, 9, 15]
        head = RestrictedLMHead(ids, enabled=True)
        head.bind(lm_weight)
        assert head.num_tokens == len(ids)
        lp, cands = head.compute(hidden, lm_weight)
        expected = _gather_then_log_softmax(hidden, lm_weight, ids)
        torch.testing.assert_close(lp, expected)
        assert cands.tolist() == ids

    def test_compute_into_matches_compute(self):
        torch.manual_seed(1)
        hidden = torch.randn(5, 6)
        lm_weight = torch.randn(12, 6)
        ids = [0, 3, 7]
        head = RestrictedLMHead(ids)
        head.bind(lm_weight)
        eager, _ = head.compute(hidden, lm_weight)
        out = torch.empty(5, len(ids), dtype=torch.float32)
        head.compute_into(hidden, lm_weight, out)
        torch.testing.assert_close(out, eager)

    def test_compute_into_requires_bind(self):
        head = RestrictedLMHead([1, 2, 3])
        hidden = torch.randn(2, 4)
        weight = torch.randn(8, 4)
        out = torch.empty(2, 3)
        with pytest.raises(RuntimeError):
            head.compute_into(hidden, weight, out)

    def test_disabled_uses_full_vocab(self):
        torch.manual_seed(2)
        hidden = torch.randn(2, 4)
        lm_weight = torch.randn(6, 4)
        head = RestrictedLMHead([0, 1], enabled=False)
        lp, cands = head.compute(hidden, lm_weight)
        expected = F.log_softmax(F.linear(hidden, lm_weight).float(), dim=-1)
        torch.testing.assert_close(lp, expected)
        assert cands is None
        assert not head.ready


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
class TestRestrictedLMHeadCudaGraph:
    def test_graph_replay_matches_eager(self):
        device = torch.device("cuda")
        torch.manual_seed(3)
        hidden = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
        lm_weight = torch.randn(32, 8, device=device, dtype=torch.bfloat16)
        ids = [2, 5, 7, 11]
        head = RestrictedLMHead(ids)
        head.bind(lm_weight)
        out = torch.empty(4, len(ids), dtype=torch.float32, device=device)
        with torch.inference_mode():
            for _ in range(3):
                head.compute_into(hidden, lm_weight, out)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                head.compute_into(hidden, lm_weight, out)
            hidden.copy_(torch.randn_like(hidden))
            eager, _ = head.compute(hidden, lm_weight)
            g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, eager, atol=2e-3, rtol=2e-3)

    def test_clone_survives_next_replay(self):
        """Pipeline pipes sharing one [bs, K] buffer must not clobber clones."""
        device = torch.device("cuda")
        torch.manual_seed(4)
        hidden = torch.randn(3, 8, device=device, dtype=torch.bfloat16)
        lm_weight = torch.randn(16, 8, device=device, dtype=torch.bfloat16)
        ids = [1, 4, 9]
        head = RestrictedLMHead(ids)
        head.bind(lm_weight)
        out = torch.empty(3, len(ids), dtype=torch.float32, device=device)
        with torch.inference_mode():
            for _ in range(3):
                head.compute_into(hidden, lm_weight, out)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                head.compute_into(hidden, lm_weight, out)
            hidden_a = hidden.clone()
            g.replay()
            clone_a = out[:2].clone()
            eager_a, _ = head.compute(hidden_a, lm_weight)
            hidden.copy_(torch.randn_like(hidden))
            g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(clone_a, eager_a[:2], atol=2e-3, rtol=2e-3)


class TestPerCodebookLMHead:
    """Per-codebook lm_head: bind_codebook, compute_level, numerical equivalence."""

    def _make_head(self, vocab=24, hidden=8, codebook_sizes=None):
        if codebook_sizes is None:
            codebook_sizes = [8, 8, 8]
        total = sum(codebook_sizes)
        torch.manual_seed(42)
        lm_weight = torch.randn(vocab, hidden)
        ids = list(range(total))
        head = RestrictedLMHead(ids, enabled=True)
        head.bind(lm_weight)
        head.bind_codebook(codebook_sizes)
        return head, lm_weight, codebook_sizes

    def test_bind_codebook_creates_level_weights(self):
        head, _, cb = self._make_head()
        assert head.per_codebook_ready
        assert head.num_levels == 3
        assert head.max_codebook_k == 8
        for i, size in enumerate(cb):
            assert head.codebook_k(i) == size

    def test_compute_level_matches_full_compute_sliced(self):
        """compute_level(h, i) == compute(h)[offset:offset+size] up to softmax domain."""
        head, lm_weight, cb = self._make_head()
        torch.manual_seed(7)
        hidden = torch.randn(5, 8)
        full_lp, _ = head.compute(hidden, lm_weight)
        offset = 0
        for level, size in enumerate(cb):
            level_lp, level_ids = head.compute_level(hidden, level)
            assert level_lp.shape == (5, size)
            assert level_ids.tolist() == list(range(offset, offset + size))
            full_slice = full_lp[:, offset : offset + size]
            # Per-level softmax differs from full-vocab softmax, but the
            # argmax ordering within the level must be preserved.
            full_order = full_slice.argsort(dim=-1, descending=True)
            level_order = level_lp.argsort(dim=-1, descending=True)
            assert torch.equal(full_order, level_order)
            offset += size

    def test_compute_level_log_softmax_sums_to_one(self):
        head, _, cb = self._make_head()
        torch.manual_seed(8)
        hidden = torch.randn(3, 8)
        for level in range(len(cb)):
            lp, _ = head.compute_level(hidden, level)
            probs = lp.exp().sum(dim=-1)
            torch.testing.assert_close(probs, torch.ones(3), atol=1e-5, rtol=1e-5)

    def test_small_last_codebook(self):
        """4-level codebook with k=2 final level (boundary tokens)."""
        head, _, cb = self._make_head(
            vocab=26, hidden=8, codebook_sizes=[8, 8, 8, 2]
        )
        assert head.num_levels == 4
        assert head.codebook_k(3) == 2
        torch.manual_seed(9)
        hidden = torch.randn(4, 8)
        lp, ids = head.compute_level(hidden, 3)
        assert lp.shape == (4, 2)
        assert ids.tolist() == [24, 25]

    def test_bind_codebook_mismatch_is_noop(self):
        """Mismatched codebook_sizes sum leaves per_codebook_ready False."""
        torch.manual_seed(0)
        lm_weight = torch.randn(10, 4)
        head = RestrictedLMHead(list(range(10)), enabled=True)
        head.bind(lm_weight)
        head.bind_codebook([5, 3])  # sum=8 != 10
        assert not head.per_codebook_ready

    def test_level_weights_are_views(self):
        """Level weight slices must share storage with _weight (zero-copy)."""
        head, _, _ = self._make_head()
        for lw in head._level_weights:
            assert lw.storage().data_ptr() == head._weight.storage().data_ptr()
