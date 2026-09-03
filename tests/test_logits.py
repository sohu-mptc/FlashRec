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
