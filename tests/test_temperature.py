"""Temperature-controlled stochastic beam search (Gumbel top-k).

T=0 must reproduce deterministic top-k exactly; T>0 must (a) vary the
selection across RNG states, (b) never corrupt cumulative logprobs with
selection noise, and (c) match log_softmax(logits / T) rescaling exactly.
"""

import torch
import torch.nn.functional as F

from flashrec.search.expand import (
    apply_temperature,
    expand_step,
    gumbel_like,
    init_from_prefill,
)


def _logits(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g, dtype=torch.float32)


class TestApplyTemperature:
    def test_identity_at_zero_and_one(self):
        lp = F.log_softmax(_logits(16, 0), dim=-1)
        assert apply_temperature(lp, 0.0) is lp
        assert apply_temperature(lp, 1.0) is lp

    def test_matches_log_softmax_of_scaled_logits(self):
        logits = _logits(32, 1)
        lp = F.log_softmax(logits, dim=-1)
        for t in (0.3, 0.7, 2.0):
            expected = F.log_softmax(logits / t, dim=-1)
            torch.testing.assert_close(apply_temperature(lp, t), expected)


class TestGumbelNoise:
    def test_finite_and_rng_dependent(self):
        x = torch.zeros(1000)
        torch.manual_seed(0)
        a = gumbel_like(x)
        torch.manual_seed(1)
        b = gumbel_like(x)
        assert torch.isfinite(a).all()
        assert not torch.equal(a, b)


class TestInitFromPrefill:
    def test_zero_temperature_unchanged(self):
        logprobs = F.log_softmax(_logits(64, 2), dim=-1)
        base = init_from_prefill(
            logprobs, beam_width=4, beam_candidates=8, max_new_tokens=4, prompt_len=1
        )
        again = init_from_prefill(
            logprobs,
            beam_width=4,
            beam_candidates=8,
            max_new_tokens=4,
            prompt_len=1,
            temperature=0.0,
        )
        assert base.last_tokens.tolist() == again.last_tokens.tolist()
        torch.testing.assert_close(base.cum_logprobs, again.cum_logprobs)

    def test_sampled_beams_vary_and_scores_are_true(self):
        logprobs = F.log_softmax(_logits(64, 3), dim=-1)
        t = 1.5
        scaled = apply_temperature(logprobs, t)
        pool = set(scaled.topk(8).indices.tolist())
        picks = set()
        for seed in range(20):
            torch.manual_seed(seed)
            bl = init_from_prefill(
                logprobs,
                beam_width=4,
                beam_candidates=8,
                max_new_tokens=4,
                prompt_len=1,
                temperature=t,
            )
            toks = bl.last_tokens.tolist()
            assert len(set(toks)) == 4  # without replacement
            assert set(toks) <= pool  # only from candidate pool
            for tok, val in zip(toks, bl.cum_logprobs.tolist()):
                assert round(abs(val - float(scaled[tok])), 5) == 0
            picks.add(tuple(toks))
        assert len(picks) > 1  # selection actually varies


class TestExpandStep:
    def _fresh_beam(self, temperature=0.0):
        logprobs = F.log_softmax(_logits(64, 4), dim=-1)
        return init_from_prefill(
            logprobs,
            beam_width=2,
            beam_candidates=4,
            max_new_tokens=4,
            prompt_len=1,
            temperature=temperature,
        )

    def test_zero_temperature_deterministic(self):
        top_tokens = torch.tensor([[1, 2, 7], [1, 2, 7]], dtype=torch.int64)
        top_lp = torch.tensor(
            [[-0.1, -1.0, -5.0], [-0.2, -0.3, -4.0]], dtype=torch.float32
        )
        outs = []
        for seed in range(3):
            torch.manual_seed(seed)
            bl = self._fresh_beam()
            expand_step(bl, top_tokens, top_lp, beam_width=2, ignore_eos=True)
            outs.append((bl.last_tokens.tolist(), bl.cum_logprobs.tolist()))
        assert outs[0] == outs[1]
        assert outs[1] == outs[2]

    def test_sampled_expand_varies_with_true_cum_logprobs(self):
        top_tokens = torch.tensor([[1, 2, 7, 9], [1, 2, 7, 9]], dtype=torch.int64)
        top_lp = torch.tensor(
            [[-0.5, -0.9, -1.2, -1.4], [-0.6, -0.8, -1.1, -1.3]], dtype=torch.float32
        )
        results = set()
        for seed in range(20):
            torch.manual_seed(seed)
            bl = self._fresh_beam()
            cum_before = bl.cum_logprobs.clone()
            parents = expand_step(
                bl, top_tokens, top_lp, beam_width=2, ignore_eos=True, temperature=1.0
            )
            assert parents is not None
            # Cumulative logprobs must be exact parent-cum + step-lp (no noise).
            for i, (p, tok) in enumerate(
                zip(parents.tolist(), bl.last_tokens.tolist())
            ):
                col = top_tokens[p].tolist().index(tok)
                expected = float(cum_before[p]) + float(top_lp[p, col])
                assert round(abs(float(bl.cum_logprobs[i]) - expected), 5) == 0
            results.add(tuple(bl.last_tokens.tolist()))
        assert len(results) > 1

    def test_final_step_sampled_keeps_true_scores(self):
        top_tokens = torch.tensor([[1, 2, 7], [1, 2, 7]], dtype=torch.int64)
        top_lp = torch.tensor(
            [[-0.5, -0.9, -1.2], [-0.6, -0.8, -1.1]], dtype=torch.float32
        )
        torch.manual_seed(0)
        bl = self._fresh_beam()
        cum_before = bl.cum_logprobs.clone()
        out = expand_step(
            bl,
            top_tokens,
            top_lp,
            beam_width=2,
            ignore_eos=True,
            will_finish=True,
            temperature=1.0,
        )
        assert out is None
        all_scores = (cum_before.unsqueeze(1) + top_lp).reshape(-1).tolist()
        for val in bl.cum_logprobs.tolist():
            assert any(abs(val - s) < 1e-5 for s in all_scores)
