"""CUDA-graph top-k + fused trie expand vs eager (no transformer)."""

from __future__ import annotations

import pytest
import torch

from flashrec.engine.graph import (
    DecodeGraphRunner,
    ExpandCaptureSpec,
    LevelCaptureSpec,
    fill_topk_tokens,
)
from flashrec.kernel.beam_trie import call_genrec_cuda, try_load_beam_trie


def _run_topk_expand(buf, spec, n, bw, C):
    stacked = buf["logprobs"].view(n, bw, -1)
    torch.topk(
        stacked,
        C,
        dim=-1,
        largest=True,
        sorted=True,
        out=(buf["topk_lp"], buf["topk_idx"]),
    )
    fill_topk_tokens(spec["cand_ids"], buf["topk_idx"], buf["topk_tok"])
    ok = call_genrec_cuda(
        buf["exp_vals"],
        buf["exp_parents"],
        buf["exp_tokens"],
        buf["exp_indices"],
        buf["exp_scratch"],
        buf["exp_cum"],
        buf["topk_lp"],
        buf["topk_tok"],
        buf["exp_nodes"],
        spec["allow"],
        buf["exp_tok_in"],
        buf["exp_tok_out"],
        spec["next_node"],
        buf["exp_node_out"],
        buf["exp_do"],
        spec["token_base"],
        spec["invalid"],
        buf["exp_col"],
    )
    if not ok:
        raise RuntimeError("call_genrec_cuda failed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
class TestGraphExpand:
    def test_fill_topk_tokens_matches_advanced_index(self):
        cand = torch.arange(10, 26, dtype=torch.int64)
        idx = torch.tensor([[[0, 3, 5], [1, 2, 9]]], dtype=torch.int64)
        out = torch.empty_like(idx)
        fill_topk_tokens(cand, idx, out)
        assert torch.equal(out, cand[idx])

    def test_graph_replay_matches_eager_and_clone_survives(self):
        if not try_load_beam_trie():
            pytest.skip("beam_trie JIT unavailable")
        device = torch.device("cuda")
        torch.manual_seed(7)
        n, bw, k_vocab, C, width = 2, 4, 16, 6, 5
        spec = {
            "cand_ids": torch.arange(k_vocab, device=device, dtype=torch.int64) + 10,
            "allow": torch.ones(4, 16, dtype=torch.uint8, device=device),
            "next_node": torch.zeros(4, 16, dtype=torch.int64, device=device),
            "token_base": 10,
            "invalid": 0,
        }
        buf = {
            "logprobs": torch.randn(
                n * bw, k_vocab, device=device, dtype=torch.float32
            ),
            "topk_lp": torch.empty(n, bw, C, device=device, dtype=torch.float32),
            "topk_idx": torch.empty(n, bw, C, device=device, dtype=torch.int64),
            "topk_tok": torch.empty(n, bw, C, device=device, dtype=torch.int64),
            "exp_cum": torch.randn(n, bw, device=device, dtype=torch.float32),
            "exp_nodes": torch.zeros(n, bw, dtype=torch.int64, device=device),
            "exp_tok_in": torch.randint(10, 20, (n, bw, width), device=device),
            "exp_tok_out": torch.empty(n, bw, width, dtype=torch.int64, device=device),
            "exp_col": torch.ones(n, dtype=torch.int32, device=device),
            "exp_do": torch.ones(n, dtype=torch.uint8, device=device),
            "exp_node_out": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_vals": torch.empty(n, bw, dtype=torch.float32, device=device),
            "exp_parents": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_tokens": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_indices": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_scratch": torch.empty(n, bw * C, dtype=torch.float32, device=device),
        }
        with torch.inference_mode():
            for _ in range(3):
                _run_topk_expand(buf, spec, n, bw, C)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _run_topk_expand(buf, spec, n, bw, C)
            buf["logprobs"].copy_(torch.randn_like(buf["logprobs"]))
            buf["exp_cum"].copy_(torch.randn_like(buf["exp_cum"]))
            buf["exp_tok_in"].copy_(
                torch.randint(10, 20, buf["exp_tok_in"].shape, device=device)
            )
            eager = {k: buf[k].clone() for k in ("logprobs", "exp_cum", "exp_tok_in")}
            _run_topk_expand(buf, spec, n, bw, C)
            eager_vals = buf["exp_vals"].clone()
            eager_tok = buf["exp_tokens"].clone()
            eager_ids = buf["exp_tok_out"].clone()
            buf["logprobs"].copy_(eager["logprobs"])
            buf["exp_cum"].copy_(eager["exp_cum"])
            buf["exp_tok_in"].copy_(eager["exp_tok_in"])
            g.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                buf["exp_vals"], eager_vals, atol=2e-4, rtol=2e-4
            )
            assert torch.equal(buf["exp_tokens"], eager_tok)
            assert torch.equal(buf["exp_tok_out"], eager_ids)
            clone_vals = buf["exp_vals"][:1].clone()
            buf["logprobs"].copy_(torch.randn_like(buf["logprobs"]))
            g.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(clone_vals, eager_vals[:1], atol=2e-4, rtol=2e-4)

    def test_clone_fused_steals_pingpong_token_plane(self):
        from flashrec.engine.graph import DecodeGraphRunner

        n, bw, width, k = 2, 4, 5, 4
        tok_out = torch.arange(n * bw * width).view(n, bw, width)
        node_out = torch.arange(n * bw).view(n, bw)
        buf = {
            "exp_pingpong": True,
            "exp_phase": 0,
            "exp_vals": torch.zeros(n, k),
            "exp_parents": torch.zeros(n, k, dtype=torch.int64),
            "exp_tokens": torch.zeros(n, k, dtype=torch.int64),
            "exp_indices": torch.zeros(n, k, dtype=torch.int64),
            "exp_tok_in": torch.zeros(n, bw, width, dtype=torch.int64),
            "exp_tok_out": tok_out,
            "exp_nodes": torch.zeros(n, bw, dtype=torch.int64),
            "exp_node_out": node_out,
        }
        fused = DecodeGraphRunner._clone_fused(
            object.__new__(DecodeGraphRunner), buf, n
        )
        assert fused.token_ids.data_ptr() == tok_out.data_ptr()
        assert fused.node_ids.data_ptr() == node_out.data_ptr()
        assert fused.vals.data_ptr() != buf["exp_vals"].data_ptr()
        buf["exp_phase"] = 1
        fused1 = DecodeGraphRunner._clone_fused(
            object.__new__(DecodeGraphRunner), buf, n
        )
        assert fused1.token_ids.data_ptr() == buf["exp_tok_in"].data_ptr()

    def test_clone_fused_steal_false_clones_even_for_n_gt_1(self):
        n, bw, width, k = 2, 4, 5, 4
        tok_out = torch.arange(n * bw * width).view(n, bw, width)
        node_out = torch.arange(n * bw).view(n, bw)
        buf = {
            "exp_pingpong": True,
            "exp_phase": 0,
            "exp_vals": torch.zeros(n, k),
            "exp_parents": torch.zeros(n, k, dtype=torch.int64),
            "exp_tokens": torch.zeros(n, k, dtype=torch.int64),
            "exp_indices": torch.zeros(n, k, dtype=torch.int64),
            "exp_tok_in": torch.zeros(n, bw, width, dtype=torch.int64),
            "exp_tok_out": tok_out,
            "exp_nodes": torch.zeros(n, bw, dtype=torch.int64),
            "exp_node_out": node_out,
        }
        fused = DecodeGraphRunner._clone_fused(
            object.__new__(DecodeGraphRunner), buf, n, steal=False
        )
        assert fused.token_ids.data_ptr() != tok_out.data_ptr()
        assert fused.node_ids.data_ptr() != node_out.data_ptr()
        assert torch.equal(fused.token_ids, tok_out)

    def test_copy_list_reuse_does_not_clobber_prior_dest(self):
        """Same pinned name, two dests: first dest must keep its values."""
        from flashrec.engine.staging import PinnedStage

        device = torch.device("cuda")
        stage = PinnedStage()
        a = torch.zeros(1, dtype=torch.int32, device=device)
        b = torch.zeros(1, dtype=torch.int32, device=device)
        stage.copy_list("exp_col_lv", [1], device, torch.int32, dest=a)
        stage.copy_list("exp_col_lv", [3], device, torch.int32, dest=b)
        assert int(a.item()) == 1
        assert int(b.item()) == 3

    def test_capture_levels_is_idempotent(self):
        runner = object.__new__(DecodeGraphRunner)
        runner.enabled = True
        runner.capture_bs = [100, 200]
        marker = object()
        runner.graphs_level = {(100, 0): marker, (200, 0): marker}
        runner.level_specs = {}
        spec = LevelCaptureSpec(
            level=0,
            logits_fn=lambda hidden, out: None,
            logprobs_k=8,
            expand_spec=ExpandCaptureSpec(
                beam_width=100,
                cand=8,
                select_k=100,
                width=5,
                cand_ids=torch.zeros(8, dtype=torch.int64),
                allow_table=torch.zeros(1, 8, dtype=torch.uint8),
                next_node=torch.zeros(1, 8, dtype=torch.int64),
                token_base=0,
                invalid_node=0,
            ),
        )
        DecodeGraphRunner.capture_levels(
            runner, lambda batch: None, lambda batch: None, {0: spec}
        )
        assert runner.graphs_level[(100, 0)] is marker
        assert runner.graphs_level[(200, 0)] is marker
        assert runner.level_specs == {}

    def test_graph_flip_kernel_matches_eager(self):
        if not try_load_beam_trie():
            pytest.skip("beam_trie JIT unavailable")
        device = torch.device("cuda")
        torch.manual_seed(8)
        n, bw, k_vocab, C, width = 2, 4, 16, 6, 5
        spec = {
            "cand_ids": torch.arange(k_vocab, device=device, dtype=torch.int64) + 10,
            "allow": torch.ones(4, 16, dtype=torch.uint8, device=device),
            "next_node": torch.zeros(4, 16, dtype=torch.int64, device=device),
            "token_base": 10,
            "invalid": 0,
        }
        buf = {
            "logprobs": torch.randn(
                n * bw, k_vocab, device=device, dtype=torch.float32
            ),
            "topk_lp": torch.empty(n, bw, C, device=device, dtype=torch.float32),
            "topk_idx": torch.empty(n, bw, C, device=device, dtype=torch.int64),
            "topk_tok": torch.empty(n, bw, C, device=device, dtype=torch.int64),
            "exp_cum": torch.randn(n, bw, device=device, dtype=torch.float32),
            "exp_nodes": torch.zeros(n, bw, dtype=torch.int64, device=device),
            "exp_tok_in": torch.empty(n, bw, width, dtype=torch.int64, device=device),
            "exp_tok_out": torch.randint(10, 20, (n, bw, width), device=device),
            "exp_col": torch.ones(n, dtype=torch.int32, device=device),
            "exp_do": torch.ones(n, dtype=torch.uint8, device=device),
            "exp_node_out": torch.zeros(n, bw, dtype=torch.int64, device=device),
            "exp_vals": torch.empty(n, bw, dtype=torch.float32, device=device),
            "exp_parents": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_tokens": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_indices": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_scratch": torch.empty(n, bw * C, dtype=torch.float32, device=device),
        }

        def _run_flip():
            stacked = buf["logprobs"].view(n, bw, -1)
            torch.topk(
                stacked,
                C,
                dim=-1,
                largest=True,
                sorted=True,
                out=(buf["topk_lp"], buf["topk_idx"]),
            )
            fill_topk_tokens(spec["cand_ids"], buf["topk_idx"], buf["topk_tok"])
            ok = call_genrec_cuda(
                buf["exp_vals"],
                buf["exp_parents"],
                buf["exp_tokens"],
                buf["exp_indices"],
                buf["exp_scratch"],
                buf["exp_cum"],
                buf["topk_lp"],
                buf["topk_tok"],
                buf["exp_node_out"],
                spec["allow"],
                buf["exp_tok_out"],
                buf["exp_tok_in"],
                spec["next_node"],
                buf["exp_nodes"],
                buf["exp_do"],
                spec["token_base"],
                spec["invalid"],
                buf["exp_col"],
            )
            if not ok:
                raise RuntimeError("call_genrec_cuda failed")

        with torch.inference_mode():
            for _ in range(2):
                _run_flip()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _run_flip()
            buf["logprobs"].copy_(torch.randn_like(buf["logprobs"]))
            buf["exp_cum"].copy_(torch.randn_like(buf["exp_cum"]))
            buf["exp_tok_out"].copy_(
                torch.randint(10, 20, buf["exp_tok_out"].shape, device=device)
            )
            eager = {
                k: buf[k].clone()
                for k in ("logprobs", "exp_cum", "exp_tok_out", "exp_node_out")
            }
            _run_flip()
            eager_vals = buf["exp_vals"].clone()
            eager_ids = buf["exp_tok_in"].clone()
            buf["logprobs"].copy_(eager["logprobs"])
            buf["exp_cum"].copy_(eager["exp_cum"])
            buf["exp_tok_out"].copy_(eager["exp_tok_out"])
            buf["exp_node_out"].copy_(eager["exp_node_out"])
            g.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                buf["exp_vals"], eager_vals, atol=2e-4, rtol=2e-4
            )
            assert torch.equal(buf["exp_tok_in"], eager_ids)

    def test_clone_fused_n1_clones_pingpong_plane(self):
        n, bw, width, k = 1, 4, 5, 4
        tok_out = torch.arange(n * bw * width).view(n, bw, width)
        node_out = torch.arange(n * bw).view(n, bw)
        buf = {
            "exp_pingpong": True,
            "exp_phase": 0,
            "exp_vals": torch.zeros(n, k),
            "exp_parents": torch.zeros(n, k, dtype=torch.int64),
            "exp_tokens": torch.zeros(n, k, dtype=torch.int64),
            "exp_indices": torch.zeros(n, k, dtype=torch.int64),
            "exp_tok_in": torch.zeros(n, bw, width, dtype=torch.int64),
            "exp_tok_out": tok_out,
            "exp_nodes": torch.zeros(n, bw, dtype=torch.int64),
            "exp_node_out": node_out,
        }
        fused = DecodeGraphRunner._clone_fused(
            object.__new__(DecodeGraphRunner), buf, n
        )
        assert fused.token_ids.data_ptr() != tok_out.data_ptr()
        assert torch.equal(fused.token_ids, tok_out)
        buf["exp_phase"] = 1
        buf["exp_tok_in"].copy_(
            torch.arange(100, 100 + n * bw * width).view(n, bw, width)
        )
        fused1 = DecodeGraphRunner._clone_fused(
            object.__new__(DecodeGraphRunner), buf, n
        )
        assert torch.equal(fused1.token_ids, buf["exp_tok_in"])
        assert fused1.token_ids.data_ptr() != buf["exp_tok_in"].data_ptr()

    def test_single_request_graph_replay_keeps_beam_diversity(self):
        if not try_load_beam_trie():
            pytest.skip("beam_trie JIT unavailable")
        device = torch.device("cuda")
        torch.manual_seed(13)
        n, bw, k_vocab, C, width = 1, 8, 16, 8, 5
        spec = {
            "cand_ids": torch.arange(k_vocab, device=device, dtype=torch.int64) + 10,
            "allow": torch.ones(4, 16, dtype=torch.uint8, device=device),
            "next_node": torch.zeros(4, 16, dtype=torch.int64, device=device),
            "token_base": 10,
            "invalid": 0,
        }
        buf = {
            "logprobs": torch.randn(
                n * bw, k_vocab, device=device, dtype=torch.float32
            ),
            "topk_lp": torch.empty(n, bw, C, device=device, dtype=torch.float32),
            "topk_idx": torch.empty(n, bw, C, device=device, dtype=torch.int64),
            "topk_tok": torch.empty(n, bw, C, device=device, dtype=torch.int64),
            "exp_cum": torch.linspace(-0.1, -2.0, bw, device=device).view(n, bw),
            "exp_nodes": torch.zeros(n, bw, dtype=torch.int64, device=device),
            "exp_tok_in": torch.arange(10, 10 + n * bw * width, device=device)
            .view(n, bw, width)
            .contiguous(),
            "exp_tok_out": torch.empty(n, bw, width, dtype=torch.int64, device=device),
            "exp_col": torch.ones(n, dtype=torch.int32, device=device),
            "exp_do": torch.ones(n, dtype=torch.uint8, device=device),
            "exp_node_out": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_vals": torch.empty(n, bw, dtype=torch.float32, device=device),
            "exp_parents": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_tokens": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_indices": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_scratch": torch.empty(n, bw * C, dtype=torch.float32, device=device),
        }
        with torch.inference_mode():
            for _ in range(3):
                _run_topk_expand(buf, spec, n, bw, C)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _run_topk_expand(buf, spec, n, bw, C)
            buf["logprobs"].copy_(torch.randn_like(buf["logprobs"]))
            buf["exp_cum"].copy_(
                torch.linspace(-0.05, -1.5, bw, device=device).view(n, bw)
            )
            buf["exp_tok_in"].copy_(
                torch.arange(20, 20 + n * bw * width, device=device).view(n, bw, width)
            )
            eager = {k: buf[k].clone() for k in ("logprobs", "exp_cum", "exp_tok_in")}
            _run_topk_expand(buf, spec, n, bw, C)
            eager_vals = buf["exp_vals"].clone()
            eager_tok = buf["exp_tokens"].clone()
            eager_ids = buf["exp_tok_out"].clone()
            assert int(eager_ids[0].unique(dim=0).shape[0]) > 1
            buf["logprobs"].copy_(eager["logprobs"])
            buf["exp_cum"].copy_(eager["exp_cum"])
            buf["exp_tok_in"].copy_(eager["exp_tok_in"])
            g.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                buf["exp_vals"], eager_vals, atol=2e-4, rtol=2e-4
            )
            assert torch.equal(buf["exp_tokens"], eager_tok)
            assert torch.equal(buf["exp_tok_out"], eager_ids)

    def test_second_replay_writes_new_token_column(self):
        """Wave-2 replay must still write col=1 (the fused SID bug was s_b=0)."""
        if not try_load_beam_trie():
            pytest.skip("beam_trie JIT unavailable")
        device = torch.device("cuda")
        torch.manual_seed(21)
        n, bw, k_vocab, C, width = 1, 8, 16, 8, 5
        spec = {
            "cand_ids": torch.arange(k_vocab, device=device, dtype=torch.int64) + 100,
            "allow": torch.ones(4, 16, dtype=torch.uint8, device=device),
            "next_node": torch.zeros(4, 16, dtype=torch.int64, device=device),
            "token_base": 100,
            "invalid": 0,
        }
        buf = {
            "logprobs": torch.randn(
                n * bw, k_vocab, device=device, dtype=torch.float32
            ),
            "topk_lp": torch.empty(n, bw, C, device=device, dtype=torch.float32),
            "topk_idx": torch.empty(n, bw, C, device=device, dtype=torch.int64),
            "topk_tok": torch.empty(n, bw, C, device=device, dtype=torch.int64),
            "exp_cum": torch.linspace(-0.1, -2.0, bw, device=device).view(n, bw),
            "exp_nodes": torch.zeros(n, bw, dtype=torch.int64, device=device),
            "exp_tok_in": torch.zeros(n, bw, width, dtype=torch.int64, device=device),
            "exp_tok_out": torch.zeros(n, bw, width, dtype=torch.int64, device=device),
            "exp_col": torch.ones(n, dtype=torch.int32, device=device),
            "exp_do": torch.ones(n, dtype=torch.uint8, device=device),
            "exp_node_out": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_vals": torch.empty(n, bw, dtype=torch.float32, device=device),
            "exp_parents": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_tokens": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_indices": torch.empty(n, bw, dtype=torch.int64, device=device),
            "exp_scratch": torch.empty(n, bw * C, dtype=torch.float32, device=device),
        }

        def _pack_and_replay(prefix_base: int):
            buf["exp_tok_in"].zero_()
            buf["exp_tok_in"][0, :, 0] = torch.arange(
                prefix_base, prefix_base + bw, device=device
            )
            buf["logprobs"].copy_(torch.randn_like(buf["logprobs"]))
            buf["exp_cum"].copy_(
                torch.linspace(-0.05, -1.5, bw, device=device).view(n, bw)
            )
            g.replay()
            return buf["exp_tok_out"][:n].clone()

        with torch.inference_mode():
            buf["exp_tok_in"][0, :, 0] = torch.arange(50, 50 + bw, device=device)
            for _ in range(3):
                _run_topk_expand(buf, spec, n, bw, C)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _run_topk_expand(buf, spec, n, bw, C)
            wave1 = _pack_and_replay(50)
            wave2 = _pack_and_replay(70)
            # col 0 is gathered from parents of the packed prefixes; col 1 is
            # the newly written token (the fused SID bug left this as 0).
            assert set(wave1[0, :, 0].tolist()) <= set(range(50, 50 + bw))
            assert set(wave2[0, :, 0].tolist()) <= set(range(70, 70 + bw))
            assert int((wave1[0, :, 1] == 0).sum()) == 0
            assert int((wave2[0, :, 1] == 0).sum()) == 0
            assert int(wave2[0].unique(dim=0).shape[0]) > 1
