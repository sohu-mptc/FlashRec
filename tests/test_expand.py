import pytest
import torch

from flashrec.search.expand import (
    expand_step,
    init_from_prefill,
    init_from_prefill_batch,
    joint_select,
)
from flashrec.search.score import beam_score
from flashrec.search.trie import BeamValidPathTrie


class TestExpand:
    def test_beam_score_matches_sglang_formula(self):
        assert round(abs(beam_score(-4.0, 2, 1.0) - -2.0), 7) == 0
        assert round(abs(beam_score(-8.0, 4, 1.0) - -2.0), 7) == 0
        assert round(abs(beam_score(-8.0, 4, 0.0) - -8.0), 7) == 0

    def test_init_from_prefill_picks_top_n(self):
        logprobs = torch.tensor(
            [-10.0, -0.1, -0.2, -5.0, -0.3, -20.0], dtype=torch.float32
        )
        bl = init_from_prefill(
            logprobs,
            beam_width=3,
            beam_candidates=6,
            max_new_tokens=4,
            prompt_len=8,
        )
        assert bl.generated_len() == 1
        assert bl.prompt_len == 8
        toks = bl.last_tokens.tolist()
        assert toks == [1, 2, 4]
        assert bl.token_ids.shape[0] == 3

    def test_init_from_prefill_batch_b1_matches_single(self):
        logprobs = torch.tensor(
            [-10.0, -0.1, -0.2, -5.0, -0.3, -20.0], dtype=torch.float32
        )
        kwargs = dict(
            beam_width=3,
            beam_candidates=6,
            max_new_tokens=4,
        )
        single = init_from_prefill(logprobs, prompt_len=8, **kwargs)
        batched = init_from_prefill_batch(
            logprobs.unsqueeze(0), prompt_lens=[8], **kwargs
        )
        assert len(batched) == 1
        assert batched[0].last_tokens.tolist() == single.last_tokens.tolist()
        assert batched[0].token_ids.shape == single.token_ids.shape
        assert torch.equal(batched[0].token_ids, single.token_ids)
        assert len(set(batched[0].last_tokens.tolist())) == 3

    def test_expand_step_dense_fast_path(self):
        logprobs = torch.zeros(8, dtype=torch.float32)
        logprobs[3] = 0.0
        logprobs[5] = -0.5
        bl = init_from_prefill(
            logprobs, beam_width=2, beam_candidates=4, max_new_tokens=4, prompt_len=1
        )
        # Each of 2 beams has 3 candidates.
        top_tokens = torch.tensor([[1, 2, 7], [1, 2, 7]], dtype=torch.int64)
        top_logprobs = torch.tensor(
            [[-0.1, -1.0, -5.0], [-0.2, -0.3, -4.0]], dtype=torch.float32
        )
        parents = expand_step(
            bl, top_tokens, top_logprobs, beam_width=2, ignore_eos=True
        )
        assert parents is not None
        assert int(bl.generated_len()) == 2
        assert bl.last_tokens.numel() == 2

    def test_expand_step_accepts_padded_logprob_rows(self):
        logprobs = torch.zeros(8, dtype=torch.float32)
        logprobs[1] = 0.0
        logprobs[2] = -0.4
        bl = init_from_prefill(
            logprobs, beam_width=2, beam_candidates=4, max_new_tokens=4, prompt_len=1
        )
        top_tokens = torch.tensor(
            [[1, 2, 7], [1, 2, 7], [0, 0, 0], [0, 0, 0]], dtype=torch.int64
        )
        top_logprobs = torch.tensor(
            [
                [-0.1, -1.0, -5.0],
                [-0.2, -0.3, -4.0],
                [-9.0, -9.0, -9.0],
                [-9.0, -9.0, -9.0],
            ],
            dtype=torch.float32,
        )
        parents = expand_step(
            bl, top_tokens, top_logprobs, beam_width=2, ignore_eos=True
        )
        assert parents is not None
        assert bl.last_tokens.numel() == 2

    def test_expand_respects_trie_mask(self):
        trie = BeamValidPathTrie(mode="trie")
        trie.children[()] = {3}
        trie.children[(3,)] = {1}
        trie.children[(3, 1)] = set()
        trie.max_depth = 2
        trie.all_tokens = {1, 3}
        trie.finalize_index(token_base=1, vocab_size=4)

        logprobs = torch.full((6,), -20.0, dtype=torch.float32)
        logprobs[3] = 0.0
        logprobs[1] = -0.1
        bl = init_from_prefill(
            logprobs,
            beam_width=1,
            beam_candidates=4,
            max_new_tokens=4,
            prompt_len=1,
            valid_path=trie,
        )
        assert int(bl.last_tokens[0].item()) == 3
        top_tokens = torch.tensor([[1, 2, 4]], dtype=torch.int64)
        top_logprobs = torch.tensor([[-0.4, -0.01, -0.02]], dtype=torch.float32)
        expand_step(
            bl,
            top_tokens,
            top_logprobs,
            beam_width=1,
            valid_path=trie,
            ignore_eos=True,
        )
        # 2 and 4 are illegal after prefix [3]; token 1 must win despite worse raw score.
        assert int(bl.last_tokens[0].item()) == 1

    def test_genrec_cuda_matches_pytorch(self):
        from flashrec.kernel.beam_trie import (
            genrec_mask_topk_expand,
            try_load_beam_trie,
        )

        torch.manual_seed(2)
        n, bw, c, width = 2, 4, 6, 5
        cum = torch.randn(n, bw)
        top_lp = torch.randn(n, bw, c)
        top_lp, idx = torch.topk(top_lp, c, dim=-1, largest=True, sorted=True)
        top_tok = torch.randint(10, 20, (n, bw, c))
        nodes = torch.zeros(n, bw, dtype=torch.int64)
        allow = torch.ones(4, 16, dtype=torch.uint8)
        token_ids = torch.randint(10, 20, (n, bw, width))
        next_node = torch.zeros(4, 16, dtype=torch.int64)
        cpu = genrec_mask_topk_expand(
            cum,
            top_lp,
            top_tok,
            select_k=bw,
            node_ids=nodes,
            allow_table=allow,
            token_base=10,
            token_ids=token_ids,
            col=1,
            next_node=next_node,
            invalid_node=0,
        )
        if not torch.cuda.is_available():
            assert tuple(cpu.vals.shape) == (n, bw)
            return
        loaded = try_load_beam_trie()
        gpu = genrec_mask_topk_expand(
            cum.cuda(),
            top_lp.cuda(),
            top_tok.cuda(),
            select_k=bw,
            node_ids=nodes.cuda(),
            allow_table=allow.cuda(),
            token_base=10,
            token_ids=token_ids.cuda(),
            col=1,
            next_node=next_node.cuda(),
            invalid_node=0,
        )
        torch.testing.assert_close(gpu.vals.cpu(), cpu.vals, atol=1e-5, rtol=1e-5)
        assert torch.equal(gpu.tokens.cpu(), cpu.tokens)
        assert torch.equal(gpu.parents.cpu(), cpu.parents)
        if loaded:
            assert torch.equal(gpu.token_ids.cpu(), cpu.token_ids)

    def test_col_tensor_matches_int(self):
        from flashrec.kernel.beam_trie import genrec_mask_topk_expand

        torch.manual_seed(5)
        n, bw, c, width = 2, 4, 6, 5
        cum = torch.randn(n, bw)
        top_lp = torch.randn(n, bw, c)
        top_lp, _ = torch.topk(top_lp, c, dim=-1, largest=True, sorted=True)
        top_tok = torch.randint(10, 20, (n, bw, c))
        nodes = torch.zeros(n, bw, dtype=torch.int64)
        allow = torch.ones(4, 16, dtype=torch.uint8)
        token_ids = torch.randint(10, 20, (n, bw, width))
        next_node = torch.zeros(4, 16, dtype=torch.int64)
        kwargs = dict(
            select_k=bw,
            node_ids=nodes,
            allow_table=allow,
            token_base=10,
            next_node=next_node,
            invalid_node=0,
        )
        as_int = genrec_mask_topk_expand(
            cum, top_lp, top_tok, token_ids=token_ids.clone(), col=1, **kwargs
        )
        as_t = genrec_mask_topk_expand(
            cum,
            top_lp,
            top_tok,
            token_ids=token_ids.clone(),
            col=torch.tensor([1, 1], dtype=torch.int32),
            **kwargs,
        )
        torch.testing.assert_close(as_t.vals, as_int.vals)
        assert torch.equal(as_t.tokens, as_int.tokens)
        assert torch.equal(as_t.token_ids, as_int.token_ids)
        mixed = genrec_mask_topk_expand(
            cum,
            top_lp,
            top_tok,
            token_ids=token_ids.clone(),
            col=torch.tensor([1, 2], dtype=torch.int32),
            **kwargs,
        )
        assert tuple(mixed.token_ids.shape) == (n, bw, width)
        assert not torch.equal(mixed.token_ids, as_int.token_ids)

    def test_genrec_cuda_skips_masked_heads(self):
        from flashrec.kernel.beam_trie import (
            genrec_mask_topk_expand,
            try_load_beam_trie,
        )

        if not torch.cuda.is_available() or not try_load_beam_trie():
            pytest.skip("beam_trie JIT unavailable")
        device = torch.device("cuda")
        n, bw, c, k = 1, 4, 4, 4
        n_nodes, vsz, base = 8, 16, 0
        allow = torch.zeros((n_nodes, vsz), device=device, dtype=torch.bool)
        allow[:, 2:] = True
        cum = torch.zeros(n, bw, device=device)
        logprobs = torch.zeros(n, bw, c, device=device)
        for b in range(bw):
            logprobs[0, b] = torch.tensor(
                [0.0, -1.0, -2.0 - 0.01 * b, -3.0 - 0.01 * b], device=device
            )
        tokens = (
            torch.arange(c, device=device, dtype=torch.int64)
            .view(1, 1, c)
            .expand(n, bw, c)
            .contiguous()
        )
        nodes = torch.zeros(n, bw, device=device, dtype=torch.int64)
        gpu = genrec_mask_topk_expand(
            cum,
            logprobs,
            tokens,
            select_k=k,
            node_ids=nodes,
            allow_table=allow,
            token_base=base,
            apply_expand=False,
            apply_advance=False,
        )
        cpu = genrec_mask_topk_expand(
            cum.cpu(),
            logprobs.cpu(),
            tokens.cpu(),
            select_k=k,
            node_ids=nodes.cpu(),
            allow_table=allow.cpu(),
            token_base=base,
            apply_expand=False,
            apply_advance=False,
        )
        torch.testing.assert_close(gpu.vals.cpu(), cpu.vals, atol=1e-5, rtol=1e-5)
        assert torch.equal(gpu.tokens.cpu(), cpu.tokens)
        assert torch.equal(gpu.parents.cpu(), cpu.parents)

    def test_genrec_cuda_production_shape(self):
        from flashrec.kernel.beam_trie import (
            genrec_mask_topk_expand,
            try_load_beam_trie,
        )

        if not torch.cuda.is_available() or not try_load_beam_trie():
            pytest.skip("beam_trie JIT unavailable")
        torch.manual_seed(11)
        n, bw, c, width = 2, 50, 100, 17  # odd L hits scalar expand path
        device = torch.device("cuda")
        cum = torch.randn(n, bw, device=device)
        top_lp = torch.randn(n, bw, c, device=device)
        top_lp, perm = torch.sort(top_lp, dim=-1, descending=True)
        top_tok = torch.randint(10, 40, (n, bw, c), device=device)
        top_tok = top_tok.gather(-1, perm)
        nodes = torch.randint(0, 8, (n, bw), device=device, dtype=torch.int64)
        allow = torch.rand(8, 48, device=device) > 0.3
        token_ids = torch.randint(10, 40, (n, bw, width), device=device)
        next_node = torch.randint(0, 8, (8, 48), device=device, dtype=torch.int64)
        col = torch.tensor([3, 8], device=device, dtype=torch.int32)
        gpu = genrec_mask_topk_expand(
            cum,
            top_lp,
            top_tok,
            select_k=bw,
            node_ids=nodes,
            allow_table=allow,
            token_base=10,
            token_ids=token_ids,
            col=col,
            next_node=next_node,
            invalid_node=0,
        )
        cpu = genrec_mask_topk_expand(
            cum.cpu(),
            top_lp.cpu(),
            top_tok.cpu(),
            select_k=bw,
            node_ids=nodes.cpu(),
            allow_table=allow.cpu(),
            token_base=10,
            token_ids=token_ids.cpu(),
            col=col.cpu(),
            next_node=next_node.cpu(),
            invalid_node=0,
        )
        torch.testing.assert_close(gpu.vals.cpu(), cpu.vals, atol=1e-4, rtol=1e-4)
        assert torch.equal(gpu.tokens.cpu(), cpu.tokens)
        assert torch.equal(gpu.parents.cpu(), cpu.parents)
        assert torch.equal(gpu.token_ids.cpu(), cpu.token_ids)
        assert torch.equal(gpu.node_ids.cpu(), cpu.node_ids)

    def test_pick_pingpong_stack_skips_aliased_rows(self):
        from flashrec.kernel.beam_trie import (
            GenrecFusedWorkspace,
            pick_pingpong_stack,
            tensors_alias,
        )

        ws = GenrecFusedWorkspace()
        device = torch.device("cpu")
        rows = [torch.arange(i * 6, i * 6 + 6).view(2, 3) for i in range(2)]
        src = pick_pingpong_stack(ws, "tok", rows, (2, 2, 3), torch.int64, device)
        assert torch.equal(src[0], rows[0])
        live = [src[i] for i in range(2)]
        src2 = pick_pingpong_stack(ws, "tok", live, (2, 2, 3), torch.int64, device)
        assert tensors_alias(src, src2)
        other = [torch.arange(100, 106).view(2, 3), torch.arange(200, 206).view(2, 3)]
        packed = pick_pingpong_stack(ws, "tok", other, (2, 2, 3), torch.int64, device)
        assert torch.equal(packed[0], other[0])
        assert torch.equal(packed[1], other[1])

    def test_workspace_canonicalizes_cuda_device(self):
        from flashrec.kernel.beam_trie import GenrecFusedWorkspace, tensors_alias

        if not torch.cuda.is_available():
            pytest.skip("cuda")
        ws = GenrecFusedWorkspace()
        a = ws.get("tok_a", (2, 4, 8), torch.int64, torch.device("cuda"))
        b = ws.get("tok_a", (2, 4, 8), torch.int64, torch.device("cuda:0"))
        assert tensors_alias(a, b)

    def test_genrec_cuda_pingpong_flips_and_matches_cpu(self):
        from flashrec.kernel.beam_trie import (
            GenrecFusedWorkspace,
            genrec_mask_topk_expand,
            tensors_alias,
            try_load_beam_trie,
        )

        if not torch.cuda.is_available() or not try_load_beam_trie():
            pytest.skip("beam_trie JIT unavailable")
        torch.manual_seed(21)
        n, bw, c, width = 2, 4, 6, 8
        device = torch.device("cuda")
        cum = torch.randn(n, bw, device=device)
        top_lp = torch.randn(n, bw, c, device=device)
        top_lp, perm = torch.sort(top_lp, dim=-1, descending=True)
        top_tok = torch.randint(10, 40, (n, bw, c), device=device)
        top_tok = top_tok.gather(-1, perm)
        nodes = torch.randint(0, 4, (n, bw), device=device, dtype=torch.int64)
        allow = torch.ones(4, 48, device=device, dtype=torch.uint8)
        token_ids = torch.randint(10, 40, (n, bw, width), device=device)
        next_node = torch.randint(0, 4, (4, 48), device=device, dtype=torch.int64)
        ws = GenrecFusedWorkspace()
        tok = token_ids
        node = nodes
        cpu_tok = token_ids.cpu()
        cpu_node = nodes.cpu()
        cpu_cum = cum.cpu()
        ptrs = []
        for step in range(4):
            col = 1 + step
            gpu = genrec_mask_topk_expand(
                cum,
                top_lp,
                top_tok,
                select_k=bw,
                node_ids=node,
                allow_table=allow,
                token_base=10,
                token_ids=tok,
                col=col,
                next_node=next_node,
                invalid_node=0,
                workspace=ws,
            )
            cpu = genrec_mask_topk_expand(
                cpu_cum,
                top_lp.cpu(),
                top_tok.cpu(),
                select_k=bw,
                node_ids=cpu_node,
                allow_table=allow.cpu(),
                token_base=10,
                token_ids=cpu_tok,
                col=col,
                next_node=next_node.cpu(),
                invalid_node=0,
            )
            torch.testing.assert_close(gpu.vals.cpu(), cpu.vals, atol=1e-5, rtol=1e-5)
            assert torch.equal(gpu.token_ids.cpu(), cpu.token_ids)
            assert torch.equal(gpu.node_ids.cpu(), cpu.node_ids)
            assert not tensors_alias(gpu.token_ids, tok)
            assert not tensors_alias(gpu.node_ids, node)
            ptrs.append((gpu.token_ids.data_ptr(), gpu.node_ids.data_ptr()))
            tok = gpu.token_ids
            node = gpu.node_ids
            cum = gpu.vals.clone()
            cpu_tok = cpu.token_ids
            cpu_node = cpu.node_ids
            cpu_cum = cpu.vals
        assert ptrs[0][0] != ptrs[1][0]
        assert ptrs[0][0] == ptrs[2][0]
        assert ptrs[1][0] == ptrs[3][0]
        assert ptrs[0][1] != ptrs[1][1]
        assert ptrs[0][1] == ptrs[2][1]

    def test_joint_select_stops_on_eos(self):
        cum = torch.zeros(2, dtype=torch.float32)
        lp = torch.tensor([[-0.1, -1.0], [-0.2, -0.3]], dtype=torch.float32)
        toks = torch.tensor([[9, 1], [2, 3]], dtype=torch.int64)
        stop = torch.tensor([9], dtype=torch.int64)
        res = joint_select(cum, lp, toks, stop, beam_width=2)
        assert int(res.num_finished.item()) >= 1
        assert int(res.num_survivors.item()) >= 1

    def test_select_final_topk_clamps_short_frontier(self):
        from flashrec.search.select import select_final_topk

        cum = torch.zeros(2, dtype=torch.float32)
        lp = torch.tensor([[-0.1, -1.0], [-0.2, -0.3]], dtype=torch.float32)
        toks = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        # beam_width larger than the flat candidate count must not raise.
        sel = select_final_topk(cum, lp, toks, beam_width=16)
        assert int(sel.tokens.numel()) == 4

    def test_genrec_cuda_wide_beam_1000(self):
        """n=1000 requires kBeamTrieBlock >= 1024 after the wide-beam smem bump."""
        from flashrec.kernel.beam_trie import (
            genrec_mask_topk_expand,
            try_load_beam_trie,
        )

        if not torch.cuda.is_available() or not try_load_beam_trie():
            pytest.skip("beam_trie JIT unavailable")
        torch.manual_seed(7)
        n, bw, c = 1, 1000, 4
        cum = torch.randn(n, bw, device="cuda")
        top_lp = torch.randn(n, bw, c, device="cuda")
        top_lp, _ = torch.topk(top_lp, c, dim=-1, largest=True, sorted=True)
        top_tok = torch.randint(10, 20, (n, bw, c), device="cuda")
        nodes = torch.zeros(n, bw, dtype=torch.int64, device="cuda")
        allow = torch.ones(2, 16, dtype=torch.uint8, device="cuda")
        token_ids = torch.randint(10, 20, (n, bw, 3), device="cuda")
        next_node = torch.zeros(2, 16, dtype=torch.int64, device="cuda")
        out = genrec_mask_topk_expand(
            cum,
            top_lp,
            top_tok,
            select_k=bw,
            node_ids=nodes,
            allow_table=allow,
            token_base=10,
            token_ids=token_ids,
            col=1,
            next_node=next_node,
            invalid_node=0,
        )
        assert tuple(out.vals.shape) == (n, bw)
        assert tuple(out.parents.shape) == (n, bw)

    def test_split_select_matches_fused(self):
        """The radix-select path must pick exactly what the K-way merge picks.

        genrec_mask_topk_expand selects survivors with one warp-merge round per
        output, so it costs ~2.3us * select_k; at RecIF's select_k=beam_width
        =1000 that is 2.26ms and 97.5% of the kernel. call_genrec_split does the
        same masked-plane argmax with torch.topk in 64us. Both take the max of
        the same plane, so the outputs must agree bit for bit -- including the
        parent/token/index reported for dead slots that had no legal child.
        """
        from flashrec.kernel import beam_trie as bt

        if not torch.cuda.is_available() or not bt.try_load_beam_trie():
            pytest.skip("beam_trie JIT unavailable")
        dev = torch.device("cuda")
        # frac controls how many children are legal; 0.0 makes every slot dead
        # and 0.02 leaves a partially dead plane, which is where the two paths
        # last disagreed.
        for n, bw, c, k, frac, seed in (
            (1, 1000, 16, 1000, 0.5, 0),
            (1, 1000, 16, 500, 0.5, 1),
            (4, 1000, 16, 1000, 0.5, 2),
            (2, 256, 16, 256, 0.5, 3),
            (1, 1000, 16, 1000, 0.02, 4),
            (1, 1000, 16, 1000, 0.0, 5),
            (1, 1000, 16, 1, 0.5, 6),
        ):
            g = torch.Generator().manual_seed(seed)
            vsz, n_nodes, width = 64, 512, 6
            cum = torch.randn(n, bw, generator=g).to(dev)
            lp = torch.randn(n, bw, c, generator=g).log_softmax(-1)
            # The fused merge assumes each row is sorted descending, which is
            # what the caller's topk(sorted=True) produces.
            lp, order = torch.sort(lp.to(dev), dim=-1, descending=True)
            top_tok = (
                torch.randint(0, vsz, (n, bw, c), generator=g, dtype=torch.int64)
                .to(dev)
                .gather(-1, order)
                .contiguous()
            )
            kwargs = dict(
                select_k=k,
                node_ids=torch.randint(
                    0, n_nodes, (n, bw), generator=g, dtype=torch.int64
                ).to(dev),
                allow_table=(torch.rand(n_nodes, vsz, generator=g) < frac)
                .to(torch.uint8)
                .to(dev),
                token_base=0,
                token_ids=torch.randint(
                    0, vsz, (n, bw, width), generator=g, dtype=torch.int64
                ).to(dev),
                col=torch.full((n,), 2, dtype=torch.int32, device=dev),
                next_node=torch.randint(
                    0, n_nodes, (n_nodes, vsz), generator=g, dtype=torch.int64
                ).to(dev),
                invalid_node=0,
            )
            got = {}
            saved = bt.GENREC_SERIAL_MERGE_MAX_K
            try:
                for name, threshold in (("fused", 1 << 30), ("split", 0)):
                    bt.GENREC_SERIAL_MERGE_MAX_K = threshold
                    got[name] = bt.genrec_mask_topk_expand(
                        cum.contiguous(),
                        lp.contiguous(),
                        top_tok,
                        workspace=bt.GenrecFusedWorkspace(),
                        **kwargs,
                    )
            finally:
                bt.GENREC_SERIAL_MERGE_MAX_K = saved
            f, s = got["fused"], got["split"]
            why = f"n={n} bw={bw} k={k} frac={frac}"
            # -inf slots must line up before comparing values: -inf - -inf is nan.
            live = torch.isfinite(f.vals)
            assert torch.equal(live, torch.isfinite(s.vals)), why
            assert torch.equal(f.vals[live], s.vals[live]), why
            for field in ("parents", "tokens", "indices", "token_ids", "node_ids"):
                assert torch.equal(
                    getattr(f, field), getattr(s, field)
                ), f"{field} differs at {why}"

    def test_split_select_used_at_wide_beam(self):
        """Wide beams must actually take the split path, not just tolerate it."""
        from flashrec.kernel import beam_trie as bt

        assert bt.GENREC_SERIAL_MERGE_MAX_K < 1000
        assert bt.GENREC_SERIAL_MERGE_MAX_K >= 1
