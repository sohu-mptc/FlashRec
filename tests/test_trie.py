import torch

from flashrec.search.trie import (
    BeamValidPathTrie,
    _codes_to_token_ids,
    build_beam_valid_path,
)


class TestBeamValidPath:
    def _tiny_trie(self) -> BeamValidPathTrie:
        trie = BeamValidPathTrie(mode="trie")
        trie.children[()] = {10, 11}
        trie.children[(10,)] = {20, 21}
        trie.children[(11,)] = {20}
        trie.children[(10, 20)] = set()
        trie.children[(10, 21)] = set()
        trie.children[(11, 20)] = set()
        trie.max_depth = 2
        trie.all_tokens = {10, 11, 20, 21}
        trie.finalize_index(token_base=10, vocab_size=12)
        return trie

    def test_trie_next_node_resolves_without_d2h_logic(self):
        trie = self._tiny_trie()
        device = torch.device("cpu")
        trie._ensure_gpu_cache(device)
        prefixes = torch.tensor([[10, 20], [10, 0], [11, 20]], dtype=torch.int64)
        nodes = trie._resolve_node_ids(prefixes, bw=3, cur_len=1, device=device)
        n10 = trie._prefix_to_node[(10,)]
        n11 = trie._prefix_to_node[(11,)]
        assert int(nodes[0].item()) == n10
        assert int(nodes[1].item()) == n10
        assert int(nodes[2].item()) == n11

        scores = torch.zeros((3, 2), dtype=torch.float32)
        cands = torch.tensor([[20, 21], [20, 99], [20, 21]], dtype=torch.int64)
        out = trie.mask_candidates(prefixes, cands, scores.clone(), cur_len=1)
        assert torch.isfinite(out[0, 0])
        assert torch.isfinite(out[0, 1])
        assert torch.isfinite(out[1, 0])
        assert torch.isinf(out[1, 1])
        assert torch.isfinite(out[2, 0])
        assert torch.isinf(out[2, 1])

    def test_incremental_node_ids_advance(self):
        trie = self._tiny_trie()
        device = torch.device("cpu")
        trie._ensure_gpu_cache(device)
        prefixes = torch.tensor([[10], [11], [10]], dtype=torch.int64)
        nodes = trie.bootstrap_node_ids(prefixes, bw=3, cur_len=1, device=device)
        parents = torch.tensor([0, 1, 2], dtype=torch.int64)
        toks = torch.tensor([20, 20, 21], dtype=torch.int64)
        nxt = trie.advance_node_ids(nodes, parents, toks)
        assert int(nxt[0].item()) == trie._prefix_to_node[(10, 20)]
        assert int(nxt[1].item()) == trie._prefix_to_node[(11, 20)]
        assert int(nxt[2].item()) == trie._prefix_to_node[(10, 21)]

    def test_codebook_and_flat_factory(self):
        ids = list(range(100, 112))
        codebook = build_beam_valid_path(
            codebook_sizes=[4, 4, 4], special_token_ids=ids
        )
        assert codebook.mode == "codebook"
        assert codebook.max_depth == 3
        assert codebook.allowed_next([], 0) == set(ids[0:4])
        assert codebook.allowed_next([], 1) == set(ids[4:8])

        flat = build_beam_valid_path(special_token_ids=ids)
        assert flat.mode == "flat"
        assert flat.allowed_next([1, 2, 3]) == set(ids)

    def test_build_from_sequences(self):
        trie = build_beam_valid_path()
        assert trie.mode == "none"
        trie = build_beam_valid_path(sid_file=None, special_token_ids=None)
        assert not trie.active


class TestLevelNodeOffsets:
    def _codebook_trie(self, codebook_sizes=(4, 4, 4)):
        """Build a trie from all valid 3-level codebook sequences."""
        token_base = 100
        children = {}
        all_tokens = set()
        max_depth = len(codebook_sizes)
        for c0 in range(codebook_sizes[0]):
            t0 = token_base + c0
            all_tokens.add(t0)
            children.setdefault((), set()).add(t0)
            for c1 in range(codebook_sizes[1]):
                offset1 = codebook_sizes[0]
                t1 = token_base + offset1 + c1
                all_tokens.add(t1)
                children.setdefault((t0,), set()).add(t1)
                for c2 in range(codebook_sizes[2]):
                    offset2 = codebook_sizes[0] + codebook_sizes[1]
                    t2 = token_base + offset2 + c2
                    all_tokens.add(t2)
                    children.setdefault((t0, t1), set()).add(t2)
                    children.setdefault((t0, t1, t2), set())
        trie = BeamValidPathTrie(
            children=children,
            all_tokens=all_tokens,
            max_depth=max_depth,
            mode="trie",
            token_base=token_base,
            vocab_size=sum(codebook_sizes),
        )
        trie.finalize_index(token_base=token_base, vocab_size=sum(codebook_sizes))
        return trie, codebook_sizes, token_base

    def test_level_node_offsets_contiguous(self):
        trie, cb, _ = self._codebook_trie()
        offsets = trie.level_node_offsets()
        # Trie has max_depth=3 depths of internal nodes (0,1,2) plus
        # depth-3 leaf entries from empty-children markers.
        assert len(offsets) == len(cb) + 1
        assert offsets[0] == 0
        for d in range(len(offsets)):
            count = trie.level_node_count(d)
            assert count > 0
            if d + 1 < len(offsets):
                assert offsets[d] + count == offsets[d + 1]

    def test_build_level_tables_all_levels_fit(self):
        trie, cb, token_base = self._codebook_trie(codebook_sizes=(4, 4, 2))
        tables = trie.build_level_tables(cb, device=torch.device("cpu"))
        assert set(tables.keys()) == {0, 1, 2}
        for level, (allow_t, next_t, level_tb, cand_ids) in tables.items():
            k = cb[level]
            n_nodes = trie.level_node_count(level)
            assert allow_t.shape == (n_nodes + 1, k)
            assert next_t.shape == (n_nodes + 1, k)
            assert cand_ids.shape == (k,)
            expected_base = token_base + sum(cb[:level])
            assert level_tb == expected_base
            assert cand_ids[0].item() == expected_base
            assert cand_ids[-1].item() == expected_base + k - 1

    def test_level_allow_table_matches_full(self):
        """Per-level allow_table entries must match the full trie's allow_table."""
        trie, cb, token_base = self._codebook_trie(codebook_sizes=(3, 3, 2))
        trie._ensure_gpu_cache(torch.device("cpu"))
        full_allow = trie._allow_table
        full_next = trie._next_node
        assert full_allow is not None

        tables = trie.build_level_tables(cb, device=torch.device("cpu"))
        offsets = trie.level_node_offsets()
        p2n = trie._prefix_to_node

        for level, (allow_t, next_t, level_tb, _) in tables.items():
            level_start = offsets[level]
            k = cb[level]
            cb_offset = sum(cb[:level])
            for prefix, global_nid in p2n.items():
                if len(prefix) != level:
                    continue
                local_nid = global_nid - level_start
                for rel in range(k):
                    global_rel = cb_offset + rel
                    assert allow_t[local_nid, rel].item() == full_allow[global_nid, global_rel].item(), (
                        f"mismatch at level={level} node={global_nid} rel={rel}"
                    )
                    if allow_t[local_nid, rel].item():
                        full_child = full_next[global_nid, global_rel].item()
                        level_child = next_t[local_nid, rel].item()
                        assert level_child == full_child, (
                            f"next_node mismatch at level={level} node={global_nid} "
                            f"rel={rel}: {level_child} != {full_child}"
                        )

    def test_node_id_remapping_roundtrip(self):
        """local_id = global_id - offset; next_node values are global."""
        trie, cb, token_base = self._codebook_trie(codebook_sizes=(3, 3, 2))
        tables = trie.build_level_tables(cb, device=torch.device("cpu"))
        offsets = trie.level_node_offsets()
        p2n = trie._prefix_to_node

        for level in range(len(cb)):
            if level not in tables:
                continue
            _, next_t, level_tb, _ = tables[level]
            level_start = offsets[level]
            for prefix, global_nid in p2n.items():
                if len(prefix) != level:
                    continue
                local_nid = global_nid - level_start
                assert local_nid >= 0
                for tok in trie.children.get(prefix, ()):
                    rel = int(tok) - level_tb
                    if 0 <= rel < cb[level]:
                        child_global = next_t[local_nid, rel].item()
                        child_key = prefix + (int(tok),)
                        expected = p2n.get(child_key)
                        if expected is not None:
                            assert child_global == expected

    def test_build_level_tables_non_trie_returns_empty(self):
        trie = BeamValidPathTrie(mode="codebook")
        trie.finalize_index()
        tables = trie.build_level_tables([4, 4])
        assert tables == {}

    def test_build_level_tables_respects_dense_max_cells(self):
        """When a level is too large for dense, it should be skipped."""
        import flashrec.search.trie as trie_mod
        trie, cb, _ = self._codebook_trie(codebook_sizes=(4, 4, 4))
        old_max = trie_mod._DENSE_MAX_CELLS
        try:
            trie_mod._DENSE_MAX_CELLS = 5
            tables = trie.build_level_tables(cb, device=torch.device("cpu"))
            for level in tables:
                n = trie.level_node_count(level)
                assert (n + 1) * cb[level] <= 5
        finally:
            trie_mod._DENSE_MAX_CELLS = old_max
