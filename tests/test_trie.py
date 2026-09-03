import torch

from flashrec.search.trie import BeamValidPathTrie, build_beam_valid_path


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
