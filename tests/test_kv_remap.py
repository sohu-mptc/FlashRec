import torch

from flashrec.scheduler.kv_remap import classify_parents, remap_by_parents


class TestKvRemap:
    def test_identity(self):
        p = torch.arange(4)
        assert classify_parents(p, 4) == "identity"

    def test_permute(self):
        p = torch.tensor([2, 0, 3, 1])
        assert classify_parents(p, 4) == "permute"

    def test_fork(self):
        p = torch.tensor([0, 0, 1, 1])
        assert classify_parents(p, 4) == "fork"

    def test_gather_window(self):
        table = torch.arange(20).view(4, 5).clone()
        rows = torch.arange(4)
        parents = torch.tensor([0, 0, 1, 1])
        remap_by_parents(table, rows, parents, prefix_len=2, seq_len=4)
        # decode window cols 2:4 gathered by parents
        assert table[0, 2:4].tolist() == [2, 3]
        assert table[1, 2:4].tolist() == [2, 3]
        assert table[2, 2:4].tolist() == [7, 8]
        assert table[3, 2:4].tolist() == [7, 8]
        # prompt prefix unchanged
        assert table[1, 0].item() == 5

    def test_identity_gather_skips_classify(self):
        table = torch.arange(20).view(4, 5).clone()
        before = table.clone()
        kind = remap_by_parents(table, torch.arange(4), torch.arange(4), 2, 4)
        assert kind == "gather"
        assert table.tolist() == before.tolist()

    def test_shrink_parents_shorter_than_pool(self):
        """Trie-budget: pool stays at beam_width while survivors shrink."""
        table = torch.arange(40).view(8, 5).clone()
        rows = torch.arange(8)
        # Keep only 3 survivors from parents 0, 2, 2 among the live prefix.
        parents = torch.tensor([0, 2, 2])
        kind = remap_by_parents(table, rows, parents, prefix_len=2, seq_len=4)
        assert kind == "gather"
        assert table[0, 2:4].tolist() == [2, 3]
        assert table[1, 2:4].tolist() == [12, 13]
        assert table[2, 2:4].tolist() == [12, 13]
        # Unused pool rows and prompt prefix stay put.
        assert table[7, 2:4].tolist() == [37, 38]
        assert table[3, 0].item() == 15
