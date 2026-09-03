import pytest
import torch

from flashrec.attention.kv_indices import gather_kv_indices
from flashrec.kvcache.pool import ReqToTokenPool, TokenToKVPool


class TestGatherAndPool:
    def test_gather_ragged(self):
        table = torch.arange(20, dtype=torch.int32).view(4, 5)
        rows = torch.tensor([0, 2, 3])
        seq = [3, 1, 4]
        out = gather_kv_indices(table, rows, seq)
        assert out.tolist() == [0, 1, 2, 10, 15, 16, 17, 18]

    def test_gather_gpu_seq_lens_with_total(self):
        if not torch.cuda.is_available():
            pytest.skip("cuda")
        table = torch.arange(20, dtype=torch.int32, device="cuda").view(4, 5)
        rows = torch.tensor([0, 2, 3], device="cuda")
        seq = torch.tensor([3, 1, 4], dtype=torch.int32, device="cuda")
        buf = torch.full((32,), -1, dtype=torch.int32, device="cuda")
        out = gather_kv_indices(table, rows, seq, out=buf, total=8)
        assert out.tolist() == [0, 1, 2, 10, 15, 16, 17, 18]
        assert int(buf[8].item()) == -1

    def test_copy_list_fills_cpu_dest_inplace(self):
        from flashrec.engine.staging import PinnedStage

        st = PinnedStage()
        dest = torch.empty(8, dtype=torch.int64, pin_memory=True)
        dest.fill_(-1)
        gpu, pin = st.copy_list(
            "seq_cpu", [3, 1, 4], torch.device("cpu"), torch.int64, dest=dest[:3]
        )
        assert dest[:3].tolist() == [3, 1, 4]
        assert int(dest[3].item()) == -1
        assert gpu.data_ptr() == dest[:3].data_ptr()
        assert pin.data_ptr() == dest[:3].data_ptr()

    def test_copy_rows_skips_cat(self):
        from flashrec.engine.staging import PinnedStage

        st = PinnedStage()
        a = torch.tensor([1, 2], dtype=torch.int64)
        b = torch.tensor([3], dtype=torch.int64)
        dest = torch.full((8,), -1, dtype=torch.int64)
        out = st.copy_rows("rows", [a, b], torch.device("cpu"), torch.int64, dest=dest)
        assert out.tolist() == [1, 2, 3]
        assert int(dest[3].item()) == -1

    def test_token_pool_dummy_slot(self):
        pool = TokenToKVPool(
            num_layers=1,
            num_tokens=8,
            num_kv_heads=1,
            head_dim=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        loc = pool.alloc(3)
        assert loc is not None
        assert int(loc.min()) >= 1
        pool.free(loc)
        assert pool.available_size() >= 3

    def test_req_pool_dummy_slot(self):
        pool = ReqToTokenPool(8, 16, torch.device("cpu"))
        idx = pool.alloc(2)
        assert int(idx.min()) >= 1
        pool.free(idx)
        assert pool.available_size() >= 2
