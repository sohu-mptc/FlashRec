import pytest
import torch

from flashrec.kernel.sglops import apply_rope_inplace
from flashrec.layers.norm import RMSNorm
from flashrec.layers.rotary import RotaryEmbedding


class TestQkRopeKv:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fused_matches_sequential_fp8_store(self):
        from flashrec.kernel.qk_rope_kv import fused_qk_norm_rope_store_fp8
        from flashrec.kvcache.pool import TokenToKVPool

        torch.manual_seed(4)
        device = torch.device("cuda")
        t, hq, hk, d = 7, 16, 8, 128
        eps = 1e-6
        rotary = RotaryEmbedding(d, 2048, 1_000_000, device=device)
        q_norm = RMSNorm(d, eps).to(device)
        k_norm = RMSNorm(d, eps).to(device)
        q = torch.randn(t, hq, d, device=device, dtype=torch.bfloat16)
        k = torch.randn(t, hk, d, device=device, dtype=torch.bfloat16)
        v = torch.randn(t, hk, d, device=device, dtype=torch.bfloat16)
        pos = torch.arange(t, device=device, dtype=torch.int64) + 3
        loc = torch.arange(1, t + 1, device=device, dtype=torch.int64)

        q_ref = q_norm(q.clone())
        k_ref = k_norm(k.clone())
        q2 = q_ref.reshape(t, -1).contiguous()
        k2 = k_ref.reshape(t, -1).contiguous()
        apply_rope_inplace(pos, q2, k2, d, rotary.cos_sin_cache)
        pool = TokenToKVPool(1, t + 4, hk, d, torch.float8_e4m3fn, device)
        pool.store(0, k2.view(t, hk, d), v, loc)

        q_f = q.clone()
        k_f = k.clone()
        v_f = v.clone()
        k_cache = torch.zeros_like(pool.k[0])
        v_cache = torch.zeros_like(pool.v[0])
        ok = fused_qk_norm_rope_store_fp8(
            q_f,
            k_f,
            v_f,
            pos,
            loc,
            q_norm.weight,
            k_norm.weight,
            eps,
            rotary.cos_sin_cache,
            k_cache,
            v_cache,
        )
        assert ok
        torch.testing.assert_close(
            q_f.float(), q2.view(t, hq, d).float(), atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            k_cache[loc].float(), pool.k[0][loc].float(), atol=2e-1, rtol=2e-1
        )
        torch.testing.assert_close(
            v_cache[loc].float(), pool.v[0][loc].float(), atol=2e-1, rtol=2e-1
        )
