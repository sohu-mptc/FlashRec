import pytest
import torch

from flashrec.kernel.sglops import fused_add_rmsnorm, per_token_quant_fp8, rmsnorm
from flashrec.layers.linear import Linear


class TestRmsFp8:
    def _dequant(self, q, scale):
        return q.float() * scale.float()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_rmsnorm_quant_matches_sequential(self):
        from flashrec.kernel.rms_fp8 import rmsnorm_per_token_quant_fp8

        torch.manual_seed(0)
        x = torch.randn(16, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        hidden = rmsnorm(x, w, 1e-6)
        q_ref, s_ref = per_token_quant_fp8(hidden)
        q, s = rmsnorm_per_token_quant_fp8(x, w, 1e-6)
        torch.testing.assert_close(
            self._dequant(q, s), self._dequant(q_ref, s_ref), atol=2e-1, rtol=2e-1
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fused_add_quant_matches_sequential(self):
        from flashrec.kernel.rms_fp8 import fused_add_rmsnorm_per_token_quant_fp8

        torch.manual_seed(1)
        x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(2048, device="cuda", dtype=torch.bfloat16)
        y_ref, res_ref = fused_add_rmsnorm(x.clone(), residual.clone(), w, 1e-6)
        q_ref, s_ref = per_token_quant_fp8(y_ref)
        q, s, res = fused_add_rmsnorm_per_token_quant_fp8(
            x.clone(), residual.clone(), w, 1e-6
        )
        torch.testing.assert_close(res.float(), res_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            self._dequant(q, s), self._dequant(q_ref, s_ref), atol=2e-1, rtol=2e-1
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_shared_qkv_quant_matches_three_linears(self):
        torch.manual_seed(2)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        q_lin = Linear(64, 128)
        k_lin = Linear(64, 32)
        v_lin = Linear(64, 32)
        q_lin.load(
            torch.randn(128, 64, device="cuda", dtype=torch.bfloat16), quantize_fp8=True
        )
        k_lin.load(
            torch.randn(32, 64, device="cuda", dtype=torch.bfloat16), quantize_fp8=True
        )
        v_lin.load(
            torch.randn(32, 64, device="cuda", dtype=torch.bfloat16), quantize_fp8=True
        )
        q_fp8, scale = per_token_quant_fp8(x)
        shared_q = q_lin.forward_fp8(q_fp8, scale, x.shape, x.dtype)
        shared_k = k_lin.forward_fp8(q_fp8, scale, x.shape, x.dtype)
        torch.testing.assert_close(shared_q, q_lin(x), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(shared_k, k_lin(x), atol=2e-2, rtol=2e-2)
