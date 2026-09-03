import pytest
import torch

from flashrec.kernel.sglops import per_token_quant_fp8, silu_and_mul


class TestSiluFp8:
    def _dequant(self, q, scale):
        return q.float() * scale.float()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_silu_quant_matches_sequential(self):
        from flashrec.kernel.silu_fp8 import silu_and_mul_per_token_quant_fp8

        torch.manual_seed(0)
        x = torch.randn(12, 96, device="cuda", dtype=torch.bfloat16)
        y = silu_and_mul(x)
        q_ref, s_ref = per_token_quant_fp8(y)
        q, s = silu_and_mul_per_token_quant_fp8(x)
        torch.testing.assert_close(
            self._dequant(q, s), self._dequant(q_ref, s_ref), atol=2e-1, rtol=2e-1
        )
