import pytest
import torch
import torch.nn.functional as F

from flashrec.kernel.sglops import silu_and_mul
from flashrec.layers.linear import Linear
from flashrec.models.qwen3 import Qwen3Config, Qwen3MLP
from flashrec.models.weight import merge_gate_up_weights


class TestSiluAndMul:
    def test_matches_pytorch(self):
        torch.manual_seed(0)
        x = torch.randn(4, 16, dtype=torch.float32)
        d = x.shape[-1] // 2
        ref = F.silu(x[..., :d]) * x[..., d:]
        torch.testing.assert_close(silu_and_mul(x), ref, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_matches_cpu(self):
        torch.manual_seed(1)
        x = torch.randn(8, 32, dtype=torch.bfloat16)
        ref = silu_and_mul(x)
        out = silu_and_mul(x.cuda())
        torch.testing.assert_close(out.cpu().float(), ref.float(), atol=2e-2, rtol=2e-2)


class TestGateUpMerge:
    def test_merge_shapes(self):
        gate = torch.randn(6, 4)
        up = torch.randn(6, 4)
        gs = torch.ones(6)
        us = torch.full((6,), 2.0)
        w, scale = merge_gate_up_weights(gate, up, gs, us)
        assert tuple(w.shape) == (12, 4)
        torch.testing.assert_close(w[:6], gate)
        torch.testing.assert_close(w[6:], up)
        torch.testing.assert_close(scale[:6], gs)
        torch.testing.assert_close(scale[6:], us)

    def test_mlp_matches_split_silu(self):
        cfg = Qwen3Config(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=32,
            rms_norm_eps=1e-6,
            rope_theta=10000,
            max_position_embeddings=32,
            tie_word_embeddings=True,
        )
        mlp = Qwen3MLP(cfg)
        gate = torch.randn(16, 8)
        up = torch.randn(16, 8)
        down = torch.randn(8, 16)
        merged, _ = merge_gate_up_weights(gate, up)
        mlp.gate_up_proj.load(merged)
        mlp.down_proj.load(down)
        x = torch.randn(3, 8)
        got = mlp(x)
        ref = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)

    def test_nvfp4_load_rejected(self):
        lin = Linear(4, 4)
        packed = torch.zeros(4, 2, dtype=torch.uint8)
        scale = torch.ones(4)
        with pytest.raises(NotImplementedError):
            lin.load(packed, weight_scale=scale)
