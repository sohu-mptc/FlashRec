"""Qwen3 (TP=1) with optional qk-norm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from flashrec.attention.flashinfer import AttentionBackend
from flashrec.core import ForwardBatch
from flashrec.kernel.qk_rope_kv import fused_qk_norm_rope_store_fp8
from flashrec.kernel.sglops import apply_rope_and_store_kv, silu_and_mul
from flashrec.kernel.silu_fp8 import silu_and_mul_per_token_quant_fp8
from flashrec.layers.linear import Linear
from flashrec.layers.norm import RMSNorm
from flashrec.layers.rotary import RotaryEmbedding


@dataclass
class Qwen3Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        cfg: Qwen3Config,
        layer_id: int,
        attn: AttentionBackend,
        rotary: RotaryEmbedding,
        fused_qk_rope_kv: bool = True,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.head_dim = cfg.head_dim
        self.num_qo = cfg.num_attention_heads
        self.num_kv = cfg.num_key_value_heads
        # Single fused projection: [q; k; v] concatenated on the output dim
        # (weights merged at load time, like gate_up_proj).
        self.qkv_proj = Linear(
            cfg.hidden_size, (self.num_qo + 2 * self.num_kv) * cfg.head_dim
        )
        self.o_proj = Linear(self.num_qo * cfg.head_dim, cfg.hidden_size)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.attn = attn
        self.rotary = rotary
        self.fused_qk_rope_kv = bool(fused_qk_rope_kv)

    def forward(
        self,
        x: torch.Tensor,
        batch: ForwardBatch,
        *,
        q_fp8: Optional[torch.Tensor] = None,
        a_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t = int(x.shape[0]) if q_fp8 is None else int(q_fp8.shape[0])
        orig = (t, self.qkv_proj.in_features)
        if q_fp8 is not None and a_scale is not None and self.qkv_proj._use_fp8:
            out_dtype = (
                x.dtype
                if x.dtype in (torch.bfloat16, torch.float16)
                else torch.bfloat16
            )
            qkv = self.qkv_proj.forward_fp8(q_fp8, a_scale, orig, out_dtype)
        else:
            qkv = self.qkv_proj(x)
        # [t, (n_q + 2*n_kv) * D] -> per-head views into the fused buffer; the
        # rope/store kernel takes explicit strides, so no split copies here.
        heads = qkv.view(t, self.num_qo + 2 * self.num_kv, self.head_dim)
        q = heads[:, : self.num_qo]
        k = heads[:, self.num_qo : self.num_qo + self.num_kv]
        v = heads[:, self.num_qo + self.num_kv :]
        stored = False
        pool = self.attn.kv_pool
        loc = batch.out_cache_loc
        if (
            self.fused_qk_rope_kv
            and loc is not None
            and pool.dtype == torch.float8_e4m3fn
        ):
            stored = fused_qk_norm_rope_store_fp8(
                q,
                k,
                v,
                batch.positions,
                loc,
                self.q_norm.weight,
                self.k_norm.weight,
                self.q_norm.eps,
                self.rotary.cos_sin_cache,
                pool.k[self.layer_id],
                pool.v[self.layer_id],
            )
        if not stored:
            q = self.q_norm(q)
            k = self.k_norm(k)
            q2 = q.reshape(t, -1)
            k2 = k.reshape(t, -1)
            if loc is not None:
                stored = apply_rope_and_store_kv(
                    batch.positions,
                    q2,
                    k2,
                    v.reshape(t, -1),
                    loc,
                    pool.k[self.layer_id].view(-1, pool.row_dim),
                    pool.v[self.layer_id].view(-1, pool.row_dim),
                    self.head_dim,
                    self.rotary.cos_sin_cache,
                )
            if not stored:
                q2, k2 = self.rotary(batch.positions, q2, k2)
            q = q2.view(t, self.num_qo, self.head_dim)
            k = k2.view(t, self.num_kv, self.head_dim)
        # FlashInfer wants a contiguous q; after the fused in-place rope, q is
        # a strided view into the qkv buffer (one small copy, k/v need none).
        o = self.attn.forward(
            q.contiguous(), k, v, self.layer_id, batch, skip_store=stored
        )
        return self.o_proj(o.reshape(t, -1))


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config, fused_silu_fp8: bool = True):
        super().__init__()
        self.gate_up_proj = Linear(cfg.hidden_size, 2 * cfg.intermediate_size)
        self.down_proj = Linear(cfg.intermediate_size, cfg.hidden_size)
        self.fused_silu_fp8 = bool(fused_silu_fp8)

    def forward(
        self,
        x: torch.Tensor,
        *,
        q_fp8: Optional[torch.Tensor] = None,
        a_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q_fp8 is not None and a_scale is not None and self.gate_up_proj._use_fp8:
            out_dtype = (
                x.dtype
                if x.dtype in (torch.bfloat16, torch.float16)
                else torch.bfloat16
            )
            orig = (int(q_fp8.shape[0]), self.gate_up_proj.in_features)
            gu = self.gate_up_proj.forward_fp8(q_fp8, a_scale, orig, out_dtype)
        else:
            gu = self.gate_up_proj(x)
            out_dtype = (
                gu.dtype
                if gu.dtype in (torch.bfloat16, torch.float16)
                else torch.bfloat16
            )
        if self.fused_silu_fp8 and self.down_proj._use_fp8:
            y_fp8, y_scale = silu_and_mul_per_token_quant_fp8(gu)
            rows = int(y_fp8.shape[0])
            return self.down_proj.forward_fp8(
                y_fp8, y_scale, (rows, self.down_proj.in_features), out_dtype
            )
        return self.down_proj(silu_and_mul(gu))


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Qwen3Config,
        layer_id: int,
        attn: AttentionBackend,
        rotary: RotaryEmbedding,
        fused_rms_fp8: bool = True,
        fused_silu_fp8: bool = True,
        fused_qk_rope_kv: bool = True,
    ):
        super().__init__()
        self.self_attn = Qwen3Attention(
            cfg, layer_id, attn, rotary, fused_qk_rope_kv=fused_qk_rope_kv
        )
        self.mlp = Qwen3MLP(cfg, fused_silu_fp8=fused_silu_fp8)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.fused_rms_fp8 = bool(fused_rms_fp8)

    def _use_fused_quant(self) -> bool:
        return bool(self.fused_rms_fp8 and self.self_attn.qkv_proj._use_fp8)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._use_fused_quant():
            if residual is None:
                residual = x
                q_fp8, a_scale = self.input_layernorm.quant_fp8(x)
            else:
                q_fp8, a_scale, residual = self.input_layernorm.quant_fp8(x, residual)
            hidden = self.self_attn(x, batch, q_fp8=q_fp8, a_scale=a_scale)
            q_fp8, a_scale, residual = self.post_attention_layernorm.quant_fp8(
                hidden, residual
            )
            hidden = self.mlp(hidden, q_fp8=q_fp8, a_scale=a_scale)
            return hidden, residual
        if residual is None:
            hidden = self.input_layernorm(x)
            residual = x
        else:
            hidden, residual = self.input_layernorm(x, residual)
        hidden = self.self_attn(hidden, batch)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual


class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        cfg: Qwen3Config,
        attn: AttentionBackend,
        device,
        dtype,
        fused_rms_fp8: bool = True,
        fused_silu_fp8: bool = True,
        fused_qk_rope_kv: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.fused_rms_fp8 = bool(fused_rms_fp8)
        self.fused_silu_fp8 = bool(fused_silu_fp8)
        self.fused_qk_rope_kv = bool(fused_qk_rope_kv)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rotary = RotaryEmbedding(
            cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, device=device
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    cfg,
                    i,
                    attn,
                    self.rotary,
                    fused_rms_fp8=self.fused_rms_fp8,
                    fused_silu_fp8=self.fused_silu_fp8,
                    fused_qk_rope_kv=self.fused_qk_rope_kv,
                )
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head_weight: Optional[torch.Tensor] = None
        self.to(device=device)

    def forward(self, batch: ForwardBatch) -> torch.Tensor:
        x = self.embed_tokens(batch.input_ids)
        residual: Optional[torch.Tensor] = None
        for layer in self.layers:
            x, residual = layer(x, residual, batch)
        if residual is None:
            return self.norm(x)
        y, _ = self.norm(x, residual)
        return y

    def lm_head(self) -> torch.Tensor:
        if self.lm_head_weight is not None:
            return self.lm_head_weight
        return self.embed_tokens.weight
