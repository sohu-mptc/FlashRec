"""Load HF safetensors into Qwen3ForCausalLM."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open

from flashrec.layers.linear import Linear
from flashrec.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

logger = logging.getLogger(__name__)


def merge_gate_up_weights(
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    gate_scale: Optional[torch.Tensor] = None,
    up_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Concat ``[gate; up]`` on the output dim. Do not requantize."""
    if tuple(gate_w.shape) != tuple(up_w.shape):
        raise ValueError(
            f"gate/up shape mismatch: {tuple(gate_w.shape)} vs {tuple(up_w.shape)}"
        )
    weight = torch.cat([gate_w, up_w], dim=0)
    scale = None
    if gate_scale is not None and up_scale is not None:
        scale = torch.cat([gate_scale.reshape(-1), up_scale.reshape(-1)], dim=0)
    return weight, scale


def merge_qkv_weights(
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Concat ``[q; k; v]`` on the output dim. Do not requantize."""
    if q_w.shape[1] != k_w.shape[1] or q_w.shape[1] != v_w.shape[1]:
        raise ValueError(
            f"qkv in-dim mismatch: {tuple(q_w.shape)} / {tuple(k_w.shape)} / {tuple(v_w.shape)}"
        )
    weight = torch.cat([q_w, k_w, v_w], dim=0)
    scale = None
    if q_scale is not None and k_scale is not None and v_scale is not None:
        scale = torch.cat(
            [q_scale.reshape(-1), k_scale.reshape(-1), v_scale.reshape(-1)], dim=0
        )
    return weight, scale


def load_hf_config(model_path: str) -> Qwen3Config:
    cfg = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    return Qwen3Config(
        hidden_size=int(cfg["hidden_size"]),
        intermediate_size=int(cfg["intermediate_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=int(cfg["num_attention_heads"]),
        num_key_value_heads=int(cfg["num_key_value_heads"]),
        head_dim=int(
            cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        ),
        vocab_size=int(cfg["vocab_size"]),
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        rope_theta=float(cfg.get("rope_theta", 1000000)),
        max_position_embeddings=int(cfg.get("max_position_embeddings", 40960)),
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),
    )


def _iter_safetensors(model_path: str):
    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    files = []
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        single = root / "model.safetensors"
        if single.is_file():
            files = [single.name]
    for name in files:
        path = root / name
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def load_weights(
    model: Qwen3ForCausalLM,
    model_path: str,
    device: torch.device,
    compute_dtype: torch.dtype,
    quantize_fp8: bool = False,
) -> None:
    linears: Dict[str, Linear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, Linear):
            linears[name] = mod
    norms = {}
    for name, mod in model.named_modules():
        if (
            name.endswith("layernorm")
            or name.endswith("q_norm")
            or name.endswith("k_norm")
            or name == "norm"
        ):
            norms[name] = mod

    scale_map: Dict[str, torch.Tensor] = {}
    pending_gate_up: Dict[str, Dict[str, torch.Tensor]] = {}
    pending_qkv: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in _iter_safetensors(model_path):
        if key.endswith(".weight_scale") or key.endswith(".weight_scale_inv"):
            rest = (
                key[len("model.layers.") :] if key.startswith("model.layers.") else ""
            )
            if rest:
                layer_id_s, _, tail = rest.partition(".")
                mod_name = f"layers.{layer_id_s}." + tail.rsplit(".", 1)[0]
                scale_map[mod_name] = tensor.to(device=device)

    for key, tensor in _iter_safetensors(model_path):
        if key.endswith(".weight_scale") or key.endswith(".weight_scale_inv"):
            continue
        t = tensor
        if key == "model.embed_tokens.weight":
            model.embed_tokens.weight = torch.nn.Parameter(
                t.to(device=device, dtype=compute_dtype), requires_grad=False
            )
            continue
        if key == "lm_head.weight":
            model.lm_head_weight = t.to(device=device, dtype=compute_dtype)
            continue
        if key == "model.norm.weight":
            model.norm.weight = torch.nn.Parameter(
                t.to(device=device, dtype=compute_dtype), requires_grad=False
            )
            continue
        if not key.startswith("model.layers."):
            continue
        rest = key[len("model.layers.") :]
        layer_id_s, _, tail = rest.partition(".")
        prefix = f"layers.{layer_id_s}."
        # tail like self_attn.q_proj.weight
        if tail.endswith(".weight"):
            mod_name = prefix + tail[: -len(".weight")]
            if mod_name.endswith(".mlp.gate_proj") or mod_name.endswith(".mlp.up_proj"):
                mlp_prefix, _, which = mod_name.rpartition(".")
                bucket = pending_gate_up.setdefault(mlp_prefix + ".", {})
                bucket[which] = t.to(device=device)
                continue
            if (
                mod_name.endswith(".self_attn.q_proj")
                or mod_name.endswith(".self_attn.k_proj")
                or mod_name.endswith(".self_attn.v_proj")
            ):
                attn_prefix, _, which = mod_name.rpartition(".")
                bucket = pending_qkv.setdefault(attn_prefix + ".", {})
                bucket[which] = t.to(device=device)
                continue
            if mod_name in linears:
                w = t.to(device=device)
                if w.dtype not in (torch.float8_e4m3fn,) and w.dtype != compute_dtype:
                    w = w.to(dtype=compute_dtype)
                linears[mod_name].load(
                    w,
                    quantize_fp8=quantize_fp8 and w.dtype != torch.float8_e4m3fn,
                    weight_scale=scale_map.get(mod_name),
                )
                continue
            if mod_name in norms:
                norms[mod_name].weight = torch.nn.Parameter(
                    t.to(device=device, dtype=compute_dtype), requires_grad=False
                )
                continue
            # q_norm / k_norm
            if "q_norm" in tail or "k_norm" in tail:
                full = prefix + tail[: -len(".weight")]
                # layers.0.self_attn.q_norm
                parts = full.split(".")
                # walk
                obj = model
                for p in parts:
                    obj = getattr(obj, p)
                obj.weight = torch.nn.Parameter(
                    t.to(device=device, dtype=compute_dtype), requires_grad=False
                )

    for attn_prefix, parts in pending_qkv.items():
        dest = attn_prefix + "qkv_proj"
        if dest not in linears:
            logger.warning("no qkv_proj module for %s", attn_prefix)
            continue
        if any(p not in parts for p in ("q_proj", "k_proj", "v_proj")):
            logger.warning("incomplete q/k/v weights for %s", attn_prefix)
            continue
        merged, scale = merge_qkv_weights(
            parts["q_proj"],
            parts["k_proj"],
            parts["v_proj"],
            scale_map.get(attn_prefix + "q_proj"),
            scale_map.get(attn_prefix + "k_proj"),
            scale_map.get(attn_prefix + "v_proj"),
        )
        w = merged
        if w.dtype not in (torch.float8_e4m3fn,) and w.dtype != compute_dtype:
            w = w.to(dtype=compute_dtype)
        linears[dest].load(
            w,
            quantize_fp8=quantize_fp8 and w.dtype != torch.float8_e4m3fn,
            weight_scale=scale,
        )

    for mlp_prefix, parts in pending_gate_up.items():
        dest = mlp_prefix + "gate_up_proj"
        if dest not in linears:
            logger.warning("no gate_up_proj module for %s", mlp_prefix)
            continue
        if "gate_proj" not in parts or "up_proj" not in parts:
            logger.warning("incomplete gate/up weights for %s", mlp_prefix)
            continue
        merged, scale = merge_gate_up_weights(
            parts["gate_proj"],
            parts["up_proj"],
            scale_map.get(mlp_prefix + "gate_proj"),
            scale_map.get(mlp_prefix + "up_proj"),
        )
        w = merged
        if w.dtype not in (torch.float8_e4m3fn,) and w.dtype != compute_dtype:
            w = w.to(dtype=compute_dtype)
        linears[dest].load(
            w,
            quantize_fp8=quantize_fp8 and w.dtype != torch.float8_e4m3fn,
            weight_scale=scale,
        )

    if model.lm_head_weight is None and model.cfg.tie_word_embeddings:
        model.lm_head_weight = model.embed_tokens.weight
