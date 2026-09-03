"""RoPE. Prefers sgl-kernel inplace, then FlashInfer, then PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn

from flashrec.kernel.sglops import apply_rope_inplace


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, base: float, device=None):
        super().__init__()
        self.head_dim = int(head_dim)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        t = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer(
            "cos_sin_cache", torch.cat((cos, sin), dim=-1), persistent=False
        )
        self._fi = None
        try:
            from flashinfer import apply_rope_with_cos_sin_cache_inplace

            self._fi = apply_rope_with_cos_sin_cache_inplace
        except Exception:
            self._fi = None

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        if apply_rope_inplace(positions, query, key, self.head_dim, self.cos_sin_cache):
            return query, key
        if self._fi is not None and query.is_cuda:
            self._fi(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_dim,
                cos_sin_cache=self.cos_sin_cache,
            )
            return query, key
        cache = self.cos_sin_cache[positions.long()]
        cos, sin = cache.chunk(2, dim=-1)
        q = query
        k = key
        squeeze = False
        if q.dim() == 2:
            q = q.view(q.shape[0], -1, self.head_dim)
            k = k.view(k.shape[0], -1, self.head_dim)
            squeeze = True
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        if squeeze:
            q = q.reshape(query.shape)
            k = k.reshape(key.shape)
        return q.to(query.dtype), k.to(key.dtype)
