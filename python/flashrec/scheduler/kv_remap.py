"""Share-on-fork KV remap: identity skip / permute rows / gather decode window."""

from __future__ import annotations

import torch


def classify_parents(parents: torch.Tensor, k: int) -> str:
    """identity / permute / fork."""
    if parents.device.type != "cpu":
        arange = torch.arange(k, device=parents.device, dtype=parents.dtype)
        if torch.equal(parents, arange):
            return "identity"
        if int(parents.min()) < 0 or int(parents.max()) >= k:
            return "fork"
        uniq = torch.unique(parents)
        if int(uniq.numel()) == k:
            return "permute"
        return "fork"
    vals = [int(x) for x in parents.tolist()]
    if any(v < 0 or v >= k for v in vals):
        return "fork"
    if vals == list(range(k)):
        return "identity"
    if len(set(vals)) == k:
        return "permute"
    return "fork"


def remap_by_parents(
    table: torch.Tensor,
    beam_rows: torch.Tensor,
    parents_rel: torch.Tensor,
    prefix_len: int,
    seq_len: int,
    *,
    skip_classify: bool = True,
) -> str:
    """Inplace gather of ``table[rows, prefix:seq]`` by parent. Returns kind.

    GPU ``classify_parents`` uses ``torch.equal`` / ``int(tensor.min())`` and
    host-syncs every decode step. Default is to always gather (an identity
    gather is cheaper than a sync).
    """
    if prefix_len >= seq_len or int(beam_rows.numel()) == 0:
        return "identity"
    n_cols = int(table.shape[1])
    seq_len = min(int(seq_len), n_cols)
    prefix_len = max(int(prefix_len), 0)
    if prefix_len >= seq_len:
        return "identity"
    rows = beam_rows.to(dtype=torch.int64)
    k = int(rows.numel())
    parents = parents_rel.to(dtype=torch.int64, device=rows.device).view(-1)
    if int(parents.numel()) != k:
        return "identity"
    parents = parents.clamp(0, k - 1)
    if not skip_classify and parents.device.type == "cpu":
        kind = classify_parents(parents, k)
        if kind == "identity":
            return kind
    else:
        kind = "gather"
    window = table[rows, prefix_len:seq_len]
    table[rows, prefix_len:seq_len] = window[parents]
    return kind
