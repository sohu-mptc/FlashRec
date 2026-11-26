"""Trailing unary finish: skip the last decode when every remaining path is unique.

Used for RecIF 4-level catalogs whose last codebook is a forced ``sid_end``.
Forced tokens contribute 0 to ``cum_logprobs`` (no extra ranking signal).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from flashrec.core import FinishReason
from flashrec.search.state import BeamSearchList
from flashrec.search.trie import BeamValidPathTrie

__all__ = [
    "apply_unary_finish",
    "can_unary_finish",
    "try_unary_finish_request",
]


def can_unary_finish(
    node_ids: torch.Tensor,
    outdeg: torch.Tensor,
    unary_child: torch.Tensor,
    remain: int,
) -> bool:
    """True when every live node has out-degree 1 for ``remain`` steps."""
    remain = int(remain)
    if remain <= 0 or node_ids is None or outdeg is None or unary_child is None:
        return False
    if int(node_ids.numel()) == 0 or int(outdeg.numel()) == 0:
        return False
    n_nodes = int(outdeg.shape[0])
    nodes = node_ids.reshape(-1).to(dtype=torch.int64)
    child = unary_child.to(device=nodes.device, dtype=torch.int64)
    deg = outdeg.to(device=nodes.device, dtype=torch.int32)
    for _ in range(remain):
        valid = (nodes >= 0) & (nodes < n_nodes)
        if not bool(valid.all().item()):
            return False
        if not bool((deg.gather(0, nodes) == 1).all().item()):
            return False
        nodes = child.gather(0, nodes)
    return True


def apply_unary_finish(
    beam_list: BeamSearchList,
    remain: int,
    unary_token: torch.Tensor,
    unary_child: torch.Tensor,
) -> None:
    """Append ``remain`` forced tokens. Parents are identity; scores unchanged."""
    remain = int(remain)
    if remain <= 0 or beam_list.token_ids is None or beam_list.node_ids is None:
        return
    device = beam_list.token_ids.device
    nodes = beam_list.node_ids.to(device=device, dtype=torch.int64).reshape(-1)
    tok_tbl = unary_token.to(device=device, dtype=torch.int64)
    child_tbl = unary_child.to(device=device, dtype=torch.int64)
    n = int(nodes.numel())
    rows = int(beam_list.token_ids.shape[0])
    if n != rows:
        n = min(n, rows)
        nodes = nodes[:n]
        beam_list.token_ids = beam_list.token_ids[:n]
    need = int(beam_list.cur_len) + remain
    if int(beam_list.token_ids.shape[1]) < need:
        pad = need - int(beam_list.token_ids.shape[1])
        beam_list.token_ids = F.pad(beam_list.token_ids, (0, pad))
    nmax = int(tok_tbl.shape[0]) - 1
    last = None
    for _ in range(remain):
        safe = nodes.clamp(0, nmax)
        last = tok_tbl.gather(0, safe)
        col = int(beam_list.cur_len)
        beam_list.token_ids[:n, col] = last
        beam_list.cur_len = col + 1
        nodes = child_tbl.gather(0, safe)
    beam_list.last_tokens = last
    beam_list.node_ids = nodes
    beam_list.dense_authoritative = True


def try_unary_finish_request(
    req,
    valid_path: Optional[BeamValidPathTrie],
    limit: int,
    gate: Optional[bool] = None,
) -> bool:
    """Finish ``req`` if the remaining SID suffix is unary. Mutates the request.

    ``gate`` short-circuits the device-side unary check when the caller has
    already evaluated it (the scheduler batches the gate for a whole tick into
    one sync); ``None`` keeps the standalone behavior.
    """
    if req is None or bool(getattr(req, "finished", False)):
        return False
    if float(getattr(req, "temperature", 0.0) or 0.0) > 0.0:
        return False
    if valid_path is None or not valid_path.active or valid_path.mode != "trie":
        return False
    bl = getattr(req, "beam_list", None)
    if bl is None or bl.node_ids is None or bl.token_ids is None:
        return False
    remain = int(limit) - int(bl.generated_len())
    if remain <= 0:
        return False
    device = bl.node_ids.device
    if not valid_path.ensure_unary_tables(device):
        return False
    if gate is None:
        sfx = getattr(valid_path, "_unary_suffix_len", None)
        if sfx is not None:
            # One sync: every live node valid and forced for >= remain steps.
            nodes = bl.node_ids.reshape(-1).to(dtype=torch.int64)
            n_nodes = int(sfx.shape[0])
            if int(nodes.numel()) == 0:
                return False
            valid = (nodes >= 0) & (nodes < n_nodes)
            safe = nodes.clamp(0, max(n_nodes - 1, 0))
            runs = sfx.to(device=nodes.device)[safe]
            ok = valid.all() & (runs >= remain).all()
            if not bool(ok.item()):
                return False
        elif not can_unary_finish(
            bl.node_ids, valid_path._outdeg, valid_path._unary_child, remain
        ):
            return False
    elif not gate:
        return False
    apply_unary_finish(bl, remain, valid_path._unary_token, valid_path._unary_child)
    req.finished = True
    req.finish_reason = FinishReason(
        type="length", length=int(getattr(req, "max_new_tokens", limit) or limit)
    )
    return True
