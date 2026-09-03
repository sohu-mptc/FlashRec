# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Valid-path (trie / codebook) constraints for GenRec-style beam search.

Supports three modes, selected automatically by configuration:

1. **Trie file** (``--sid-vocab-file``): each line / JSON list is a valid
   SID token sequence. Expansion only keeps next tokens that continue a valid
   prefix (xGR-style valid path constraint).

2. **Position codebooks** (``--sid-codebook-sizes``): partition
   ``sid_token_range`` into per-step vocabularies. Step ``t`` may
   only emit tokens from codebook ``t``.

3. **Flat allowlist**: when only ``sid_token_range`` is set, every
   step allows that full set (cheap safety filter; usually a no-op if lm_head
   already restricts logits).

Hot path: ``mask_candidates`` keeps candidate tokens on GPU. Trie mode uses a
cached ``allow_table[node, token-base]`` bool matrix; only the short prefix
columns are synced once to resolve ``prefix -> node_id``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch

logger = logging.getLogger(__name__)

# Sentinel node ids for GPU allow_table rows / unconstrained beams.
_UNCONSTRAINED_NODE = -1

# Dense allow/next tables cost (nodes+1)*vocab cells each. Beyond this budget
# (e.g. public 8192^3 catalogs with ~1M internal nodes -> ~25G cells) switch to
# a sorted-key CSR layout: membership/transition via searchsorted over edges.
_DENSE_MAX_CELLS = int(
    os.environ.get("FLASHREC_TRIE_DENSE_MAX_CELLS", str(1 << 29))
)


@dataclass
class BeamValidPathTrie:
    """Prefix trie over valid SID sequences + optional flat/codebook allowlists."""

    # prefix_tuple -> allowed next token ids
    children: Dict[Tuple[int, ...], Set[int]] = field(default_factory=dict)
    # step -> allowed token ids (codebook / flat). None = unconstrained at that step.
    step_allow: Optional[List[Optional[Set[int]]]] = None
    # Union of all allowed tokens (for quick flat checks).
    all_tokens: Optional[Set[int]] = None
    max_depth: int = 0
    mode: str = "none"  # none | trie | codebook | flat
    token_base: Optional[int] = None
    vocab_size: Optional[int] = None

    # --- CPU index (built lazily / at finalize) ---
    _prefix_to_node: Dict[Tuple[int, ...], int] = field(
        default_factory=dict, repr=False
    )
    _invalid_node: int = field(default=-1, repr=False)
    _index_ready: bool = field(default=False, repr=False)

    # --- GPU caches (per device) ---
    _cache_device: Optional[torch.device] = field(default=None, repr=False)
    _allow_table: Optional[torch.Tensor] = field(
        default=None, repr=False
    )  # [N+1, V] bool
    _next_node: Optional[torch.Tensor] = field(
        default=None, repr=False
    )  # [N+1, V] int64
    # CSR fallback for large tries (dense tables would not fit on GPU):
    # sorted keys node*V+rel and the matched child ids (or sentinels).
    _csr_keys: Optional[torch.Tensor] = field(default=None, repr=False)  # [E] int64
    _csr_child: Optional[torch.Tensor] = field(default=None, repr=False)  # [E] int64
    _step_allow_ids: Optional[List[Optional[torch.Tensor]]] = field(
        default=None, repr=False
    )  # list of 1D int64 allowlists
    _flat_allow_ids: Optional[torch.Tensor] = field(default=None, repr=False)
    # Step-depth cache: cur_len -> allow_table slice reuse hint (codebook/flat no-op).
    _mask_step_cache_len: int = field(default=-1, repr=False)
    _mask_step_cache_ids: Optional[torch.Tensor] = field(default=None, repr=False)

    @property
    def active(self) -> bool:
        return self.mode != "none"

    def allowed_next(
        self, prefix: Sequence[int], step: Optional[int] = None
    ) -> Optional[Set[int]]:
        """Return allowed next tokens for ``prefix``.

        ``None`` means unconstrained (do not mask).
        Empty set means no valid continuation.
        """
        if not self.active:
            return None

        depth = len(prefix) if step is None else int(step)

        if self.mode == "flat":
            return self.all_tokens

        if self.mode == "trie":
            key = tuple(int(x) for x in prefix)
            allowed = self.children.get(key)
            if allowed is not None:
                return allowed
            if len(key) >= self.max_depth:
                return None
            return set()

        if self.step_allow is not None:
            if depth < 0:
                return set()
            if depth >= len(self.step_allow):
                return self.all_tokens if self.all_tokens is not None else set()
            step_set = self.step_allow[depth]
            if step_set is None:
                return self.all_tokens
            return step_set

        return self.all_tokens

    def finalize_index(
        self, token_base: Optional[int] = None, vocab_size: Optional[int] = None
    ) -> None:
        """Build CPU prefix→node map and infer token_base / vocab_size."""
        if self.mode == "none":
            self._index_ready = True
            return

        if token_base is not None:
            self.token_base = int(token_base)
        if vocab_size is not None:
            self.vocab_size = int(vocab_size)

        if self.token_base is None or self.vocab_size is None:
            if self.all_tokens:
                lo = min(self.all_tokens)
                hi = max(self.all_tokens)
                if self.token_base is None:
                    self.token_base = int(lo)
                if self.vocab_size is None:
                    self.vocab_size = int(hi - self.token_base + 1)
            else:
                self.token_base = int(self.token_base or 0)
                self.vocab_size = int(self.vocab_size or 1)

        if self.mode == "trie":
            # Stable node ids: root first, then remaining prefixes.
            prefixes = list(self.children.keys())
            if () not in self.children:
                # Ensure root exists even if empty children (shouldn't happen).
                self.children[()] = set()
                prefixes = list(self.children.keys())
            prefixes.sort(key=lambda p: (len(p), p))
            # Put root at 0.
            if () in prefixes:
                prefixes.remove(())
                prefixes.insert(0, ())
            self._prefix_to_node = {p: i for i, p in enumerate(prefixes)}
            self._invalid_node = len(prefixes)  # extra all-False row
        self._index_ready = True

    def _ensure_index(self) -> None:
        if not self._index_ready:
            self.finalize_index()

    def _ensure_gpu_cache(self, device: torch.device) -> None:
        self._ensure_index()
        if self._cache_device == device and (
            (
                self.mode == "trie"
                and (self._allow_table is not None or self._csr_keys is not None)
            )
            or (self.mode in ("flat", "codebook") and self._flat_allow_ids is not None)
        ):
            return

        self._cache_device = device
        base = int(self.token_base or 0)
        vsz = max(int(self.vocab_size or 1), 1)

        if self.mode == "trie":
            n_nodes = len(self._prefix_to_node)
            if (n_nodes + 1) * vsz > _DENSE_MAX_CELLS:
                self._build_csr_cache(device, base, vsz, n_nodes)
                return
            # Build on CPU then one H2D — per-cell GPU writes are extremely slow.
            table_cpu = torch.zeros((n_nodes + 1, vsz), dtype=torch.bool)
            # next_node[node, rel] = child node id, or invalid / unconstrained sentinel.
            next_cpu = torch.full(
                (n_nodes + 1, vsz), self._invalid_node, dtype=torch.int32
            )
            p2n = self._prefix_to_node
            for prefix, node_id in p2n.items():
                for tok in self.children.get(prefix, ()):
                    rel = int(tok) - base
                    if 0 <= rel < vsz:
                        table_cpu[node_id, rel] = True
                        child_key = prefix + (int(tok),)
                        child_id = p2n.get(child_key)
                        if child_id is not None:
                            next_cpu[node_id, rel] = int(child_id)
                        elif len(child_key) >= self.max_depth:
                            next_cpu[node_id, rel] = _UNCONSTRAINED_NODE
            self._allow_table = table_cpu.to(device, non_blocking=False)
            self._next_node = next_cpu.to(device, non_blocking=False)
            self._csr_keys = None
            self._csr_child = None
            self._step_allow_ids = None
            self._flat_allow_ids = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            logger.info(
                "Beam valid-path GPU allow_table ready: nodes=%d vocab=%d device=%s "
                "(~%.1f MB)",
                n_nodes,
                vsz,
                device,
                (table_cpu.numel() + next_cpu.numel() * 4) / (1024 * 1024),
            )
            return

        # flat / codebook: cache 1D allow id tensors (for torch.isin).
        if self.all_tokens:
            self._flat_allow_ids = torch.tensor(
                sorted(self.all_tokens), dtype=torch.int64, device=device
            )
        else:
            self._flat_allow_ids = torch.empty(0, dtype=torch.int64, device=device)

        step_tensors: List[Optional[torch.Tensor]] = []
        if self.step_allow is not None:
            for s in self.step_allow:
                if s is None:
                    step_tensors.append(None)
                elif not s:
                    step_tensors.append(
                        torch.empty(0, dtype=torch.int64, device=device)
                    )
                else:
                    step_tensors.append(
                        torch.tensor(sorted(s), dtype=torch.int64, device=device)
                    )
        self._step_allow_ids = step_tensors
        self._allow_table = None

    def _build_csr_cache(
        self, device: torch.device, base: int, vsz: int, n_nodes: int
    ) -> None:
        """Sorted-key edge layout for tries whose dense tables would not fit.

        Key = node_id * vsz + rel_token; membership and child transition are a
        single ``searchsorted`` + gather, fully on GPU with no host sync.
        """
        p2n = self._prefix_to_node
        invalid = int(self._invalid_node)
        max_depth = int(self.max_depth)
        keys: List[int] = []
        child_ids: List[int] = []
        for prefix, node_id in p2n.items():
            toks = self.children.get(prefix)
            if not toks:
                continue
            row = node_id * vsz
            for tok in toks:
                rel = int(tok) - base
                if not (0 <= rel < vsz):
                    continue
                keys.append(row + rel)
                child_key = prefix + (int(tok),)
                child_id = p2n.get(child_key)
                if child_id is not None:
                    child_ids.append(int(child_id))
                elif len(child_key) >= max_depth:
                    child_ids.append(_UNCONSTRAINED_NODE)
                else:
                    child_ids.append(invalid)
        keys_t = torch.tensor(keys, dtype=torch.int64)
        child_t = torch.tensor(child_ids, dtype=torch.int64)
        order = torch.argsort(keys_t)
        self._csr_keys = keys_t[order].to(device, non_blocking=False)
        self._csr_child = child_t[order].to(device, non_blocking=False)
        self._allow_table = None
        self._next_node = None
        self._step_allow_ids = None
        self._flat_allow_ids = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        logger.info(
            "Beam valid-path GPU CSR ready: nodes=%d vocab=%d edges=%d device=%s "
            "(~%.1f MB; dense would need ~%.1f GB)",
            n_nodes,
            vsz,
            keys_t.numel(),
            device,
            keys_t.numel() * 16 / (1024 * 1024),
            (n_nodes + 1) * vsz * 5 / (1024**3),
        )

    def _csr_lookup(
        self, nodes: torch.Tensor, rel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Look up edges (node, rel) in the CSR cache.

        Both inputs must be int64, same shape, with nodes >= 0 and rel within
        [0, vsz). Returns (found_mask, child_ids) — child is the invalid
        sentinel where not found.
        """
        keys = nodes * max(int(self.vocab_size or 1), 1) + rel
        csr_keys = self._csr_keys
        e = int(csr_keys.numel())
        if e == 0:
            found = torch.zeros_like(keys, dtype=torch.bool)
            return found, torch.full_like(keys, int(self._invalid_node))
        idx = torch.searchsorted(csr_keys, keys)
        idx_c = idx.clamp(max=e - 1)
        found = (idx < e) & (csr_keys[idx_c] == keys)
        child = torch.where(
            found,
            self._csr_child[idx_c],
            torch.full_like(keys, int(self._invalid_node)),
        )
        return found, child

    def _resolve_node_ids(
        self,
        prefixes: Optional[torch.Tensor],
        bw: int,
        cur_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Map beam prefixes to node ids entirely on GPU via ``_next_node`` walk.

        Falls back to a tiny D2H path only when the transition table is missing.
        """
        if cur_len <= 0 or prefixes is None or not isinstance(prefixes, torch.Tensor):
            return torch.zeros(bw, dtype=torch.int64, device=device)

        cols = min(int(cur_len), int(prefixes.shape[1]))
        if cols <= 0:
            return torch.zeros(bw, dtype=torch.int64, device=device)

        if self._next_node is not None:
            base = int(self.token_base or 0)
            vsz = int(self._next_node.shape[1])
            invalid = int(self._invalid_node)
            nodes = torch.zeros(bw, dtype=torch.int64, device=device)
            pref = prefixes[:bw, :cols].to(dtype=torch.int64)
            for t in range(cols):
                rel = pref[:, t] - base
                in_range = (rel >= 0) & (rel < vsz)
                rel_c = rel.clamp(0, max(vsz - 1, 0))
                # Keep unconstrained (-1); walk only valid non-negative nodes.
                unconstrained = nodes < 0
                n_nodes = int(self._next_node.shape[0])
                safe = nodes.clamp(min=0, max=max(n_nodes - 1, 0))
                nxt = self._next_node[safe, rel_c]
                # OOR tokens or already-invalid → invalid node (scalar broadcast).
                nxt = torch.where(in_range, nxt, invalid)
                nodes = torch.where(unconstrained, nodes, nxt)
            return nodes

        if self._csr_keys is not None:
            base = int(self.token_base or 0)
            vsz = max(int(self.vocab_size or 1), 1)
            invalid = int(self._invalid_node)
            nodes = torch.zeros(bw, dtype=torch.int64, device=device)
            pref = prefixes[:bw, :cols].to(dtype=torch.int64)
            for t in range(cols):
                rel = pref[:, t] - base
                in_range = (rel >= 0) & (rel < vsz)
                unconstrained = nodes < 0
                found, child = self._csr_lookup(
                    nodes.clamp(min=0), rel.clamp(0, max(vsz - 1, 0))
                )
                nxt = torch.where(
                    found & in_range, child, torch.full_like(nodes, invalid)
                )
                nodes = torch.where(unconstrained, nodes, nxt)
            return nodes

        # Legacy fallback: short GenRec prefixes (usually ≤2 ints/beam).
        pref_cpu = prefixes[:bw, :cols].to(dtype=torch.int64).cpu()
        rows = pref_cpu.tolist()
        p2n = self._prefix_to_node
        invalid = self._invalid_node
        max_depth = self.max_depth
        out = [0] * bw
        for i, row in enumerate(rows):
            key = tuple(int(x) for x in row)
            node = p2n.get(key)
            if node is not None:
                out[i] = node
            elif len(key) >= max_depth:
                out[i] = _UNCONSTRAINED_NODE
            else:
                out[i] = invalid
        return torch.tensor(out, dtype=torch.int64, device=device)

    def mask_candidates(
        self,
        prefixes: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_scores: torch.Tensor,
        cur_len: int,
        node_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mask invalid candidates to ``-inf`` on GPU; returns the scores tensor.

        When ``node_ids`` is provided (incremental trie state), skips the prefix
        walk entirely — the hot path after the first decode step.
        """
        if not self.active:
            return candidate_scores

        device = candidate_tokens.device
        neg_inf = float("-inf")
        self._ensure_gpu_cache(device)

        if self.mode in ("flat", "codebook"):
            return self._mask_step_allowlist(
                candidate_tokens, candidate_scores, cur_len, neg_inf
            )

        if self.mode != "trie" or (
            self._allow_table is None and self._csr_keys is None
        ):
            return candidate_scores

        bw = candidate_tokens.shape[0]
        if (
            isinstance(node_ids, torch.Tensor)
            and node_ids.numel() >= bw
            and node_ids.device == device
        ):
            used_nodes = node_ids[:bw]
        else:
            used_nodes = self._resolve_node_ids(prefixes, bw, cur_len, device)

        base = int(self.token_base or 0)
        if self._allow_table is None:
            # CSR membership: one searchsorted over the sorted edge keys.
            vsz = max(int(self.vocab_size or 1), 1)
            rel = candidate_tokens.to(dtype=torch.int64) - base
            in_range = (rel >= 0) & (rel < vsz)
            nodes_mat = used_nodes.to(dtype=torch.int64).unsqueeze(1).expand_as(rel)
            unconstrained = nodes_mat < 0
            found, _ = self._csr_lookup(nodes_mat.clamp(min=0), rel.clamp(0, vsz - 1))
            ok = (found & in_range) | unconstrained
            return candidate_scores.masked_fill(~ok, neg_inf)

        # Fused CUDA path: one kernel instead of sub/clamp/gather/masked_fill.
        if (
            device.type == "cuda"
            and candidate_scores.dtype == torch.float32
            and candidate_tokens.dtype == torch.int64
            and used_nodes.dtype == torch.int64
        ):
            try:
                from flashrec.kernel.beam_trie import trie_mask_candidates

                scores_in = candidate_scores
                if not scores_in.is_contiguous():
                    scores_in = scores_in.contiguous()
                return trie_mask_candidates(
                    scores_in,
                    candidate_tokens.contiguous(),
                    used_nodes.contiguous(),
                    self._allow_table,
                    base,
                    neg_inf=neg_inf,
                )
            except Exception:
                pass

        vsz = int(self._allow_table.shape[1])
        rel = candidate_tokens.to(dtype=torch.int64) - base
        in_range = (rel >= 0) & (rel < vsz)
        rel_clamped = rel.clamp(0, vsz - 1)
        unconstrained = used_nodes < 0
        n_nodes = int(self._allow_table.shape[0])
        safe_nodes = used_nodes.clamp(min=0, max=max(n_nodes - 1, 0))
        ok = self._allow_table[safe_nodes.unsqueeze(1), rel_clamped] & in_range
        ok = ok | unconstrained.unsqueeze(1)
        return candidate_scores.masked_fill(~ok, neg_inf)

    def advance_node_ids(
        self,
        node_ids: torch.Tensor,
        parents: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Gather parent trie nodes and transition on ``tokens`` (one GPU step)."""
        device = node_ids.device
        self._ensure_gpu_cache(device)
        if self._next_node is None:
            if self._csr_keys is None:
                return node_ids
            base = int(self.token_base or 0)
            vsz = max(int(self.vocab_size or 1), 1)
            invalid = int(self._invalid_node)
            parents = parents.to(device=device, dtype=torch.int64).view(-1)
            toks = tokens.to(device=device, dtype=torch.int64).view(-1)
            max_p = max(int(node_ids.shape[0]) - 1, 0)
            old = node_ids[parents.clamp(min=0, max=max_p)]
            rel = toks - base
            in_range = (rel >= 0) & (rel < vsz)
            unconstrained = old < 0
            found, child = self._csr_lookup(
                old.clamp(min=0), rel.clamp(0, max(vsz - 1, 0))
            )
            nxt = torch.where(found & in_range, child, torch.full_like(old, invalid))
            return torch.where(unconstrained, old, nxt)
        base = int(self.token_base or 0)
        invalid = int(self._invalid_node)
        if device.type == "cuda" and node_ids.dtype == torch.int64:
            try:
                from flashrec.kernel.beam_trie import trie_advance_nodes

                return trie_advance_nodes(
                    node_ids,
                    parents,
                    tokens,
                    self._next_node,
                    base,
                    invalid,
                )
            except Exception:
                pass
        vsz = int(self._next_node.shape[1])
        parents = parents.to(device=device, dtype=torch.int64).view(-1)
        toks = tokens.to(device=device, dtype=torch.int64).view(-1)
        max_p = max(int(node_ids.shape[0]) - 1, 0)
        old = node_ids[parents.clamp(min=0, max=max_p)]
        rel = toks - base
        in_range = (rel >= 0) & (rel < vsz)
        rel_c = rel.clamp(0, max(vsz - 1, 0))
        unconstrained = old < 0
        n_nodes = int(self._next_node.shape[0])
        safe = old.clamp(min=0, max=max(n_nodes - 1, 0))
        nxt = self._next_node[safe, rel_c]
        nxt = torch.where(in_range, nxt, invalid)
        return torch.where(unconstrained, old, nxt)

    def bootstrap_node_ids(
        self,
        prefixes: Optional[torch.Tensor],
        bw: int,
        cur_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Resolve node ids once (e.g. after prefill) for incremental masking."""
        self._ensure_gpu_cache(device)
        return self._resolve_node_ids(prefixes, bw, cur_len, device)

    def _mask_step_allowlist(
        self,
        candidate_tokens: torch.Tensor,
        candidate_scores: torch.Tensor,
        cur_len: int,
        neg_inf: float,
    ) -> torch.Tensor:
        """Flat / codebook path with cached allow id tensors.

        Reuses the allow-id tensor for the same ``cur_len`` within a decode
        step across requests (common when a wave shares SID depth).
        """
        allow_t: Optional[torch.Tensor]
        if self.mode == "flat":
            allow_t = self._flat_allow_ids
        else:
            # codebook — cache the resolved tensor for this depth
            if (
                self._mask_step_cache_len == cur_len
                and self._mask_step_cache_ids is not None
            ):
                allow_t = self._mask_step_cache_ids
            elif self._step_allow_ids is None or cur_len >= len(self._step_allow_ids):
                allow_t = self._flat_allow_ids
                self._mask_step_cache_len = cur_len
                self._mask_step_cache_ids = allow_t
            else:
                allow_t = self._step_allow_ids[cur_len]
                if allow_t is None:
                    allow_t = self._flat_allow_ids
                self._mask_step_cache_len = cur_len
                self._mask_step_cache_ids = allow_t

        if allow_t is None:
            return candidate_scores
        if allow_t.numel() == 0:
            return candidate_scores.masked_fill(
                torch.ones_like(candidate_scores, dtype=torch.bool), neg_inf
            )
        ok = torch.isin(candidate_tokens, allow_t)
        return candidate_scores.masked_fill(~ok, neg_inf)


def _parse_sid_line(line: str) -> Optional[List[int]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # Support "1 2 3", "1,2,3", "1\t2\t3"
    for sep in [",", "\t", " "]:
        if sep in line:
            parts = [p for p in line.replace("\t", " ").replace(",", " ").split() if p]
            break
    else:
        parts = [line]
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def load_sequences_from_file(path: Union[str, Path]) -> List[List[int]]:
    """Load valid SID sequences from a text or JSON file.

    Supported JSON shapes:
    - list of int lists: ``[[1,2,3], ...]``
    - sid2vid map: ``{"0,1,2": [{"pid": ...}], ...}`` — returns codebook
      index sequences (not yet remapped to lm_head token ids). Prefer
      ``build_trie_from_sid2vid_file`` for GenRec serving.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"beam valid SID file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".json" or text[:1] in "[{":
        data = json.loads(text)
        if isinstance(data, dict):
            # sid2vid: keys are "c0,c1,c2" codebook indices.
            seqs = []
            for key in data:
                parts = [p.strip() for p in str(key).split(",") if p.strip()]
                if not parts:
                    continue
                seqs.append([int(x) for x in parts])
            return seqs
        if not isinstance(data, list):
            raise ValueError(
                f"{path}: JSON root must be a list of sequences or a sid2vid dict"
            )
        seqs = []
        for item in data:
            if not isinstance(item, (list, tuple)):
                raise ValueError(f"{path}: each JSON item must be a list of ints")
            seqs.append([int(x) for x in item])
        return seqs

    seqs = []
    for line in text.splitlines():
        parsed = _parse_sid_line(line)
        if parsed:
            seqs.append(parsed)
    return seqs


def _codes_to_token_ids(
    codes: Sequence[int],
    token_base: int,
    codebook_sizes: Sequence[int],
) -> List[int]:
    if len(codes) > len(codebook_sizes):
        raise ValueError(
            f"SID length {len(codes)} exceeds codebook steps {len(codebook_sizes)}"
        )
    out: List[int] = []
    offset = 0
    for i, code in enumerate(codes):
        size = int(codebook_sizes[i])
        c = int(code)
        if c < 0 or c >= size:
            raise ValueError(f"SID code {c} out of range for codebook[{i}] size {size}")
        out.append(int(token_base) + offset + c)
        offset += size
    return out


def is_sid2vid_json(path: Union[str, Path]) -> bool:
    path = Path(path)
    if path.suffix.lower() != ".json":
        return False
    # Peek without full parse when possible.
    with path.open("r", encoding="utf-8") as f:
        # skip whitespace
        while True:
            ch = f.read(1)
            if not ch:
                return False
            if not ch.isspace():
                return ch == "{"
    return False


def _finalize_trie(
    trie: BeamValidPathTrie,
    *,
    token_base: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> BeamValidPathTrie:
    trie.finalize_index(token_base=token_base, vocab_size=vocab_size)
    return trie


def build_trie_from_sid2vid_file(
    path: Union[str, Path],
    *,
    token_base: int,
    codebook_sizes: Optional[Sequence[int]] = None,
) -> BeamValidPathTrie:
    """Build trie from GenRec ``sid2vid.json`` (keys like ``\"12,34,56\"``).

    Codebook indices are mapped to lm_head token ids::

        token[i] = token_base + sum(codebook_sizes[:i]) + code[i]
    """
    path = Path(path)
    if codebook_sizes is None:
        codebook_sizes = (512, 512, 512)
    codebook_sizes = [int(x) for x in codebook_sizes]

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected sid2vid object (dict of SID keys)")

    children: Dict[Tuple[int, ...], Set[int]] = {}
    all_tokens: Set[int] = set()
    max_depth = 0
    n = 0
    for key in data:
        parts = [p.strip() for p in str(key).split(",") if p.strip()]
        if not parts:
            continue
        codes = [int(x) for x in parts]
        try:
            tokens = _codes_to_token_ids(codes, token_base, codebook_sizes)
        except ValueError as e:
            logger.warning("Skip invalid SID %r: %s", key, e)
            continue
        n += 1
        max_depth = max(max_depth, len(tokens))
        for i, tok in enumerate(tokens):
            all_tokens.add(tok)
            children.setdefault(tuple(tokens[:i]), set()).add(tok)

    if n == 0:
        return BeamValidPathTrie(mode="none")
    logger.info(
        "Built beam valid-path trie from sid2vid %s: %d SIDs, max_depth=%d, "
        "nodes=%d, token_base=%d, codebooks=%s",
        path,
        n,
        max_depth,
        len(children),
        token_base,
        list(codebook_sizes),
    )
    return _finalize_trie(
        BeamValidPathTrie(
            children=children,
            all_tokens=all_tokens,
            max_depth=max_depth,
            mode="trie",
            token_base=int(token_base),
            vocab_size=int(sum(codebook_sizes)),
        )
    )


def build_trie_from_sequences(sequences: Iterable[Sequence[int]]) -> BeamValidPathTrie:
    children: Dict[Tuple[int, ...], Set[int]] = {}
    max_depth = 0
    all_tokens: Set[int] = set()
    n = 0
    for seq in sequences:
        if not seq:
            continue
        n += 1
        max_depth = max(max_depth, len(seq))
        for i, tok in enumerate(seq):
            tok = int(tok)
            all_tokens.add(tok)
            key = tuple(int(x) for x in seq[:i])
            children.setdefault(key, set()).add(tok)
    if n == 0:
        return BeamValidPathTrie(mode="none")
    logger.info(
        "Built beam valid-path trie: %d sequences, max_depth=%d, nodes=%d",
        n,
        max_depth,
        len(children),
    )
    return _finalize_trie(
        BeamValidPathTrie(
            children=children,
            all_tokens=all_tokens,
            max_depth=max_depth,
            mode="trie",
        )
    )


def build_codebook_constraint(
    special_token_ids: Sequence[int],
    codebook_sizes: Sequence[int],
) -> BeamValidPathTrie:
    """Partition special ids (sorted order preserved) into per-step codebooks."""
    ids = [int(x) for x in special_token_ids]
    sizes = [int(s) for s in codebook_sizes]
    if any(s <= 0 for s in sizes):
        raise ValueError("sid-codebook-sizes must be positive integers")
    need = sum(sizes)
    if need > len(ids):
        raise ValueError(
            f"sid-codebook-sizes sum ({need}) exceeds "
            f"sid_token_range size ({len(ids)})"
        )
    step_allow: List[Optional[Set[int]]] = []
    offset = 0
    for s in sizes:
        step_allow.append(set(ids[offset : offset + s]))
        offset += s
    all_tokens = set(ids[:need])
    logger.info(
        "Built beam codebook constraint: steps=%s covering %d/%d special ids",
        list(sizes),
        need,
        len(ids),
    )
    return _finalize_trie(
        BeamValidPathTrie(
            step_allow=step_allow,
            all_tokens=all_tokens,
            max_depth=len(sizes),
            mode="codebook",
            token_base=int(ids[0]) if ids else 0,
            vocab_size=need,
        )
    )


def build_flat_allowlist(special_token_ids: Sequence[int]) -> BeamValidPathTrie:
    ids = set(int(x) for x in special_token_ids)
    if not ids:
        return BeamValidPathTrie(mode="none")
    lo = min(ids)
    hi = max(ids)
    return _finalize_trie(
        BeamValidPathTrie(
            step_allow=[ids],
            all_tokens=ids,
            max_depth=0,
            mode="flat",
            token_base=lo,
            vocab_size=hi - lo + 1,
        )
    )


def build_beam_valid_path(
    *,
    sid_file: Optional[str] = None,
    codebook_sizes: Optional[Sequence[int]] = None,
    special_token_ids: Optional[Sequence[int]] = None,
) -> BeamValidPathTrie:
    """Factory: prefer trie file, then codebook sizes, then flat allowlist.

    For GenRec ``sid2vid.json`` (dict keys ``\"c0,c1,c2\"``), codebook indices are
    remapped to lm_head token ids using ``special_token_ids[0]`` as base and
    ``codebook_sizes`` (default ``512,512,512`` when unset).
    """
    if sid_file:
        if is_sid2vid_json(sid_file):
            if not special_token_ids:
                raise ValueError(
                    "sid2vid.json requires a SID token range (inferred from "
                    "the tokenizer, or pass --sid / --sid-token-range) to map "
                    "codebook indices to token ids"
                )
            sizes = list(codebook_sizes) if codebook_sizes else [512, 512, 512]
            trie = build_trie_from_sid2vid_file(
                sid_file,
                token_base=int(special_token_ids[0]),
                codebook_sizes=sizes,
            )
            if trie.active:
                return trie
            logger.warning("sid-vocab-file %s produced empty sid2vid trie", sid_file)
        else:
            seqs = load_sequences_from_file(sid_file)
            # If sequences look like small codebook indices and we have special ids,
            # remap them the same way as sid2vid.
            if (
                special_token_ids
                and seqs
                and max(max(s) for s in seqs) < 4096
                and int(special_token_ids[0]) > 4096
            ):
                sizes = list(codebook_sizes) if codebook_sizes else [512, 512, 512]
                remapped = [
                    _codes_to_token_ids(s, int(special_token_ids[0]), sizes)
                    for s in seqs
                ]
                trie = build_trie_from_sequences(remapped)
                trie.finalize_index(
                    token_base=int(special_token_ids[0]),
                    vocab_size=int(sum(sizes)),
                )
            else:
                trie = build_trie_from_sequences(seqs)
            if trie.active:
                return trie
            logger.warning("sid-vocab-file %s produced empty trie", sid_file)

    if codebook_sizes and special_token_ids:
        return build_codebook_constraint(special_token_ids, codebook_sizes)

    if special_token_ids:
        return build_flat_allowlist(special_token_ids)

    return BeamValidPathTrie(mode="none")
