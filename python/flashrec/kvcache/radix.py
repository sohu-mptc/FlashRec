"""Prompt-only prefix KV cache (decode tokens are never inserted).

Per-token trie: match the longest cached prefix, share physical KV slots,
refcount via lock so live requests are not evicted. Last prompt token is
intentionally left unmatched so extend always produces next-token logits.

Eviction follows SGLang RadixCache: maintain an ``evictable_leaves`` set and
pop LRU leaves from a heap. Do not scan the whole tree per token.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch


@dataclass(eq=False)
class _Node:
    children: Dict[int, _Node] = field(default_factory=dict)
    kv_index: int = -1
    lock: int = 0
    last_access: int = 0
    parent: Optional[_Node] = None
    token: int = -1

    def __hash__(self) -> int:
        return id(self)


class PrefixCache:
    def __init__(self, allocator=None, device=None):
        self.root = _Node()
        self.allocator = allocator
        self.device = device if device is not None else torch.device("cpu")
        self._clock = 0
        self.num_cached_tokens = 0
        self.evictable_leaves: Set[_Node] = set()
        # Persistent lazy LRU heap: entries are (last_access, id, node); stale
        # entries (node no longer evictable, or re-accessed since push) are
        # skipped at pop time. Keeps evict() amortized O(log n) instead of a
        # full heapify of every leaf per call (200ms freezes at cap).
        self._heap: List[Tuple[int, int, _Node]] = []

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _update_leaf_status(self, node: Optional[_Node]) -> None:
        if node is None or node is self.root:
            if node in self.evictable_leaves:
                self.evictable_leaves.discard(node)
            return
        if node.lock > 0 or node.children:
            self.evictable_leaves.discard(node)
            return
        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)
            heapq.heappush(self._heap, (node.last_access, id(node), node))

    def prefix_len(self, tokens: Sequence[int]) -> int:
        """Longest cached prefix. Does not ``_tick()`` (LPM peek)."""
        node = self.root
        n = 0
        for tok in tokens:
            child = node.children.get(int(tok))
            if child is None:
                break
            node = child
            n += 1
        return n

    def match_cpu(self, tokens: Sequence[int]) -> Tuple[int, List[int]]:
        node = self.root
        indices: List[int] = []
        now = self._tick()
        for tok in tokens:
            child = node.children.get(int(tok))
            if child is None:
                break
            node = child
            node.last_access = now
            indices.append(int(node.kv_index))
        return len(indices), indices

    def match(self, tokens: Sequence[int]) -> Tuple[int, torch.Tensor]:
        n, indices = self.match_cpu(tokens)
        if n == 0:
            return 0, torch.empty(0, dtype=torch.int64, device=self.device)
        return n, torch.tensor(indices, dtype=torch.int64, device=self.device)

    def _as_kv_list(self, kv_indices) -> List[int]:
        if isinstance(kv_indices, list):
            if not kv_indices or isinstance(kv_indices[0], int):
                return kv_indices
            return [int(x) for x in kv_indices]
        if isinstance(kv_indices, torch.Tensor):
            t = kv_indices.detach().reshape(-1)
            if t.device.type != "cpu":
                t = t.to("cpu")
            return [int(x) for x in t.tolist()]
        return [int(x) for x in kv_indices]

    def insert_cpu(self, tokens: Sequence[int], kv_indices) -> List[int]:
        """Insert ``tokens[i] -> kv_indices[i]``. Return canonical tree indices."""
        if len(tokens) == 0:
            return []
        kv_list = self._as_kv_list(kv_indices)
        if len(kv_list) != len(tokens):
            raise ValueError("tokens and kv_indices must have the same length")
        out: List[int] = []
        node = self.root
        now = self._tick()
        for i, tok in enumerate(tokens):
            t = int(tok)
            child = node.children.get(t)
            if child is None:
                child = _Node(
                    parent=node,
                    token=t,
                    kv_index=int(kv_list[i]),
                    last_access=now,
                )
                node.children[t] = child
                self.num_cached_tokens += 1
                self._update_leaf_status(node)
                self._update_leaf_status(child)
            else:
                child.last_access = now
            out.append(int(child.kv_index))
            node = child
        return out

    def insert_after(
        self,
        prefix: Sequence[int],
        suffix: Sequence[int],
        suffix_kv,
    ) -> List[int]:
        """Walk ``prefix`` (must exist), then insert ``suffix``. Return suffix KV."""
        if not suffix:
            return []
        kv_list = self._as_kv_list(suffix_kv)
        if len(kv_list) != len(suffix):
            raise ValueError("suffix and suffix_kv must have the same length")
        node = self.root
        for tok in prefix:
            child = node.children.get(int(tok))
            if child is None:
                break
            node = child
        out: List[int] = []
        now = self._tick()
        for i, tok in enumerate(suffix):
            t = int(tok)
            child = node.children.get(t)
            if child is None:
                child = _Node(
                    parent=node,
                    token=t,
                    kv_index=int(kv_list[i]),
                    last_access=now,
                )
                node.children[t] = child
                self.num_cached_tokens += 1
                self._update_leaf_status(node)
                self._update_leaf_status(child)
            else:
                child.last_access = now
            out.append(int(child.kv_index))
            node = child
        return out

    def insert(self, tokens: Sequence[int], kv_indices: torch.Tensor) -> torch.Tensor:
        """Insert ``tokens[i] -> kv_indices[i]``. Return canonical tree indices.

        Existing nodes keep their original KV slot; the caller should free any
        duplicate slots that differ from the returned tensor.
        """
        out = self.insert_cpu(tokens, kv_indices)
        if not out:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return torch.tensor(out, dtype=torch.int64, device=self.device)

    def lock(self, tokens: Sequence[int]) -> None:
        node = self.root
        for tok in tokens:
            child = node.children.get(int(tok))
            if child is None:
                return
            child.lock += 1
            self._update_leaf_status(child)
            node = child

    def unlock(self, tokens: Sequence[int]) -> None:
        node = self.root
        for tok in tokens:
            child = node.children.get(int(tok))
            if child is None:
                return
            child.lock = max(0, child.lock - 1)
            self._update_leaf_status(child)
            node = child

    def evict(self, num_tokens: int) -> int:
        """Free unlocked leaves (lazy LRU heap) until ``num_tokens`` slots are released."""
        need = max(int(num_tokens), 0)
        if need <= 0 or not self.evictable_leaves:
            return 0
        heap = self._heap
        freed_ids: List[int] = []
        while len(freed_ids) < need and heap:
            prio, nid, leaf = heapq.heappop(heap)
            if leaf not in self.evictable_leaves:
                continue  # stale: died or got children/lock since push
            if leaf.last_access != prio:
                # re-accessed since push; reinsert with fresh priority
                heapq.heappush(heap, (leaf.last_access, nid, leaf))
                continue
            kv = int(leaf.kv_index)
            parent = leaf.parent
            if parent is not None:
                parent.children.pop(leaf.token, None)
            leaf.parent = None
            self.evictable_leaves.discard(leaf)
            self.num_cached_tokens = max(0, self.num_cached_tokens - 1)
            if kv >= 0:
                freed_ids.append(kv)
            if parent is not None and parent is not self.root:
                self._update_leaf_status(parent)
        if freed_ids and self.allocator is not None:
            self.allocator.free(freed_ids)
        return len(freed_ids)
