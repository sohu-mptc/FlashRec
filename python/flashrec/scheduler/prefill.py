"""Group waiting requests that share seq_len for one EXTEND launch."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Sequence, TypeVar

T = TypeVar("T")


def group_by_prompt_len(
    reqs: Sequence[T], prompt_len_of: Callable[[T], int]
) -> List[List[T]]:
    buckets: Dict[int, List[T]] = defaultdict(list)
    order: List[int] = []
    for req in reqs:
        n = int(prompt_len_of(req))
        if n not in buckets:
            order.append(n)
        buckets[n].append(req)
    return [buckets[k] for k in order]
