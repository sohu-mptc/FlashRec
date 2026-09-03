"""Slot-budget wave batching: Σ beam_width ≤ slots, soft-admit, prefer-decode-after-expand."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


def batch_wait_seconds(
    n_jobs: int,
    slots: int,
    budget: int,
    wait_s: float,
    wait_max_s: float,
    recent_batch: int = 1,
    target_reqs: int = 8,
) -> float:
    """Adaptive straggler wait before first expand.

    conc=1 (recent_batch==1): 1ms cap. High-QPS underfill or below
    ``target_reqs``: wait_max so 8-wide GenRec does not start as 1 then 7.
    """
    wait_s = max(float(wait_s), 0.0)
    wait_max_s = max(float(wait_max_s), 0.0)
    n_jobs = int(n_jobs)
    slots = int(slots)
    budget = max(int(budget), 1)
    target = max(int(target_reqs), 1)
    recent = max(int(recent_batch), 1)
    if n_jobs <= 0 or wait_s <= 0:
        return 0.0
    high_qps = recent >= 4
    under_target = n_jobs < target
    if n_jobs < 2 and not high_qps:
        return min(wait_s, 0.001)
    if (under_target or slots * 2 < budget) and wait_max_s > wait_s:
        return wait_max_s
    return wait_s


def pack_release_count(
    n_jobs: int,
    recent_batch: int = 1,
    target_reqs: int = 8,
    high_cap: int = 16,
) -> int:
    """How many jobs to hand to generate_many after the wait window.

    High-QPS (recent_batch>=4): only emit K in {8,16} when possible so the
    decode graph stays 400/800. 9 queued → 8 now, 1 stays in the HTTP queue.
    Conc=1 keeps the whole handful.
    """
    n = max(int(n_jobs), 0)
    if n <= 0:
        return 0
    if int(recent_batch) < 4:
        return n
    cap = max(int(high_cap), 1)
    target = max(int(target_reqs), 1)
    if n >= cap:
        return cap
    # Burst of 12–15: take them together (600–750 → 800-row graph)
    # instead of 8 now and 4–7 stranded until the wave ends.
    if n >= target + max(target // 2, 4):
        return n
    if n >= target:
        return target
    return n


def can_admit_job(
    used_slots: int,
    used_reqs: int,
    need: int,
    budget: int,
    soft_admit: int,
) -> bool:
    if used_reqs >= int(soft_admit):
        return False
    if used_reqs > 0 and used_slots + int(need) > int(budget):
        return False
    return True


def group_by_beam_depth(
    reqs: Sequence[T],
    beam_width_of: Callable[[T], int],
    generated_len_of: Callable[[T], int],
) -> List[List[T]]:
    """Keep SID fused-expand groups lock-step: same n and same generated_len."""
    buckets: Dict[Tuple[int, int], List[T]] = {}
    order: List[Tuple[int, int]] = []
    for req in reqs:
        key = (int(beam_width_of(req)), int(generated_len_of(req)))
        if key not in buckets:
            order.append(key)
            buckets[key] = []
        buckets[key].append(req)
    return [buckets[k] for k in order]


def should_prefer_decode(
    running: Sequence,
    waiting: Sequence,
    used_slots: int,
    budget: int,
    *,
    any_expanded: Optional[bool] = None,
    short_genrec: bool = False,
) -> bool:
    """Prefer decode when running is expanded and full, or nothing is waiting.

    Short GenRec (depth<=3) locks decode after the first expand so late
    prefills cannot interleave. Longer jobs still allow underfill top-up.
    """
    if not running:
        return False
    if any_expanded is None:
        any_expanded = any(bool(getattr(r, "expanded", False)) for r in running)
    if not any_expanded:
        return False
    if short_genrec:
        return True
    return int(used_slots) >= int(budget) or len(waiting) == 0
