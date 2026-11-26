"""Triton-style iteration loop: dynamic preferred batch + in-flight prefill."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, TypeVar

from flashrec.scheduler.batching import should_prefer_decode

T = TypeVar("T")


def _prefix_fit(
    waiting: Sequence,
    remaining: int,
    beam_width_of: Callable,
    *,
    row_remaining: Optional[int] = None,
    row_width_of: Optional[Callable] = None,
) -> tuple:
    take = 0
    need = 0
    row_need = 0
    room = int(remaining)
    row_room = room if row_remaining is None else int(row_remaining)
    row_width_of = row_width_of or beam_width_of
    for req in waiting:
        width = int(beam_width_of(req))
        rows = int(row_width_of(req))
        if need + width > room or row_need + rows > row_room:
            break
        need += width
        row_need += rows
        take += 1
    return take, need, row_need


def can_inflight_prefill(
    waiting: Sequence,
    used_slots: int,
    budget: int,
    min_reqs: int = 8,
    beam_width_of: Callable = lambda r: r.beam_width,
    pack_min: int = 6,
    pack_ratio: float = 0.75,
    used_rows: Optional[int] = None,
    row_budget: Optional[int] = None,
    row_width_of: Optional[Callable] = None,
) -> bool:
    """Admit mid-wave prefill for a full 8/16, or a near-full leftover pack.

    1–5 late arrivals still wait (keeps 400-row decode intact). 6–7 into a
    leftover 400 slots is the 90 QPS Poisson case: they almost fill the GPU.
    """
    remaining = int(budget) - int(used_slots)
    row_width_of = row_width_of or beam_width_of
    rows_used = int(used_slots if used_rows is None else used_rows)
    rows_budget = int(budget if row_budget is None else row_budget)
    row_remaining = rows_budget - rows_used
    if remaining <= 0 or row_remaining <= 0 or not waiting:
        return False
    min_p = max(int(min_reqs), 1)
    if len(waiting) >= min_p:
        need = sum(int(beam_width_of(r)) for r in waiting[:min_p])
        row_need = sum(int(row_width_of(r)) for r in waiting[:min_p])
        if need <= remaining and row_need <= row_remaining:
            return True
    take, need, _ = _prefix_fit(
        waiting,
        remaining,
        beam_width_of,
        row_remaining=row_remaining,
        row_width_of=row_width_of,
    )
    if take < max(int(pack_min), 1):
        return False
    return need >= remaining * float(pack_ratio)


def take_prefill_batch(
    waiting: List,
    *,
    running_empty: bool,
    used_slots: int,
    budget: int,
    preferred: Sequence[int],
    inflight_min: int,
    beam_width_of: Callable = lambda r: r.beam_width,
    pack_min: int = 6,
    pack_ratio: float = 0.75,
    allow_topup: bool = True,
    used_rows: Optional[int] = None,
    row_budget: Optional[int] = None,
    row_width_of: Optional[Callable] = None,
) -> List:
    """Pop a prefill batch from ``waiting``. Mutates the list.

    Idle: take the largest preferred size we have, else the whole handful
    (conc=1). In-flight: preferred 8/16, a near-full leftover pack (6–7),
    or a slot-fitting top-up so conc=8 stays full as replacements arrive.
    """
    if not waiting:
        return []
    row_width_of = row_width_of or beam_width_of
    rows_used = int(used_slots if used_rows is None else used_rows)
    rows_budget = int(budget if row_budget is None else row_budget)

    def _fits(k: int) -> bool:
        slot_need = sum(int(beam_width_of(waiting[i])) for i in range(k))
        row_need = sum(int(row_width_of(waiting[i])) for i in range(k))
        return (
            int(used_slots) + slot_need <= int(budget)
            and rows_used + row_need <= rows_budget
        )

    prefs = sorted({int(x) for x in preferred if int(x) > 0})
    min_p = max(int(inflight_min), 1)
    if not prefs:
        prefs = [min_p]
    n = len(waiting)
    if not running_empty:
        candidates = [k for k in sorted(prefs, reverse=True) if n >= k]
        if min_p not in candidates and n >= min_p:
            candidates.append(min_p)
        for k in candidates:
            if _fits(k):
                batch = waiting[:k]
                del waiting[:k]
                return batch
        remaining = int(budget) - int(used_slots)
        row_remaining = rows_budget - rows_used
        take, need, _ = _prefix_fit(
            waiting,
            remaining,
            beam_width_of,
            row_remaining=row_remaining,
            row_width_of=row_width_of,
        )
        if take >= max(int(pack_min), 1) and need >= remaining * float(pack_ratio):
            batch = waiting[:take]
            del waiting[:take]
            return batch
        # SGLang-style 1-for-1 top-up: keep conc=8 full as replacements arrive.
        if allow_topup and take >= 1:
            batch = waiting[:take]
            del waiting[:take]
            return batch
        return []
    for k in sorted(prefs, reverse=True):
        if n >= k and _fits(k):
            batch = waiting[:k]
            del waiting[:k]
            return batch
    take, _, _ = _prefix_fit(
        waiting,
        int(budget) - int(used_slots),
        beam_width_of,
        row_remaining=rows_budget - rows_used,
        row_width_of=row_width_of,
    )
    batch = waiting[:take]
    del waiting[:take]
    return batch


def _live_width_of(req, fallback: Callable) -> int:
    """``req.live_width`` when the request has one, else its requested width."""
    live = getattr(req, "live_width", None)
    if live is None:
        return int(fallback(req))
    try:
        return max(int(live), 0)
    except (TypeError, ValueError):
        return int(fallback(req))


class InflightLoop:
    """Persistent waiting/running sets. One ``step`` is one scheduler tick."""

    def __init__(
        self,
        slots: int,
        preferred: Sequence[int],
        inflight_min: int,
        short_genrec: bool,
        prefill_fn: Callable[[List], None],
        decode_step_fn: Callable[[List], None],
        complete_fn: Callable[[List], None],
        beam_width_of: Optional[Callable] = None,
        generated_len_of: Optional[Callable] = None,
        is_finished: Optional[Callable] = None,
        free_fn: Optional[Callable[[List], None]] = None,
        pack_min: int = 6,
        pack_ratio: float = 0.75,
        kv_check_fn: Optional[Callable[[int], bool]] = None,
        slot_cost_of: Optional[Callable] = None,
        row_cap: Optional[int] = None,
        on_admit_fn: Optional[Callable[[List], None]] = None,
    ):
        self.slots = max(int(slots), 1)
        self.preferred = [int(x) for x in preferred if int(x) > 0] or [8, 16]
        self.inflight_min = max(int(inflight_min), 1)
        self.short_genrec = bool(short_genrec)
        self.prefill_fn = prefill_fn
        self.decode_step_fn = decode_step_fn
        self.complete_fn = complete_fn
        self.free_fn = free_fn
        self.pack_min = max(int(pack_min), 1)
        self.pack_ratio = float(pack_ratio)
        self.kv_check_fn = kv_check_fn
        self.beam_width_of = beam_width_of or (lambda r: int(r.beam_width))
        self.slot_cost_of = slot_cost_of or self.beam_width_of
        self.row_cap = max(int(row_cap if row_cap is not None else slots), 1)
        self.on_admit_fn = on_admit_fn
        self.stat_prefill_waves = 0
        self.stat_prefill_requests = 0
        self.stat_prefill_multi_waves = 0
        self.stat_decode_waves = 0
        self.stat_decode_requests = 0
        self.stat_decode_multi_waves = 0
        self.stat_decode_max_requests = 0
        self.stat_decode_max_rows = 0
        self.generated_len_of = generated_len_of or (
            lambda r: (
                int(r.beam_list.generated_len()) if getattr(r, "beam_list", None) else 0
            )
        )
        self.is_finished = is_finished or (
            lambda r: bool(getattr(r, "finished", False))
        )
        self.waiting: List = []
        self.running: List = []

    def submit(self, req) -> None:
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def used_slots(self) -> int:
        return sum(int(self.slot_cost_of(r)) for r in self.running)

    def waiting_slots(self) -> int:
        return sum(int(self.slot_cost_of(r)) for r in self.waiting)

    def used_rows(self) -> int:
        """Requested beam rows in the running set, independent of admission cost.

        This is the *requested* width sum, which is what admission and the row
        cap are charged against. It is not the number of rows the decode step
        actually drives -- see ``used_live_rows``.
        """
        return sum(int(self.beam_width_of(r)) for r in self.running)

    def waiting_rows(self) -> int:
        return sum(int(self.beam_width_of(r)) for r in self.waiting)

    def used_live_rows(self) -> int:
        """Rows the decode step actually drives right now.

        ``BeamRequest.live_width`` drops below the requested width when a
        candidate window is narrower than the beam, and can widen again on a
        later step, so this is not monotone. Reporting it next to
        ``used_rows`` is what lets a row-accounting claim distinguish "the
        budget was charged n" from "n rows were really scheduled".
        """
        return sum(_live_width_of(r, self.beam_width_of) for r in self.running)

    def waiting_live_rows(self) -> int:
        return sum(_live_width_of(r, self.beam_width_of) for r in self.waiting)

    def wants_prefill(self) -> bool:
        if not self.waiting:
            return False
        if not self.running:
            return True
        if should_prefer_decode(
            self.running,
            self.waiting,
            self.used_slots(),
            self.slots,
            short_genrec=self.short_genrec,
        ):
            return False
        need = int(self.slot_cost_of(self.waiting[0]))
        row_need = int(self.beam_width_of(self.waiting[0]))
        if self.used_slots() + need > self.slots:
            return False
        if self.used_rows() + row_need > self.row_cap:
            return False
        if self.kv_check_fn is not None and not self.kv_check_fn(row_need):
            return False
        return True

    def _maybe_prefill(self) -> None:
        if not self.wants_prefill():
            return
        batch = take_prefill_batch(
            self.waiting,
            running_empty=not self.running,
            used_slots=self.used_slots(),
            budget=self.slots,
            preferred=self.preferred,
            inflight_min=self.inflight_min,
            beam_width_of=self.slot_cost_of,
            pack_min=self.pack_min,
            pack_ratio=self.pack_ratio,
            allow_topup=True,
            used_rows=self.used_rows(),
            row_budget=self.row_cap,
            row_width_of=self.beam_width_of,
        )
        if not batch:
            return
        self.stat_prefill_waves += 1
        self.stat_prefill_requests += len(batch)
        if len(batch) >= 2:
            self.stat_prefill_multi_waves += 1
        if self.on_admit_fn is not None:
            self.on_admit_fn(batch)
        self.prefill_fn(batch)
        done = [r for r in batch if self.is_finished(r)]
        self.running.extend(r for r in batch if not self.is_finished(r))
        if done:
            self.complete_fn(done)

    def _decode_running(self) -> List:
        """One forward over all live reqs. Returns newly finished reqs."""
        live = [r for r in self.running if not self.is_finished(r)]
        if live:
            rows = sum(_live_width_of(req, self.beam_width_of) for req in live)
            self.stat_decode_waves += 1
            self.stat_decode_requests += len(live)
            if len(live) >= 2:
                self.stat_decode_multi_waves += 1
            if len(live) > self.stat_decode_max_requests:
                self.stat_decode_max_requests = len(live)
            if rows > self.stat_decode_max_rows:
                self.stat_decode_max_rows = rows
            self.decode_step_fn(live)
        done = [r for r in self.running if self.is_finished(r)]
        self.running = [r for r in self.running if not self.is_finished(r)]
        return done

    def _release_done(
        self, done: List, peek: Optional[Callable[[], None]] = None
    ) -> None:
        """Free slots and complete. Prefill waits for the next tick (XOR)."""
        if not done:
            return
        if self.free_fn is not None:
            self.free_fn(done)
        if peek is not None:
            peek()
        self.complete_fn(done)

    def _unexpanded(self) -> bool:
        return bool(self.running) and not any(
            bool(getattr(r, "expanded", False)) for r in self.running
        )

    def _fill_before_expand(self, peek: Optional[Callable[[], None]] = None) -> None:
        """SGLang fill-before-first-expand: pack to 8 before the first decode."""
        target = max(self.inflight_min, self.preferred[0] if self.preferred else 8)
        while self._unexpanded() and len(self.running) < target:
            if peek is not None:
                peek()
            n_before = len(self.running)
            w_before = len(self.waiting)
            self._maybe_prefill()
            if len(self.running) >= target:
                break
            if len(self.running) == n_before and len(self.waiting) == w_before:
                break

    def step(self) -> None:
        self.running = [r for r in self.running if not self.is_finished(r)]
        self._maybe_prefill()
        self._fill_before_expand()
        if self.running:
            self._release_done(self._decode_running())

    def run_burst(
        self, peek: Optional[Callable[[], None]] = None, max_steps: int = 4096
    ) -> None:
        """Keep decode on-GPU; peek between steps; prefill via side stream when possible.

        Barrier waves of 8 unique prompts already hit ~100ms / ~80 RPS. Mid-wave
        XOR (prefill-only ticks) stretched EvalScope 1-for-1 to ~146ms — do not
        skip decode when top-ups arrive; rely on prefill_stream overlap instead.
        """
        steps = 0
        while self.has_work() and steps < int(max_steps):
            if peek is not None:
                peek()
            self.running = [r for r in self.running if not self.is_finished(r)]
            self._maybe_prefill()
            self._fill_before_expand(peek)
            if self.running:
                done = self._decode_running()
                if done:
                    if self.free_fn is not None:
                        self.free_fn(done)
                    self.complete_fn(done)
            elif not self.waiting:
                break
            steps += 1
