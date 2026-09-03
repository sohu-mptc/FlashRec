"""Triton-style iteration loop: dynamic preferred batch + in-flight prefill."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, TypeVar

from flashrec.scheduler.batching import should_prefer_decode

T = TypeVar("T")


def _prefix_fit(waiting: Sequence, remaining: int, beam_width_of: Callable) -> tuple:
    take = 0
    need = 0
    room = int(remaining)
    for req in waiting:
        width = int(beam_width_of(req))
        if need + width > room:
            break
        need += width
        take += 1
    return take, need


def can_inflight_prefill(
    waiting: Sequence,
    used_slots: int,
    budget: int,
    min_reqs: int = 8,
    beam_width_of: Callable = lambda r: r.beam_width,
    pack_min: int = 6,
    pack_ratio: float = 0.75,
) -> bool:
    """Admit mid-wave prefill for a full 8/16, or a near-full leftover pack.

    1–5 late arrivals still wait (keeps 400-row decode intact). 6–7 into a
    leftover 400 slots is the 90 QPS Poisson case: they almost fill the GPU.
    """
    remaining = int(budget) - int(used_slots)
    if remaining <= 0 or not waiting:
        return False
    min_p = max(int(min_reqs), 1)
    if len(waiting) >= min_p:
        need = sum(int(beam_width_of(r)) for r in waiting[:min_p])
        if need <= remaining:
            return True
    take, need = _prefix_fit(waiting, remaining, beam_width_of)
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
) -> List:
    """Pop a prefill batch from ``waiting``. Mutates the list.

    Idle: take the largest preferred size we have, else the whole handful
    (conc=1). In-flight: preferred 8/16, a near-full leftover pack (6–7),
    or a slot-fitting top-up so conc=8 stays full as replacements arrive.
    """
    if not waiting:
        return []
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
            need = sum(int(beam_width_of(waiting[i])) for i in range(k))
            if int(used_slots) + need <= int(budget):
                batch = waiting[:k]
                del waiting[:k]
                return batch
        remaining = int(budget) - int(used_slots)
        take, need = _prefix_fit(waiting, remaining, beam_width_of)
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
        if n >= k:
            batch = waiting[:k]
            del waiting[:k]
            return batch
    batch = list(waiting)
    waiting.clear()
    return batch


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
        self.beam_width_of = beam_width_of or (lambda r: int(r.beam_width))
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
        return sum(int(self.beam_width_of(r)) for r in self.running)

    def waiting_slots(self) -> int:
        return sum(int(self.beam_width_of(r)) for r in self.waiting)

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
        need = int(self.beam_width_of(self.waiting[0]))
        return self.used_slots() + need <= self.slots

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
            beam_width_of=self.beam_width_of,
            pack_min=self.pack_min,
            pack_ratio=self.pack_ratio,
            allow_topup=True,
        )
        if not batch:
            return
        self.prefill_fn(batch)
        done = [r for r in batch if self.is_finished(r)]
        self.running.extend(r for r in batch if not self.is_finished(r))
        if done:
            self.complete_fn(done)

    def _decode_running(self) -> List:
        """One forward over all live reqs. Returns newly finished reqs."""
        live = [r for r in self.running if not self.is_finished(r)]
        if live:
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
