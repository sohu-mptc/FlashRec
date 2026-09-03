"""N-way round-robin decode pipeline (SGLang beam_decode_pipeline).

Same-beam expand/KV cannot overlap the next forward of those beams. With N
independent request waves on one CUDA stream:

  [GPU: forward(i)]  [CPU: process(j) + prepare(j) + launch(j)]  for j != i

so postprocess of one wave hides behind another wave's GPU time.

Default N=2. Needs at least 2 concurrent requests to materialize. Do not launch
multiple GEMMs per tick — one forward at a time, expand of another pipe overlaps.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

import torch

T = TypeVar("T")

# (pipe_id, launch payload)
PipeResult = Tuple[int, object]


def _prefill_ready(req) -> bool:
    """True when this req's async prefill has finished (or never used a side stream).

    Leaves ``prefill_done`` in place so the first decode launch can
    ``wait_event`` it and establish a cross-stream dependency.
    """
    ev = getattr(req, "prefill_done", None)
    if ev is None:
        return True
    query = getattr(ev, "query", None)
    if not callable(query):
        return True
    try:
        return bool(query())
    except Exception:
        return True


def assign_buckets(reqs: Sequence[T], stages: int) -> List[List[T]]:
    """Round-robin reqs into ``stages`` buckets. Empty buckets are dropped."""
    n = len(reqs)
    if n == 0:
        return []
    k = max(int(stages), 1)
    out: List[List[T]] = [[] for _ in range(k)]
    for i, req in enumerate(reqs):
        out[i % k].append(req)
    return [p for p in out if p]


class DecodePipeline:
    def __init__(
        self,
        stages: int = 2,
        enabled: bool = True,
        idle_collapse_ms: float = 8.0,
    ):
        self.stages = max(int(stages), 1)
        self.enabled = bool(enabled)
        # Graph replay stays on the capture/default stream; expand/remap
        # overlap on expand_stream.
        self.expand_stream: Optional[torch.cuda.Stream] = None
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.prefill_stream: Optional[torch.cuda.Stream] = None
        self.idle_collapse_ms = max(float(idle_collapse_ms), 0.0)
        self.pipes: Optional[List[List]] = None
        self.turn: int = 0
        self.current_pipe_id: int = 0
        self.result_queue: Deque[PipeResult] = deque()
        self.pending_prefills: Deque[List] = deque()
        self.held_prefills: List = []
        self.idle_since: float = 0.0
        self.expand_done: Dict[Optional[int], object] = {}
        self.remap_done: Dict[Optional[int], object] = {}

    def ensure_streams(self) -> None:
        """Create expand/copy/prefill CUDA streams on first GPU use (not at import/test)."""
        if not self.enabled or not torch.cuda.is_available():
            return
        if self.expand_stream is None:
            self.expand_stream = torch.cuda.Stream()
        if self.copy_stream is None:
            self.copy_stream = torch.cuda.Stream()
        if self.prefill_stream is None:
            self.prefill_stream = torch.cuda.Stream()

    @property
    def active(self) -> bool:
        return self.pipes is not None

    def n_pipes(self) -> int:
        return len(self.pipes) if self.pipes is not None else 0

    def inflight(self, pipe_id: int) -> bool:
        return any(pid == pipe_id for pid, _ in self.result_queue)

    def split(self, reqs: Sequence[T]) -> List[List[T]]:
        """Bucket assignment without activating the state machine."""
        live = list(reqs)
        if not live:
            return []
        if not self.enabled or len(live) < 2:
            return [live]
        stages = max(2, min(self.stages, len(live)))
        return assign_buckets(live, stages)

    def try_materialize(self, reqs: Sequence, is_finished: Callable) -> bool:
        if not self.enabled or self.active:
            return self.active
        active = [r for r in reqs if not is_finished(r)]
        if len(active) < 2:
            return False
        stages = max(2, min(self.stages, len(active)))
        self.pipes = assign_buckets(active, stages)
        while len(self.pipes) < self.stages:
            self.pipes.append([])
        self.turn = 0
        self.current_pipe_id = 0
        self.result_queue.clear()
        self.pending_prefills.clear()
        self.held_prefills = []
        self.idle_since = 0.0
        self.expand_done.clear()
        self.remap_done.clear()
        logger.info(
            "decode pipeline materialized: stages=%d sizes=%s",
            self.n_pipes(),
            [len(p) for p in self.pipes],
        )
        return True

    def try_dematerialize(
        self,
        is_finished: Callable,
        now: Optional[float] = None,
        wait_events: Optional[Callable[[], None]] = None,
    ) -> None:
        """Sticky collapse: keep empty pipe slots while any work remains."""
        if not self.active:
            return
        if self.result_queue or self.pending_prefills or self.held_prefills:
            self.idle_since = 0.0
            return
        if any(not is_finished(r) for p in self.pipes for r in p):
            self.idle_since = 0.0
            return
        t = time.perf_counter() if now is None else float(now)
        if self.idle_since <= 0.0:
            self.idle_since = t
            return
        if (t - self.idle_since) * 1000.0 < self.idle_collapse_ms:
            return
        if wait_events is not None:
            wait_events()
        self.pipes = None
        self.result_queue.clear()
        self.pending_prefills.clear()
        self.held_prefills = []
        self.expand_done.clear()
        self.remap_done.clear()
        self.idle_since = 0.0

    def sync_pipes_from_running(self, running: Sequence, is_finished: Callable) -> None:
        """Drop finished reqs; spread newcomers; re-split a new idle wave."""
        if not self.active:
            return
        live = [r for r in running if not is_finished(r)]
        live_ids = {id(r) for r in live}
        known = set()
        for i, p in enumerate(self.pipes):
            kept = [r for r in p if id(r) in live_ids]
            self.pipes[i] = kept
            known.update(id(r) for r in kept)
        newcomers = [r for r in live if id(r) not in known]
        ready: List = []
        held: List = []
        for req in newcomers:
            if _prefill_ready(req):
                ready.append(req)
            else:
                held.append(req)
        self.held_prefills = held
        while self.n_pipes() < self.stages and len(live) >= 2:
            self.pipes.append([])
        # Only resplit a brand-new wave (all previous reqs gone). Mid-wave
        # leftover+top-up must keep pipe identities so 3-pipe overlap survives.
        if (
            not known
            and ready
            and not held
            and len(live) >= 2
            and not self.result_queue
            and not self.pending_prefills
        ):
            want = max(2, min(self.stages, len(live)))
            self.pipes = assign_buckets(live, want)
            while len(self.pipes) < self.stages:
                self.pipes.append([])
            self.turn = 0
            self.current_pipe_id = 0
            logger.info(
                "decode pipeline resized: stages=%d sizes=%s",
                self.n_pipes(),
                [len(p) for p in self.pipes],
            )
            return
        if ready:
            self.merge_prefill(ready)

    def free_pipes(self) -> List[int]:
        return [i for i in range(self.n_pipes()) if not self.inflight(i)]

    def _spread_reqs(self, reqs: Sequence) -> List:
        """Place reqs one-by-one onto the current smallest free pipe."""
        leftover: List = []
        for req in reqs:
            free = self.free_pipes()
            if not free:
                leftover.append(req)
                continue
            target = min(free, key=lambda i: (len(self.pipes[i]), i))
            self.pipes[target].append(req)
        return leftover

    def merge_prefill(self, reqs: Sequence) -> None:
        if not reqs or not self.active:
            return
        if not self.free_pipes():
            self.pending_prefills.append(list(reqs))
            return
        leftover = self._spread_reqs(reqs)
        if leftover:
            self.pending_prefills.append(leftover)

    def flush_pending_prefills(self) -> None:
        if not self.active:
            return
        while self.pending_prefills:
            if not self.free_pipes():
                break
            prefill = self.pending_prefills.popleft()
            leftover = self._spread_reqs(prefill)
            if leftover:
                self.pending_prefills.appendleft(leftover)
                break

    def _live(self, pipe_id: int, is_finished: Callable) -> List:
        return [r for r in self.pipes[pipe_id] if not is_finished(r)]

    def process_for_turn(self, turn: int, process_fn: Callable[[object], None]) -> bool:
        q = self.result_queue
        for i, (pid, launch) in enumerate(q):
            if pid != turn:
                continue
            del q[i]
            process_fn(launch)
            self.flush_pending_prefills()
            return True
        return False

    def pick_turn(self, is_finished: Callable) -> Optional[int]:
        if not self.active:
            return None
        n = self.n_pipes()
        turn = self.turn % n
        for step in range(n):
            cand = (turn + step) % n
            if self._live(cand, is_finished):
                return cand
        return None

    def get_next(
        self,
        process_fn: Callable[[object], None],
        is_finished: Callable,
    ) -> Optional[List]:
        """Process the prior result for a pipe and return live reqs to launch."""
        if not self.active:
            return None
        self.flush_pending_prefills()
        turn = self.pick_turn(is_finished)
        if turn is None:
            self.flush_pending_prefills()
            turn = self.pick_turn(is_finished)
        if turn is None:
            if self.result_queue:
                pid, _ = self.result_queue[0]
                self.process_for_turn(pid, process_fn)
                return self.get_next(process_fn, is_finished)
            return None

        if self.inflight(turn):
            n = self.n_pipes()
            alt = None
            for step in range(1, n):
                cand = (turn + step) % n
                if not self.inflight(cand) and self._live(cand, is_finished):
                    alt = cand
                    break
            if alt is None:
                self.process_for_turn(turn, process_fn)
                if self.inflight(turn):
                    return None
            else:
                turn = alt

        self.process_for_turn(turn, process_fn)
        live = self._live(turn, is_finished)
        if not live:
            self.turn = (turn + 1) % self.n_pipes()
            return self.get_next(process_fn, is_finished)
        self.current_pipe_id = turn
        self.turn = (turn + 1) % self.n_pipes()
        return live

    def enqueue(self, pipe_id: int, launch: object) -> None:
        self.result_queue.append((int(pipe_id), launch))

    def drain_all(self, process_fn: Callable[[object], None]) -> None:
        while self.result_queue:
            _, launch = self.result_queue.popleft()
            process_fn(launch)

    def run(
        self,
        reqs: Sequence,
        is_finished: Callable,
        launch_forward: Callable[[List], object],
        process_result: Callable[[object], None],
    ) -> None:
        """Drain to completion (CLI / tests). Serving uses get_next + enqueue."""
        live = [r for r in reqs if not is_finished(r)]
        if not live:
            return
        if self.try_materialize(live, is_finished):
            while True:
                batch = self.get_next(process_result, is_finished)
                if batch is None:
                    if self.result_queue:
                        self.drain_all(process_result)
                        continue
                    break
                launch = launch_forward(batch)
                self.enqueue(self.current_pipe_id, launch)
            self.drain_all(process_result)
            return
        while True:
            live = [r for r in reqs if not is_finished(r)]
            if not live:
                return
            launch = launch_forward(live)
            process_result(launch)
