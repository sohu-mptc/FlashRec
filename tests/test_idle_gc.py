"""Idle-window GC policy + STW repro (no model / GPU required)."""

from __future__ import annotations

import gc
import threading
import time
from types import SimpleNamespace

import pytest

import flashrec.server.api as api
from flashrec.config import BeamRecConfig
from flashrec.server.api import BeamRecServer, _tune_gc_after_startup


class TestTuneGc:
    def test_disables_automatic_gc(self):
        was_enabled = gc.isenabled()
        try:
            gc.enable()
            _tune_gc_after_startup()
            assert gc.isenabled() is False
        finally:
            if was_enabled:
                gc.enable()
            else:
                gc.disable()


@pytest.mark.parametrize(
    "full, expected",
    [
        (True, [(16, 0), (19, 2), (27, 0), (29, 2)]),
        (False, [(16, 0), (27, 0)]),
    ],
)
def test_idle_gc_waits_for_worker_and_rate_limits_full_scans(
    monkeypatch, full, expected
):
    server = BeamRecServer.__new__(BeamRecServer)
    server.config = BeamRecConfig(model_path="unused")
    if not full:
        server.config.idle_gc_full = False
    server._cv = threading.Condition()
    server._stop = False
    now = 0
    profiling = False
    server.profiler = SimpleNamespace(has_work=lambda: profiling)
    polls = iter(
        [
            # time, worker idle since, queued, profiling
            (11, None, False, False),
            (12, 0, True, False),
            (13, 0, False, True),
            (14, 13.5, False, False),
            (16, 13.5, False, False),
            (18, 13.5, False, False),
            (19, 13.5, False, False),
            (21, 13.5, False, False),
            (27, 13.5, False, False),
            (29, 13.5, False, False),
        ]
    )

    def wait(*, timeout):
        nonlocal now, profiling
        poll = next(polls, None)
        if poll is None:
            return True
        now, server._worker_idle_since, queued, profiling = poll
        server._queue = [object()] if queued else []
        return False

    server._idle_gc_stop = SimpleNamespace(wait=wait, is_set=lambda: False)
    collected = []

    def collect(generation):
        collected.append((now, generation))
        return 0

    monkeypatch.setattr(gc, "collect", collect)
    monkeypatch.setattr(
        api, "time", SimpleNamespace(monotonic=lambda: now, perf_counter=lambda: now)
    )
    server._idle_gc_loop(
        poll_s=0.5, idle_s=1, min_interval_s=8, full_idle_s=5, full_interval_s=10
    )

    assert collected == expected


def test_worker_reports_busy_while_waiting_to_fill_batch(monkeypatch):
    server = BeamRecServer.__new__(BeamRecServer)
    server.config = BeamRecConfig(model_path="unused", target_batch_requests=2)
    server._stop = False
    server._graph_ready = threading.Event()
    server._worker_idle_since = time.monotonic()
    server._queue = [SimpleNamespace(n=50)]
    server.engine = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        ensure_cuda_graph=lambda: None,
        has_work=lambda: False,
        inflight=SimpleNamespace(
            running=[],
            waiting=[],
            used_slots=lambda: 0,
            waiting_slots=lambda: 0,
            used_rows=lambda: 0,
            waiting_rows=lambda: 0,
        ),
    )
    server.profiler = SimpleNamespace(has_work=lambda: False, poll=lambda: None)
    server._run_warmup_iters = lambda: None
    server._pop_queue_locked = lambda *_: [server._queue.pop()]
    observed = []

    class ObservedCondition(threading.Condition):
        def wait(self, timeout=None):
            if not observed:
                observed.append((bool(server._queue), server._worker_idle_since))
            return super().wait(timeout)

    def submit(jobs, job_by_rid):
        server._stop = True

    server._cv = ObservedCondition()
    server._submit_jobs = submit
    monkeypatch.setattr(api, "batch_wait_seconds", lambda *args, **kwargs: 0.01)

    server._run_worker()

    assert observed == [(False, None)]


class TestGcStwRepro:
    """Reproduce the online GC signature without loading a model.

    Online logs showed empty ``gen=0`` / ``gen=1`` pauses (``collected=0``)
    every few minutes / ~4h because long-lived post-freeze objects (radix
    nodes, etc.) keep surviving. Exact wall times need a production-sized
    heap; these tests pin the *mechanism*: auto GC fires mid-allocation,
    scans live objects for nothing, and stops firing after ``gc.disable()``.
    """

    @staticmethod
    def _install_probe(pauses: list, *, min_ms: float = 0.0):
        state: dict = {}

        def _cb(phase: str, info: dict) -> None:
            if phase == "start":
                state["t0"] = time.perf_counter()
                return
            t0 = state.pop("t0", None)
            if t0 is None:
                return
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if dt_ms >= min_ms:
                pauses.append(
                    (
                        dt_ms,
                        int(info.get("generation", -1)),
                        int(info.get("collected", -1)),
                    )
                )

        gc.callbacks.append(_cb)
        return _cb

    def test_auto_gc_empty_scan_fires_then_disable_stops_it(self):
        was_enabled = gc.isenabled()
        old_threshold = gc.get_threshold()
        pauses: list = []
        cb = self._install_probe(pauses)
        live: list = []
        try:
            gc.enable()
            gc.collect()
            gc.collect()
            gc.freeze()
            # Aggressive thresholds so a small retained graph trips gen0/gen1
            # the same way production trips them over hours of traffic.
            gc.set_threshold(200, 5, 100)
            pauses.clear()
            for i in range(30_000):
                live.append({i: i, "x": [i, i + 1]})

            gens = {g for _, g, _ in pauses}
            assert 0 in gens or 1 in gens, pauses[:10]
            # Same signature as online: scan finds nothing still reachable.
            assert any(c == 0 for _, _, c in pauses), pauses[:10]

            pauses.clear()
            gc.disable()
            n_before = len(pauses)
            for i in range(30_000):
                live.append({i: i})
            assert len(pauses) == n_before
            assert gc.isenabled() is False
        finally:
            gc.callbacks.remove(cb)
            gc.set_threshold(*old_threshold)
            if was_enabled:
                gc.enable()
            else:
                gc.disable()
            if hasattr(gc, "unfreeze"):
                gc.unfreeze()
            del live

    def test_large_live_graph_empty_collect_is_measurable(self):
        """Scaled-down version of the online empty full-heap scan.

        Production gen1 was ~300ms over a much richer resident set. Here we
        only require a clearly non-trivial pause with ``collected=0``.
        """
        was_enabled = gc.isenabled()
        pauses: list = []
        cb = self._install_probe(pauses)
        live = None
        try:
            gc.enable()
            gc.collect()
            gc.collect()
            gc.freeze()
            # ~3e5 small containers is enough for a multi-ms collect(2) on
            # typical CI hosts while staying cheap for the default suite.
            live = [{i: [i, i + 1, i + 2]} for i in range(300_000)]
            pauses.clear()
            collected = int(gc.collect(2))
            gen2 = [p for p in pauses if p[1] == 2]
            assert collected == 0
            assert gen2, pauses
            assert gen2[0][2] == 0
            # Loose floor: proves we paid for a real scan, not a no-op.
            assert gen2[0][0] >= 0.5, gen2
        finally:
            gc.callbacks.remove(cb)
            if was_enabled:
                gc.enable()
            else:
                gc.disable()
            if hasattr(gc, "unfreeze"):
                gc.unfreeze()
            del live
