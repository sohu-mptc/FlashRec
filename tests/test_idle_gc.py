"""Idle-window GC policy + STW repro (no model / GPU required)."""

from __future__ import annotations

import gc
import time

from flashrec.server.api import _idle_gc_should_run, _tune_gc_after_startup


class TestIdleGcPolicy:
    def test_idle_path(self):
        assert (
            _idle_gc_should_run(
                inflight=False,
                queued=False,
                idle_for=2.0,
                since_gc=60.0,
                idle_s=1.0,
                min_interval_s=30.0,
                max_defer_s=300.0,
            )
            == "idle"
        )

    def test_min_interval_blocks(self):
        assert (
            _idle_gc_should_run(
                inflight=False,
                queued=False,
                idle_for=10.0,
                since_gc=5.0,
                idle_s=1.0,
                min_interval_s=30.0,
                max_defer_s=300.0,
            )
            is None
        )

    def test_inflight_blocks(self):
        assert (
            _idle_gc_should_run(
                inflight=True,
                queued=False,
                idle_for=10.0,
                since_gc=600.0,
                idle_s=1.0,
                min_interval_s=30.0,
                max_defer_s=300.0,
            )
            is None
        )

    def test_deferred_between_bursts(self):
        assert (
            _idle_gc_should_run(
                inflight=False,
                queued=True,
                idle_for=0.0,
                since_gc=400.0,
                idle_s=1.0,
                min_interval_s=30.0,
                max_defer_s=300.0,
            )
            == "deferred"
        )

    def test_queued_not_yet_deferred(self):
        assert (
            _idle_gc_should_run(
                inflight=False,
                queued=True,
                idle_for=0.0,
                since_gc=60.0,
                idle_s=1.0,
                min_interval_s=30.0,
                max_defer_s=300.0,
            )
            is None
        )


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
