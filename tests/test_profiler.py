import gzip
import json
import tempfile
import time
from pathlib import Path

from flashrec.profiler import ProfileCommand, TorchProfiler, trace_range


class TestTorchProfiler:
    def test_start_stop_exports_chrome_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            prof = TorchProfiler()
            ok, _ = prof.request_start(
                ProfileCommand(
                    output_dir=tmp,
                    activities=["CPU"],
                    with_stack=False,
                    record_shapes=False,
                    profile_prefix="unit",
                )
            )
            assert ok
            prof.poll()
            assert prof.in_progress
            with trace_range("flashrec.dummy"):
                total = 0
                for i in range(2000):
                    total += i
                assert total > 0
            ok, _ = prof.request_stop()
            assert ok
            prof.poll()
            # Export runs on a writer thread so it cannot stall the GPU worker;
            # the file exists only once that thread is joined.
            assert prof.wait_for_exports(timeout=120.0)
            traces = list(Path(tmp).glob("*.trace.json.gz"))
            assert len(traces) == 1
            with gzip.open(traces[0], "rt") as f:
                data = json.load(f)
            events = data.get("traceEvents") or data
            names = {
                ev.get("name")
                for ev in events
                if isinstance(ev, dict) and ev.get("name")
            }
            assert any(
                "flashrec.dummy" in n or n == "flashrec.dummy" for n in names
            ), names

    def test_num_steps_auto_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            prof = TorchProfiler()
            prof.request_start(
                ProfileCommand(
                    output_dir=tmp,
                    activities=["CPU"],
                    num_steps=2,
                    with_stack=False,
                )
            )
            prof.poll()
            for _ in range(5):
                prof.on_forward(is_prefill=False)
                _ = sum(range(200))
                prof.after_forward(is_prefill=False)
            assert not prof._in_progress
            assert prof.wait_for_exports(timeout=120.0)
            assert list(Path(tmp).glob("*.trace.json.gz"))

    def test_stop_does_not_block_on_export(self):
        """poll() must return before the trace is written.

        export_chrome_trace() takes seconds on a real decode trace and used to
        run on the GPU worker thread, stalling every in-flight request (measured
        7.3s at beam=1000, which showed up as the benchmark's p99).
        """
        with tempfile.TemporaryDirectory() as tmp:
            prof = TorchProfiler()
            prof.request_start(
                ProfileCommand(output_dir=tmp, activities=["CPU"], with_stack=False)
            )
            prof.poll()
            for _ in range(50):
                prof.on_forward(is_prefill=False)
                prof.after_forward(is_prefill=False)
            prof.request_stop()
            t0 = time.perf_counter()
            prof.poll()
            stop_s = time.perf_counter() - t0
            assert prof.wait_for_exports(timeout=120.0)
            export = list(Path(tmp).glob("*.trace.json.gz"))
            assert export, "writer thread never produced a trace"
            # The handoff itself is a thread start; without it poll() would pay
            # the full serialization cost.
            assert stop_s < 1.0, f"poll() blocked {stop_s:.2f}s on export"

    def test_double_start_rejected(self):
        prof = TorchProfiler()
        ok, _ = prof.request_start(ProfileCommand(activities=["CPU"]))
        assert ok
        ok, msg = prof.request_start(ProfileCommand(activities=["CPU"]))
        assert not ok
        assert "already" in msg

    def test_stop_without_start_rejected(self):
        prof = TorchProfiler()
        ok, msg = prof.request_stop()
        assert not ok
        assert "not in progress" in msg
