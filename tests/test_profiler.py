import gzip
import json
import tempfile
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
            assert list(Path(tmp).glob("*.trace.json.gz"))

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
