from __future__ import annotations

import threading
from types import SimpleNamespace

from flashrec.server.api import BeamRecServer


class _Inflight:
    def __init__(self) -> None:
        self.running = [object(), object()]

    def used_rows(self) -> int:
        return 250

    def used_slots(self) -> int:
        return 2000


def test_mark_admitted_tracks_scheduler_high_watermarks() -> None:
    server = BeamRecServer.__new__(BeamRecServer)
    server.config = SimpleNamespace(
        admission_unit="request", admission_request_rows=1000
    )
    server.engine = SimpleNamespace(inflight=_Inflight())
    server._stats_lock = threading.Lock()
    server._stats = {
        "admitted_requests": 0,
        "admitted_rows": 0,
        "rejected": 0,
        "peak_used_rows": 0,
        "peak_used_slots": 0,
        "peak_running_requests": 0,
    }
    requests = [SimpleNamespace(beam_width=50), SimpleNamespace(beam_width=500)]

    server._mark_admitted(requests)

    assert server._stats["admitted_requests"] == 2
    assert server._stats["admitted_rows"] == 550
    assert server._stats["peak_used_rows"] == 800
    assert server._stats["peak_used_slots"] == 4000
    assert server._stats["peak_running_requests"] == 4
    assert all(request.admitted_at is not None for request in requests)
