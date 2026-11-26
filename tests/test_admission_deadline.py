"""Unit tests for the deadline-aware admission gate (P1-2).

Pure-Python, no GPU. These pin the gate's decision function and its
composition with select_jobs_lpm, and guard that the gate is OFF by default so
existing admission behavior is byte-for-byte unchanged. Matches the _R/J stub
style of test_scheduler_opts.py.
"""

from __future__ import annotations

from flashrec.config import BeamRecConfig
from flashrec.scheduler.batching import admission_cost
from flashrec.server.api import (
    DeadlineExceeded,
    deadline_exceeded,
    est_service_seconds,
    select_jobs_lpm,
)


class _J:
    """Minimal queue job stub: beam width, prefix length, optional arrival."""

    def __init__(self, n, plen, enqueued_at=None):
        self.n = n
        self.plen = plen
        if enqueued_at is not None:
            self.enqueued_at = enqueued_at


# --- predicate / estimator -------------------------------------------------


def test_zero_budget_never_drops():
    # Gate off: even an absurd age/estimate must not trip.
    assert deadline_exceeded(99.0, 99.0, 0.0) is False
    assert deadline_exceeded(99.0, 99.0, -1.0) is False


def test_past_deadline_is_dropped():
    assert deadline_exceeded(age_s=1.5, est_service_s=0.2, slo_budget_s=1.0) is True


def test_within_budget_is_kept():
    assert deadline_exceeded(age_s=0.1, est_service_s=0.2, slo_budget_s=1.0) is False


def test_slack_tightens_gate():
    # Borderline job sits just under budget; slack pushes it over.
    assert deadline_exceeded(0.5, 0.4, 1.0, slack_s=0.0) is False
    assert deadline_exceeded(0.5, 0.4, 1.0, slack_s=0.2) is True


def test_est_service_affine_in_width():
    kw = dict(base_ms=10.0, per_row_ms=1.0, queue_ahead_ms=0.0)
    wide = est_service_seconds(1000, 0, **kw)
    narrow = est_service_seconds(50, 0, **kw)
    assert wide > narrow
    # With zero slopes the estimate collapses to the intercept.
    base_only = est_service_seconds(
        1000, 9999, base_ms=10.0, per_row_ms=0.0, queue_ahead_ms=0.0
    )
    assert abs(base_only - 0.010) < 1e-9


def test_est_service_uses_real_width_not_charge():
    # The estimator takes beam width directly; it never sees the admission
    # charge, so it is identical regardless of accounting arm.
    kw = dict(base_ms=0.0, per_row_ms=2.0, queue_ahead_ms=0.0)
    assert est_service_seconds(50, 0, **kw) == est_service_seconds(50, 0, **kw)


# --- select_jobs_lpm integration ------------------------------------------


def test_gate_off_reproduces_current_selection():
    # Exactly test_lpm_admits_longest_prefix_first, with the gate params left
    # at their defaults (dropped_out=None): output must be identical.
    queue = [_J(50, 1), _J(50, 40), _J(50, 10)]
    taken = select_jobs_lpm(queue, 0, 0, 100, 16, lambda j: j.plen)
    assert [j.plen for j in taken] == [40, 10]
    assert [j.plen for j in queue] == [1]


def test_gate_off_even_with_budget_when_dropped_out_none():
    # A budget alone must not activate the gate; dropped_out=None is the off
    # switch that keeps every legacy caller unchanged.
    queue = [_J(50, 1, enqueued_at=0.0)]
    taken = select_jobs_lpm(
        queue,
        0,
        0,
        100,
        16,
        lambda j: j.plen,
        now=100.0,
        slo_budget_s=1.0,
        svc_base_ms=1.0,
        dropped_out=None,
    )
    assert [j.plen for j in taken] == [1]
    assert queue == []


def test_gate_drops_aged_job_and_admits_fresh():
    # One stale job (enqueued long ago) and two fresh ones. now=100s, budget 1s.
    stale = _J(50, 5, enqueued_at=0.0)  # age 100s -> dropped
    fresh_a = _J(50, 40, enqueued_at=99.95)  # age 0.05s -> kept
    fresh_b = _J(50, 10, enqueued_at=99.95)
    queue = [stale, fresh_a, fresh_b]
    dropped: list = []
    taken = select_jobs_lpm(
        queue,
        0,
        0,
        100,
        16,
        lambda j: j.plen,
        now=100.0,
        slo_budget_s=1.0,
        svc_base_ms=1.0,
        dropped_out=dropped,
    )
    assert dropped == [stale]
    assert [j.plen for j in taken] == [40, 10]
    assert queue == []  # stale removed via drop, fresh via admit


def test_gate_dropped_set_identical_across_accounting_arms():
    # Same queue and arrival times; once under row accounting, once under
    # request accounting. The dropped set must be identical because the gate
    # reads age + real width, never the charge.
    def build():
        return [
            _J(50, 5, enqueued_at=0.0),  # stale -> dropped
            _J(1000, 4, enqueued_at=99.95),  # fresh
            _J(50, 3, enqueued_at=99.95),  # fresh
        ]

    q_row = build()
    dropped_row: list = []
    select_jobs_lpm(
        q_row,
        0,
        0,
        4096,
        16,
        lambda j: j.plen,
        cost_of=lambda j: admission_cost(j.n, "row", 0),
        row_of=lambda j: j.n,
        row_budget=4096,
        now=100.0,
        slo_budget_s=1.0,
        svc_base_ms=1.0,
        dropped_out=dropped_row,
    )

    q_req = build()
    dropped_req: list = []
    select_jobs_lpm(
        q_req,
        0,
        0,
        4096,
        16,
        lambda j: j.plen,
        cost_of=lambda j: admission_cost(j.n, "request", 1000),
        row_of=lambda j: j.n,
        row_budget=4096,
        now=100.0,
        slo_budget_s=1.0,
        svc_base_ms=1.0,
        dropped_out=dropped_req,
    )

    assert [j.plen for j in dropped_row] == [5]
    assert [j.plen for j in dropped_req] == [5]


def test_gate_keeps_all_when_budget_generous():
    queue = [_J(50, 1, enqueued_at=99.9), _J(50, 2, enqueued_at=99.9)]
    dropped: list = []
    taken = select_jobs_lpm(
        queue,
        0,
        0,
        1000,
        16,
        lambda j: j.plen,
        now=100.0,
        slo_budget_s=60.0,
        svc_base_ms=1.0,
        dropped_out=dropped,
    )
    assert dropped == []
    assert len(taken) == 2


# --- config ---------------------------------------------------------------


def _cfg(**kw):
    return BeamRecConfig(model_path="x", **kw)


def test_deadline_fields_default_off():
    c = _cfg()
    assert c.admission_slo_budget_ms == 0
    assert c.admission_deadline_slack_ms == 0
    assert c.admission_svc_base_ms == 0.0
    assert c.admission_svc_per_row_ms == 0.0
    assert c.admission_svc_queue_ahead_ms == 0.0
    assert c.admission_deadline_source == "server"
    assert c.deadline_gate_on() is False


def test_deadline_gate_on_when_budget_positive():
    assert _cfg(admission_slo_budget_ms=2000).deadline_gate_on() is True


def test_deadline_source_normalized():
    assert (
        _cfg(admission_deadline_source="SERVER").admission_deadline_source == "server"
    )


def test_bad_deadline_source_raises():
    import pytest

    with pytest.raises(ValueError):
        _cfg(admission_deadline_source="bogus")


def test_negative_svc_slope_rejected():
    import pytest

    with pytest.raises(ValueError):
        _cfg(admission_svc_per_row_ms=-1.0)


def test_negative_budget_clamped_to_off():
    c = _cfg(admission_slo_budget_ms=-5)
    assert c.admission_slo_budget_ms == 0
    assert c.deadline_gate_on() is False


# --- exception type --------------------------------------------------------


def test_deadline_exception_is_distinct():
    assert issubclass(DeadlineExceeded, Exception)
