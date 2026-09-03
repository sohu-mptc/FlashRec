"""Torch profiler control for FlashRec HTTP /start_profile /stop_profile.

Start/stop must run on the GPU worker thread (same stream as decode). The
HTTP handlers only enqueue a command; ``poll`` / ``on_forward`` apply it.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def _default_output_dir() -> Path:
    raw = (
        os.getenv("FLASHREC_TORCH_PROFILER_DIR")
        or os.getenv("SGLANG_TORCH_PROFILER_DIR")
        or "/tmp"
    )
    return Path(raw).expanduser()


@contextmanager
def trace_range(name: str):
    """Chrome-trace + NVTX range around a scheduler region."""
    with torch.profiler.record_function(name):
        pushed = False
        if torch.cuda.is_available():
            try:
                torch.cuda.nvtx.range_push(name)
                pushed = True
            except Exception:
                pass
        try:
            yield
        finally:
            if pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass


@dataclass
class ProfileCommand:
    output_dir: Optional[str] = None
    start_step: Optional[int] = None
    num_steps: Optional[int] = None
    activities: Optional[List[str]] = None
    profile_by_stage: bool = False
    with_stack: Optional[bool] = None
    record_shapes: Optional[bool] = None
    profile_prefix: Optional[str] = None
    profile_id: Optional[str] = None


class TorchProfiler:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending_start: Optional[ProfileCommand] = None
        self._pending_stop = False
        self._wake = threading.Event()
        self._in_progress = False
        self._configured = False
        self._profiler = None
        self.forward_ct = 0
        self._start_forward_ct: Optional[int] = None
        self._stage_prefill_ct = 0
        self._stage_decode_ct = 0
        self._target_stage_ct: Optional[int] = None
        self._current_stage: Optional[str] = None
        self._cmd: Optional[ProfileCommand] = None
        self._profiled_ct = 0
        self.output_dir: Path = _default_output_dir()

    def has_work(self) -> bool:
        return self._wake.is_set()

    @property
    def in_progress(self) -> bool:
        with self._lock:
            return self._in_progress or self._pending_start is not None

    def request_start(self, cmd: ProfileCommand) -> tuple[bool, str]:
        with self._lock:
            if self._in_progress or self._pending_start is not None:
                return (
                    False,
                    "Profiling is already in progress. Call /stop_profile first.",
                )
            if not cmd.profile_id:
                cmd.profile_id = str(time.time())
            self._pending_start = cmd
            self._pending_stop = False
            self._wake.set()
        return True, "Succeeded"

    def request_stop(self) -> tuple[bool, str]:
        with self._lock:
            if not (
                self._in_progress or self._pending_start is not None or self._configured
            ):
                return False, "Profiling is not in progress. Call /start_profile first."
            self._pending_stop = True
            self._wake.set()
        return True, "Succeeded"

    def poll(self) -> None:
        """Apply pending start/stop on the worker thread."""
        start_cmd: Optional[ProfileCommand] = None
        do_stop = False
        with self._lock:
            if self._pending_stop:
                do_stop = True
                self._pending_stop = False
            if self._pending_start is not None and not do_stop:
                start_cmd = self._pending_start
                self._pending_start = None
        if do_stop:
            self._stop(final=True)
        if start_cmd is not None:
            self._configure(start_cmd)
            if start_cmd.start_step is None and not start_cmd.profile_by_stage:
                self._start()
        with self._lock:
            if self._pending_start is None and not self._pending_stop:
                self._wake.clear()

    def on_forward(self, is_prefill: bool) -> None:
        """Call *before* the model forward so this kernel is in the trace."""
        self.forward_ct += 1
        self.poll()
        stage = "prefill" if is_prefill else "decode"
        if not self._configured:
            return
        cmd = self._cmd
        if cmd is None:
            return
        if cmd.profile_by_stage:
            self._step_by_stage(stage)
            return
        if (
            self._start_forward_ct is not None
            and not self._in_progress
            and self.forward_ct >= self._start_forward_ct
        ):
            self._start(stage_suffix=stage)

    def after_forward(self, is_prefill: bool) -> None:
        """Call *after* the model forward so num_steps counts completed kernels."""
        if not self._in_progress:
            return
        self._profiled_ct += 1
        cmd = self._cmd
        if cmd is None:
            return
        if cmd.profile_by_stage:
            return
        if cmd.num_steps is not None and self._profiled_ct >= int(cmd.num_steps):
            stage = "prefill" if is_prefill else "decode"
            self._stop(stage_suffix=stage, final=True)

    def _configure(self, cmd: ProfileCommand) -> None:
        self._cmd = cmd
        self._configured = True
        self._profiled_ct = 0
        self.output_dir = (
            Path(cmd.output_dir).expanduser()
            if cmd.output_dir
            else _default_output_dir()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if cmd.start_step:
            self._start_forward_ct = max(int(cmd.start_step), self.forward_ct + 1)
        else:
            self._start_forward_ct = None
        if cmd.num_steps and cmd.profile_by_stage:
            self._target_stage_ct = int(cmd.num_steps)
            self._stage_prefill_ct = 0
            self._stage_decode_ct = 0
        else:
            self._target_stage_ct = None
        logger.info(
            "Profiler configured: dir=%s id=%s num_steps=%s by_stage=%s",
            self.output_dir,
            cmd.profile_id,
            cmd.num_steps,
            cmd.profile_by_stage,
        )

    def _activities(self) -> List[torch.profiler.ProfilerActivity]:
        names = (
            list(self._cmd.activities)
            if self._cmd and self._cmd.activities
            else ["CPU", "GPU"]
        )
        mapping = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
            "CUDA": torch.profiler.ProfilerActivity.CUDA,
        }
        out: List[torch.profiler.ProfilerActivity] = []
        for name in names:
            act = mapping.get(str(name).upper())
            if act is None:
                continue
            if (
                act == torch.profiler.ProfilerActivity.CUDA
                and not torch.cuda.is_available()
            ):
                continue
            if act not in out:
                out.append(act)
        if not out:
            out.append(torch.profiler.ProfilerActivity.CPU)
        return out

    def _start(self, stage_suffix: Optional[str] = None) -> None:
        if self._in_progress:
            return
        activities = self._activities()
        with_stack = (
            True
            if self._cmd is None or self._cmd.with_stack is None
            else bool(self._cmd.with_stack)
        )
        record_shapes = (
            False
            if self._cmd is None or self._cmd.record_shapes is None
            else bool(self._cmd.record_shapes)
        )
        self._profiler = torch.profiler.profile(
            activities=activities,
            with_stack=with_stack,
            record_shapes=record_shapes,
        )
        self._profiler.start()
        self._in_progress = True
        self._current_stage = stage_suffix
        self._profiled_ct = 0
        logger.info(
            "Profiling starts%s. Traces will be saved to %s (id=%s)",
            f" for {stage_suffix}" if stage_suffix else "",
            self.output_dir,
            None if self._cmd is None else self._cmd.profile_id,
        )

    def _stop(self, stage_suffix: Optional[str] = None, *, final: bool = False) -> None:
        if not self._in_progress and self._profiler is None:
            self._configured = False
            self._cmd = None
            return
        suffix = stage_suffix or self._current_stage
        logger.info("Stop profiling%s...", f" for {suffix}" if suffix else "")
        if self._profiler is not None:
            self._profiler.stop()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self._trace_path(suffix)
            self._profiler.export_chrome_trace(str(path))
            logger.info("Profiling done. Trace saved to %s", path)
        self._profiler = None
        self._in_progress = False
        self._current_stage = None
        keep = (
            not final
            and self._cmd is not None
            and self._cmd.profile_by_stage
            and suffix == "prefill"
        )
        if not keep:
            self._configured = False
            self._cmd = None
            self._start_forward_ct = None

    def _trace_path(self, stage_suffix: Optional[str]) -> Path:
        cmd = self._cmd
        parts: List[str] = []
        if cmd and cmd.profile_prefix:
            parts.append(str(cmd.profile_prefix).strip("-"))
        parts.append(str(cmd.profile_id if cmd and cmd.profile_id else time.time()))
        if stage_suffix:
            parts.append(stage_suffix)
        return self.output_dir / ("-".join(parts) + ".trace.json.gz")

    def _step_by_stage(self, stage: str) -> None:
        target = self._target_stage_ct
        if stage == "prefill":
            if self._stage_prefill_ct == 0:
                if self._in_progress and self._current_stage != "prefill":
                    self._stop(stage_suffix=self._current_stage)
                self._start(stage_suffix="prefill")
            self._stage_prefill_ct += 1
            if (
                target is not None
                and self._stage_prefill_ct > target
                and self._in_progress
            ):
                self._stop(stage_suffix="prefill")
        else:
            if self._stage_decode_ct == 0:
                if self._in_progress:
                    self._stop(stage_suffix=self._current_stage or "prefill")
                self._start(stage_suffix="decode")
            self._stage_decode_ct += 1
            if (
                target is not None
                and self._stage_decode_ct > target
                and self._in_progress
            ):
                self._stop(stage_suffix="decode", final=True)
