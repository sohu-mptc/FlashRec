"""OpenAI-compatible /v1/chat/completions with wave batching."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from flashrec.config import BeamRecConfig
from flashrec.core import BeamRequest, BeamResult
from flashrec.profiler import ProfileCommand, TorchProfiler, trace_range
from flashrec.scheduler.batching import (
    batch_wait_seconds,
    can_admit_job,
    pack_release_count,
)
from flashrec.scheduler.scheduler import BeamRecEngine

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ProfileReqInput(BaseModel):
    output_dir: Optional[str] = None
    start_step: Optional[int] = None
    num_steps: Optional[int] = None
    activities: Optional[List[str]] = None
    profile_by_stage: bool = False
    with_stack: Optional[bool] = None
    record_shapes: Optional[bool] = None
    profile_prefix: Optional[str] = None


def _opt_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    n: int = 1
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: float = 0.0
    stream: bool = False
    chat_template_kwargs: Dict[str, Any] = Field(default_factory=dict)
    separate_reasoning: bool = False


def _message_text(content: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") in (None, "text"):
            parts.append(str(item.get("text", "")))
        elif isinstance(item, str):
            parts.append(item)
    return "".join(parts)


def _dumps_json(obj: Any) -> bytes:
    try:
        import orjson

        return orjson.dumps(obj)
    except ImportError:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )


def _json_response(payload: Dict[str, Any]) -> Response:
    return Response(content=_dumps_json(payload), media_type="application/json")


def _plain_score(value) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def select_jobs_lpm(
    queue: List[_Job],
    used_slots: int,
    used_reqs: int,
    budget: int,
    soft_admit: int,
    prefix_len_of,
    aging_ms: int = 0,
    now: Optional[float] = None,
) -> List[_Job]:
    """Admit jobs from ``queue`` by longest prefix first. Mutates ``queue``.

    Pure prefix order starves short prompts: with a wide beam only one or two
    jobs fit per wave, so a steady stream of long shared-prefix prompts keeps
    jumping ahead of them. Jobs waiting longer than ``aging_ms`` are promoted
    ahead of the prefix ranking, oldest first.
    """
    t = time.monotonic() if now is None else float(now)
    aging_s = max(int(aging_ms), 0) / 1000.0
    scored = []
    for i, job in enumerate(queue):
        waited = t - getattr(job, "enqueued_at", t)
        aged = bool(aging_s) and waited >= aging_s
        # Aged jobs sort first by longest wait; the rest keep prefix order.
        scored.append(
            ((0, -waited, i) if aged else (1, -int(prefix_len_of(job)), i), job)
        )
    scored.sort(key=lambda x: x[0])
    taken: List[_Job] = []
    taken_ids = set()
    slots = int(used_slots)
    for _, job in scored:
        need = int(job.n)
        if not can_admit_job(slots, used_reqs + len(taken), need, budget, soft_admit):
            continue
        taken.append(job)
        taken_ids.add(id(job))
        slots += need
    if taken_ids:
        queue[:] = [job for job in queue if id(job) not in taken_ids]
    return taken


def _usage_dict(result: BeamResult) -> Dict[str, Any]:
    prompt_tokens = int(result.prompt_tokens)
    completion_tokens = int(result.completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": int(result.cached_tokens)},
    }


@dataclass
class _Job:
    oai: ChatCompletionRequest
    messages: List[Dict[str, Any]]
    n: int
    max_tokens: int
    future: asyncio.Future
    request: Optional[BeamRequest] = None
    ready: threading.Event = field(default_factory=threading.Event)
    error: Optional[BaseException] = None
    enqueued_at: float = field(default_factory=time.monotonic)


class BeamRecServer:
    def __init__(self, config: BeamRecConfig):
        self.config = config
        self.engine = BeamRecEngine(config)
        self.profiler = TorchProfiler()
        self.engine.attach_profiler(self.profiler)
        self._queue: List[_Job] = []
        self._cv = threading.Condition()
        self._stop = False
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        self.app = FastAPI(title="FlashRec")
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/v1/models", self.list_models, methods=["GET"])
        self.app.add_api_route(
            "/v1/chat/completions", self.chat_completions, methods=["POST"]
        )
        self.app.add_api_route(
            "/start_profile", self.start_profile, methods=["GET", "POST"]
        )
        self.app.add_api_route(
            "/stop_profile", self.stop_profile, methods=["GET", "POST"]
        )

    async def health(self):
        return {"status": "ok"}

    async def list_models(self):
        return {
            "object": "list",
            "data": [
                {
                    "id": self.config.model_path,
                    "object": "model",
                    "owned_by": "FlashRec",
                }
            ],
        }

    async def start_profile(self, request: Request):
        obj = ProfileReqInput()
        if request.method == "POST":
            try:
                data = await request.json()
                if isinstance(data, dict) and data:
                    obj = ProfileReqInput(**data)
            except Exception:
                pass
        cmd = ProfileCommand(
            output_dir=obj.output_dir,
            start_step=_opt_int(obj.start_step),
            num_steps=_opt_int(obj.num_steps),
            activities=obj.activities,
            profile_by_stage=bool(obj.profile_by_stage),
            with_stack=obj.with_stack,
            record_shapes=obj.record_shapes,
            profile_prefix=obj.profile_prefix,
        )
        ok, message = self.profiler.request_start(cmd)
        with self._cv:
            self._cv.notify()
        if not ok:
            return PlainTextResponse(message + "\n", status_code=400)
        return PlainTextResponse("Start profiling.\n", status_code=200)

    async def stop_profile(self):
        ok, message = self.profiler.request_stop()
        with self._cv:
            self._cv.notify()
        if not ok:
            return PlainTextResponse(message + "\n", status_code=400)
        return PlainTextResponse(
            "Stop profiling. This will take some time.\n", status_code=200
        )

    async def chat_completions(self, request: ChatCompletionRequest):
        messages = [
            {"role": m.role, "content": _message_text(m.content)}
            for m in request.messages
        ]
        n = max(int(request.n), 1)
        max_tokens = int(
            request.max_completion_tokens
            or request.max_tokens
            or self.config.max_tokens
        )
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        job = _Job(
            oai=request,
            messages=messages,
            n=n,
            max_tokens=max_tokens,
            future=fut,
        )
        self.engine.host_pool.submit(self._tokenize_job, job)
        with self._cv:
            self._queue.append(job)
            self._cv.notify()
        result: BeamResult = await fut
        if request.stream:
            # Beam search finishes before stream starts. Build SSE off the
            # event loop — 50 beams × many json.dumps on the loop cut EvalScope
            # conc=8 from ~80 RPS (non-stream smoke) to ~50.
            chunks = await loop.run_in_executor(
                None, self._stream_openai_bytes, result, request
            )

            async def _gen():
                for chunk in chunks:
                    yield chunk

            return StreamingResponse(_gen(), media_type="text/event-stream")
        return _json_response(self._to_openai(result, request))

    def _ensure_request(self, job: _Job) -> BeamRequest:
        if job.request is None:
            job.request = self.engine.make_request(
                messages=job.messages,
                n=job.n,
                max_tokens=job.max_tokens,
                chat_template_kwargs=job.oai.chat_template_kwargs,
                temperature=job.oai.temperature,
            )
        return job.request

    def _tokenize_job(self, job: _Job) -> None:
        try:
            self._ensure_request(job)
        except Exception as exc:
            job.error = exc
        finally:
            job.ready.set()

    def _wait_job(self, job: _Job) -> None:
        if not job.ready.is_set():
            job.ready.wait()
        if job.error is not None:
            raise job.error

    def _job_prefix_len(self, job: _Job) -> int:
        self._wait_job(job)
        self._ensure_request(job)
        cache = self.engine.prefix_cache
        if cache is None or job.request is None:
            return 0
        ids = job.request.input_ids
        if len(ids) <= 1:
            return 0
        return int(cache.prefix_len(ids[:-1]))

    def _pop_queue_locked(
        self, used_slots: int, used_reqs: int, budget: int, soft_admit: int
    ) -> List[_Job]:
        policy = str(getattr(self.config, "schedule_policy", "lpm") or "lpm").lower()
        if policy == "lpm" and self.engine.prefix_cache is not None:
            return select_jobs_lpm(
                self._queue,
                used_slots,
                used_reqs,
                budget,
                soft_admit,
                self._job_prefix_len,
                aging_ms=int(getattr(self.config, "lpm_aging_ms", 0)),
            )
        taken: List[_Job] = []
        while self._queue:
            need = int(self._queue[0].n)
            if not can_admit_job(
                used_slots, used_reqs + len(taken), need, budget, soft_admit
            ):
                break
            taken.append(self._queue.pop(0))
            used_slots += need
        return taken

    def _drain_nowait(
        self,
        used_slots: int,
        used_reqs: int,
        budget: int,
        soft_admit: int,
    ) -> List[_Job]:
        with self._cv:
            if not self._queue:
                return []
            return self._pop_queue_locked(used_slots, used_reqs, budget, soft_admit)

    def _submit_jobs(self, jobs: List[_Job], job_by_rid: Dict[str, _Job]) -> None:
        try:
            for job in jobs:
                self._wait_job(job)
                if job.request is None:
                    continue
                job_by_rid[job.request.rid] = job
                self.engine.submit(job.request)
        except Exception as exc:
            for job in jobs:
                self._set_future(job, exc=exc)
                if job.request is not None:
                    job_by_rid.pop(job.request.rid, None)

    def _run_worker(self) -> None:
        if self.engine.device.type == "cuda":
            import torch

            torch.cuda.set_device(self.engine.device)
        try:
            self.engine.ensure_cuda_graph()
        except Exception:
            logger.exception("CUDA graph capture failed; continuing with eager decode")
        wait_s = max(int(self.config.batch_wait_ms), 0) / 1000.0
        wait_max_s = max(int(self.config.batch_wait_max_ms), 0) / 1000.0
        prefs = self.config.preferred_batch_sizes()
        target_admit = self.config.target_admit_reqs()
        soft_admit = self.config.soft_admit_max_reqs()
        high_cap = max(prefs[-1], target_admit)
        budget = self.config.resolved_batch_slots()
        recent_batch = 1
        job_by_rid: Dict[str, _Job] = {}

        def on_complete(req: BeamRequest, result: BeamResult) -> None:
            job = job_by_rid.pop(req.rid, None)
            if job is not None:
                self._set_future(job, result=result)

        self.engine._on_complete = on_complete

        def peek() -> None:
            extra = self._drain_nowait(
                used_slots=self.engine.inflight.used_slots()
                + self.engine.inflight.waiting_slots(),
                used_reqs=len(self.engine.inflight.waiting)
                + len(self.engine.inflight.running),
                budget=budget,
                soft_admit=soft_admit,
            )
            if extra:
                self._submit_jobs(extra, job_by_rid)

        while not self._stop:
            jobs: Optional[List[_Job]] = None
            with self._cv:
                while (
                    not self._queue
                    and not self.engine.has_work()
                    and not self._stop
                    and not self.profiler.has_work()
                ):
                    self._cv.wait()
                if self._stop:
                    return
                busy = self.engine.has_work()
                if self._queue:
                    used_slots = (
                        self.engine.inflight.used_slots()
                        + self.engine.inflight.waiting_slots()
                    )
                    used_reqs = len(self.engine.inflight.waiting) + len(
                        self.engine.inflight.running
                    )
                    # LPM (SGLang schedule-policy): longest shared prefix first.
                    jobs = self._pop_queue_locked(
                        used_slots, used_reqs, budget, soft_admit
                    )
                    if not jobs:
                        jobs = [self._queue.pop(0)]
                    slots = used_slots + sum(int(j.n) for j in jobs)
                    if not busy:
                        adapt_wait = batch_wait_seconds(
                            len(jobs),
                            slots - used_slots,
                            budget,
                            wait_s,
                            wait_max_s,
                            recent_batch,
                            target_reqs=target_admit,
                        )
                        underfilled = len(jobs) < target_admit and slots < budget
                        if underfilled and adapt_wait > 0:
                            deadline = time.time() + adapt_wait
                            extended = False
                            with trace_range(
                                f"flashrec.batch.wait ms={int(adapt_wait * 1000)} jobs={len(jobs)}"
                            ):
                                while (
                                    time.time() < deadline
                                    and len(jobs) < max(target_admit, 1)
                                    and slots < budget
                                ):
                                    remaining = deadline - time.time()
                                    if remaining <= 0:
                                        break
                                    if not self._queue:
                                        self._cv.wait(timeout=remaining)
                                        continue
                                    need = int(self._queue[0].n)
                                    if not can_admit_job(
                                        slots,
                                        used_reqs + len(jobs),
                                        need,
                                        budget,
                                        soft_admit,
                                    ):
                                        break
                                    jobs.append(self._queue.pop(0))
                                    slots += need
                                    # Second arrival means conc>1: pack to 8 (SGLang fill-before-expand).
                                    if (
                                        not extended
                                        and len(jobs) >= 2
                                        and wait_max_s > adapt_wait
                                    ):
                                        deadline = time.time() + wait_max_s
                                        extended = True
                        k = pack_release_count(
                            len(jobs), recent_batch, target_admit, high_cap
                        )
                        if k < len(jobs):
                            leftover = jobs[k:]
                            jobs = jobs[:k]
                            self._queue[0:0] = leftover
            self.profiler.poll()
            if jobs:
                recent_batch = max(len(jobs), (recent_batch + len(jobs)) // 2, 1)
                self._submit_jobs(jobs, job_by_rid)
            if self.engine.has_work():
                try:
                    self.engine.run_burst(peek)
                except Exception as exc:
                    logger.exception("inflight step failed")
                    stuck = list(job_by_rid.values())
                    job_by_rid.clear()
                    self.engine.inflight.waiting.clear()
                    self.engine.inflight.running.clear()
                    for job in stuck:
                        self._set_future(job, exc=exc)
            self.profiler.poll()

    @staticmethod
    def _set_future(
        job: _Job,
        result: Optional[BeamResult] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        def _apply() -> None:
            if job.future.done():
                return
            if exc is not None:
                job.future.set_exception(exc)
            else:
                job.future.set_result(result)

        job.future.get_loop().call_soon_threadsafe(_apply)

    def _to_openai(
        self, result: BeamResult, request: ChatCompletionRequest
    ) -> Dict[str, Any]:
        created = int(time.time())
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        choices = []
        for i, seq in enumerate(result.sequences):
            finish = (
                "stop" if (seq.finish_reason or {}).get("type") == "stop" else "length"
            )
            choices.append(
                {
                    "index": i,
                    "message": {"role": "assistant", "content": seq.text},
                    "finish_reason": finish,
                    "sglext": {"sequence_score": _plain_score(seq.beam_score)},
                }
            )
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": request.model or self.config.model_path,
            "choices": choices,
            "usage": _usage_dict(result),
        }

    def _stream_openai_bytes(
        self, result: BeamResult, request: ChatCompletionRequest
    ) -> List[bytes]:
        """Compact SSE: one content chunk (all beams) + usage + DONE.

        Avoids ~2N per-beam events that inflate bytes ~3× and stall the
        asyncio loop under EvalScope ``stream=True``.
        """
        created = int(time.time())
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        model = request.model or self.config.model_path
        choices = []
        for i, seq in enumerate(result.sequences):
            finish = (
                "stop" if (seq.finish_reason or {}).get("type") == "stop" else "length"
            )
            choices.append(
                {
                    "index": i,
                    "delta": {"role": "assistant", "content": seq.text},
                    "finish_reason": finish,
                    "sglext": {"sequence_score": _plain_score(seq.beam_score)},
                }
            )

        def _sse(payload: Dict[str, Any]) -> bytes:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode(
                "utf-8"
            )

        return [
            _sse(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": choices,
                }
            ),
            _sse(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": _usage_dict(result),
                }
            ),
            b"data: [DONE]\n\n",
        ]


def _install_gc_probe() -> None:
    import gc

    state: dict = {}

    def _cb(phase: str, info: dict) -> None:
        if phase == "start":
            state["t0"] = time.perf_counter()
            return
        t0 = state.pop("t0", None)
        if t0 is None:
            return
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if dt_ms >= 5.0:
            logger.warning(
                "GC pause %.1fms gen=%s collected=%s uncollectable=%s counts=%s",
                dt_ms,
                info.get("generation"),
                info.get("collected"),
                info.get("uncollectable"),
                gc.get_count(),
            )

    gc.callbacks.append(_cb)


def _tune_gc_after_startup() -> None:
    """Keep startup-resident objects out of every future GC scan.

    The valid-SID map and tokenizer tables are millions of permanently live
    objects. Without freeze() each gen-2 pass retraces all of them and stalls
    every in-flight request for ~200ms while collecting nothing.
    """
    import gc

    gc.collect()
    gc.collect()
    gc.freeze()
    gc.set_threshold(20_000, 50, 100)
    logger.info(
        "GC tuned: frozen=%d threshold=%s", gc.get_freeze_count(), gc.get_threshold()
    )


def serve(config: BeamRecConfig) -> None:
    _install_gc_probe()
    server = BeamRecServer(config)
    _tune_gc_after_startup()
    uvicorn.run(
        server.app,
        host=config.host,
        port=int(config.port),
        log_level=config.log_level,
    )
