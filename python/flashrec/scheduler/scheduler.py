"""GenRec beam scheduler: radix → prefill (batched) → expand n rows → decode pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from flashrec.config import BeamRecConfig
from flashrec.core import (
    BeamRequest,
    BeamResult,
    BeamSequence,
    FinishReason,
    ForwardBatch,
)
from flashrec.engine.engine import ModelEngine
from flashrec.engine.graph import ExpandCaptureSpec
from flashrec.engine.staging import PinnedStage
from flashrec.hostpool import HostPool
from flashrec.kernel.beam_trie import (
    GenrecFusedResult,
    GenrecFusedWorkspace,
    genrec_mask_topk_expand,
    pick_pingpong_stack,
    rows_alias_stack,
    try_load_beam_trie,
)
from flashrec.profiler import trace_range
from flashrec.scheduler.batching import group_by_beam_depth
from flashrec.scheduler.kv_remap import remap_by_parents
from flashrec.scheduler.loop import InflightLoop
from flashrec.scheduler.pipeline import DecodePipeline
from flashrec.scheduler.warmup import warmup_shared_prefix
from flashrec.search.expand import (
    apply_temperature,
    expand_step,
    init_from_prefill,
    init_from_prefill_batch,
)
from flashrec.search.score import beam_score
from flashrec.search.trie import build_beam_valid_path
from flashrec.tokenizer import TokenizerAdapter

logger = logging.getLogger(__name__)

# Kill switch for the numpy finalize fast path (rollout safety / A-B benching).
_DENSE_FINALIZE = os.getenv("FLASHREC_DENSE_FINALIZE", "1") != "0"


@dataclass
class _DecodeLaunch:
    reqs: List[BeamRequest]
    logprobs: Optional[torch.Tensor]
    cand_ids: Optional[torch.Tensor]
    pipe_id: Optional[int] = None
    replay_done: Optional[torch.cuda.Event] = None
    fused: Optional[GenrecFusedResult] = None


class BeamRecEngine:
    def __init__(self, config: BeamRecConfig):
        self.config = config
        self.runner = ModelEngine(config)
        self.device = self.runner.device
        self.tokenizer = TokenizerAdapter(config.model_path)
        self.host_pool = HostPool(int(getattr(config, "host_worker_threads", 4) or 0))
        self.prefix_cache = self.runner.prefix_cache
        special = config.parsed_sid_token_ids()
        self.valid_path = build_beam_valid_path(
            sid_file=config.sid_vocab_file,
            codebook_sizes=config.parsed_codebook_sizes(),
            special_token_ids=special,
        )
        self.boundary_ids = config.parsed_boundary_token_ids()
        self.system_prompt = config.system_prompt_text()
        self.pipeline = DecodePipeline(
            stages=max(config.pipeline_stages, 1), enabled=config.enable_pipeline
        )
        self._fused_ws = GenrecFusedWorkspace()
        self._fused_ws_pipe: Dict[Optional[int], GenrecFusedWorkspace] = {}
        self._expand_pipe_id: Optional[int] = None
        self._stage = PinnedStage()
        self._on_complete: Optional[Callable] = None
        self._defer_remap = False
        self._pending_remaps: List[Tuple[BeamRequest, torch.Tensor]] = []
        self._last_prefill_ev = None
        self._wave_pool_rows: Optional[torch.Tensor] = None
        self._wave_seq_base: Optional[torch.Tensor] = None
        self._wave_seq_base_sum: int = 0
        self._wave_reqs_key: Optional[tuple] = None
        self._wave_n_rows: int = 0
        self._wave_step: int = 0
        depth = int(getattr(self.valid_path, "max_depth", 0) or 0)
        prefs = config.preferred_batch_sizes()
        self.inflight = InflightLoop(
            slots=config.resolved_batch_slots(),
            preferred=prefs,
            inflight_min=config.target_admit_reqs(),
            short_genrec=0 < depth <= 3,
            prefill_fn=self._prefill_for_loop,
            decode_step_fn=self._decode_tick,
            complete_fn=self._complete_reqs,
            free_fn=self._free_reqs,
            pack_min=int(getattr(config, "decode_pack_min_requests", 6) or 6),
            pack_ratio=float(getattr(config, "decode_pack_ratio", 0.75) or 0.75),
        )
        if self.device.type == "cuda":
            self.valid_path._ensure_gpu_cache(self.device)
            if try_load_beam_trie():
                logger.info("beam_trie JIT loaded")
            else:
                logger.info("beam_trie JIT fallback (PyTorch)")
        if config.enable_warmup:
            n_warm = warmup_shared_prefix(self)
            if n_warm:
                logger.info("radix warmup locked %d shared prefix tokens", n_warm)
        self.profiler = None
        logger.info(
            "FlashRec ready: depth=%s radix=%s graph=%s pipeline=%s",
            getattr(self.valid_path, "max_depth", None),
            bool(self.prefix_cache),
            self.runner.graph is not None,
            self.pipeline.enabled,
        )
        self.pipeline.ensure_streams()

    def attach_profiler(self, profiler) -> None:
        self.profiler = profiler
        self.runner.profiler = profiler

    def ensure_cuda_graph(self) -> None:
        spec = self._expand_capture_spec()
        self.runner.ensure_cuda_graph(expand_spec=spec)
        if self.runner.graph is not None:
            # Replay stays on the capture stream; expand overlaps on expand_stream.
            self.pipeline.ensure_streams()

    def _expand_capture_spec(self) -> Optional[ExpandCaptureSpec]:
        if not self.config.enable_graph_expand or not self.config.enable_fused_expand:
            return None
        vp = self.valid_path
        if not vp.active or vp.mode != "trie":
            return None
        if not try_load_beam_trie():
            return None
        try:
            self.runner.lm_head.bind(self.runner.model.lm_head())
        except Exception:
            return None
        if not self.runner.lm_head.ready or not self.runner.lm_head.num_tokens:
            return None
        vp._ensure_gpu_cache(self.device)
        if vp._allow_table is None or vp._next_node is None:
            return None
        bw = int(self.config.beam_width)
        k = int(self.runner.lm_head.num_tokens)
        if bw <= 0 or k <= 0:
            return None
        allow = vp._allow_table
        if allow.dtype == torch.bool:
            allow = allow.view(torch.uint8)
        nxt = vp._next_node
        if nxt.dtype != torch.int64:
            nxt = nxt.to(dtype=torch.int64)
        return ExpandCaptureSpec(
            beam_width=bw,
            cand=min(max(bw * 2, bw), k),
            select_k=bw,
            width=int(self.config.max_tokens),
            cand_ids=self.runner.lm_head.token_ids.contiguous(),
            allow_table=allow.contiguous(),
            next_node=nxt.contiguous(),
            token_base=int(vp.token_base or 0),
            invalid_node=int(vp._invalid_node),
        )

    def _can_graph_expand(self, reqs: List[BeamRequest], n_rows: int) -> bool:
        graph = self.runner.graph
        if graph is None or not graph.can_replay_expand(n_rows):
            return False
        spec = graph.expand_spec
        bw = int(graph.expand_bw or 0)
        if spec is None or bw <= 0 or int(n_rows) % bw != 0:
            return False
        if len(reqs) != int(n_rows) // bw:
            return False
        width = int(spec.width)
        for req in reqs:
            if not req.ignore_eos or int(req.beam_width) != bw:
                return False
            # Graph-captured fused select is deterministic top-k only.
            if req.temperature > 0.0:
                return False
            bl = req.beam_list
            if bl is None or bl.token_ids is None or bl.cum_logprobs is None:
                return False
            if bl.node_ids is None:
                return False
            if int(bl.token_ids.shape[0]) != bw or int(bl.token_ids.shape[-1]) != width:
                return False
        return True

    def _iter_other_reqs(self, skip_reqs: List[BeamRequest]):
        skip = {id(r) for r in skip_reqs}
        pipes = self.pipeline.pipes
        if not pipes:
            return
        for pipe in pipes:
            for r in pipe:
                if id(r) not in skip and not r.finished:
                    yield r

    def _detach_plane_aliases(
        self, plane: torch.Tensor, skip_reqs: List[BeamRequest], field: str
    ) -> None:
        """Clone live views that alias ``plane`` so the next graph write is safe."""
        target = plane.untyped_storage().data_ptr()
        for req in self._iter_other_reqs(skip_reqs):
            bl = req.beam_list
            if bl is None:
                continue
            live = getattr(bl, field, None)
            if live is None or not isinstance(live, torch.Tensor):
                continue
            if live.untyped_storage().data_ptr() == target:
                setattr(bl, field, live.clone())

    def _pack_graph_expand(self, reqs: List[BeamRequest], buf: dict) -> None:
        with trace_range(f"flashrec.pack_expand reqs={len(reqs)}"):
            n = len(reqs)
            n_pad = int(buf["exp_n"])
            device = self.device
            torch.stack(
                [
                    r.beam_list.cum_logprobs.to(device=device, dtype=torch.float32)
                    for r in reqs
                ],
                out=buf["exp_cum"][:n],
            )
            graph = self.runner.graph
            if graph is not None:
                tok_src, tok_dst, node_src, node_dst = graph.expand_io(buf, n)
            else:
                tok_src, tok_dst = buf["exp_tok_in"][:n], buf["exp_tok_out"]
                node_src, node_dst = buf["exp_nodes"][:n], buf["exp_node_out"]
            self._detach_plane_aliases(tok_dst, reqs, "token_ids")
            self._detach_plane_aliases(node_dst, reqs, "node_ids")
            tok_rows = [
                r.beam_list.token_ids.to(device=device, dtype=torch.int64) for r in reqs
            ]
            if not rows_alias_stack(tok_rows, tok_src):
                torch.stack(tok_rows, out=tok_src)
            node_rows = [
                r.beam_list.node_ids.to(device=device, dtype=torch.int64) for r in reqs
            ]
            if not rows_alias_stack(node_rows, node_src):
                torch.stack(node_rows, out=node_src)
            cols = [int(r.beam_list.cur_len) for r in reqs]
            self._stage.copy_list(
                "exp_col", cols, device, torch.int32, dest=buf["exp_col"][:n]
            )
            buf["exp_do"][:n].fill_(1)
            if n_pad > n:
                buf["exp_do"][n:n_pad].zero_()
                buf["exp_col"][n:n_pad].zero_()

    def _apply_fused(self, reqs: List[BeamRequest], fused: GenrecFusedResult) -> None:
        for i, req in enumerate(reqs):
            col = int(req.beam_list.cur_len)
            req.beam_list.cum_logprobs = fused.vals[i]
            req.beam_list.last_tokens = fused.tokens[i]
            req.beam_list.token_ids = fused.token_ids[i]
            req.beam_list.cur_len = col + 1
            req.beam_list.dense_authoritative = True
            if fused.node_ids is not None:
                req.beam_list.node_ids = fused.node_ids[i]
            limit = self.generation_token_limit(req.max_new_tokens)
            will_finish = req.beam_list.generated_len() >= limit
            parents = fused.parents[i]
            if will_finish:
                req.finished = True
                req.finish_reason = FinishReason(
                    type="length", length=req.max_new_tokens
                )
            else:
                self._remap_req(req, parents)
            req.expanded = True

    def _cuda_stream(self):
        if torch.cuda.is_available():
            return torch.cuda.current_stream()
        return None

    def _wait_event(self, ev, stream=None) -> None:
        if ev is None:
            return
        if torch.cuda.is_available():
            (stream or torch.cuda.current_stream()).wait_event(ev)

    def _wait_pipe_ready(self, pipe_id: Optional[int], *, pop: bool = True) -> None:
        """Order this pipe's next forward after its expand/remap. Do not wait others."""
        stream = self._cuda_stream()
        for store in (self.pipeline.expand_done, self.pipeline.remap_done):
            ev = store.pop(pipe_id, None) if pop else store.get(pipe_id)
            self._wait_event(ev, stream)

    def _wait_reqs_expand(self, reqs: List[BeamRequest]) -> None:
        self._wait_reqs_expand_on(self._cuda_stream(), reqs)

    def _wait_reqs_expand_on(self, stream, reqs: List[BeamRequest]) -> None:
        if not reqs:
            return
        if not self.pipeline.active:
            self._wait_event(self.pipeline.expand_done.get(None), stream)
            self._wait_event(self.pipeline.remap_done.get(None), stream)
            return
        ids = {id(r) for r in reqs}
        for pid, pipe in enumerate(self.pipeline.pipes or []):
            if any(id(r) in ids for r in pipe):
                self._wait_event(self.pipeline.expand_done.get(pid), stream)
                self._wait_event(self.pipeline.remap_done.get(pid), stream)

    def _prefill_overlap(self) -> bool:
        return bool(
            self.pipeline.active
            and self.pipeline.prefill_stream is not None
            and getattr(self.runner.attn, "prefill_overlap_ok", False)
            and torch.cuda.is_available()
        )

    def _d2h_tensors(
        self,
        tensors: List[Optional[torch.Tensor]],
        wait_reqs: Optional[List[BeamRequest]] = None,
    ) -> List[Optional[torch.Tensor]]:
        """Copy GPU tensors to pinned host without draining the default stream.

        ``Tensor.cpu()`` on the default stream waits for in-flight 3-pipe
        decode. DMA on ``copy_stream`` + ``Event.synchronize`` only waits
        expand of the finished reqs (SGLang ``beam_copy_stream``).
        """
        out: List[Optional[torch.Tensor]] = []
        pending: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for t in tensors:
            if t is None:
                out.append(None)
                continue
            t = t.detach()
            if t.device.type != "cuda":
                out.append(t)
                continue
            host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            out.append(host)
            pending.append((host, t))
        if not pending:
            return out
        copy_stream = self.pipeline.copy_stream
        if copy_stream is None or not torch.cuda.is_available():
            for host, src in pending:
                host.copy_(src)
            return out
        if wait_reqs:
            self._wait_reqs_expand_on(copy_stream, wait_reqs)
        expand = self.pipeline.expand_stream
        if expand is not None:
            copy_stream.wait_stream(expand)
        with torch.cuda.stream(copy_stream):
            for host, src in pending:
                host.copy_(src, non_blocking=True)
            done = torch.cuda.Event()
            done.record(copy_stream)
        done.synchronize()
        return out

    def _wait_all_pipeline_events(self) -> None:
        stream = self._cuda_stream()
        for store in (self.pipeline.expand_done, self.pipeline.remap_done):
            for ev in list(store.values()):
                self._wait_event(ev, stream)
            store.clear()

    def generation_token_limit(self, max_new_tokens: int) -> int:
        requested = int(max_new_tokens)
        depth = int(getattr(self.valid_path, "max_depth", 0) or 0)
        if self.boundary_ids is not None and depth > 0 and requested >= depth + 2:
            return depth
        return requested

    def tokenize_chat(
        self,
        messages: List[Dict[str, Any]],
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        msgs = list(messages)
        if self.system_prompt and not any(m.get("role") == "system" for m in msgs):
            msgs = [{"role": "system", "content": self.system_prompt}] + msgs
        ids = self.tokenizer.apply_chat(msgs, chat_template_kwargs)
        return self._maybe_append_begin(ids)

    def tokenize_text(self, prompt: str) -> List[int]:
        return self._maybe_append_begin(self.tokenizer.encode(prompt))

    def _maybe_append_begin(self, input_ids: List[int]) -> List[int]:
        if not self.boundary_ids:
            return input_ids
        begin = int(self.boundary_ids[0])
        depth = int(getattr(self.valid_path, "max_depth", 0) or 0)
        if depth <= 0:
            return input_ids
        out = list(input_ids)
        if not out or out[-1] != begin:
            out.append(begin)
        return out

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        input_ids: Optional[Sequence[int]] = None,
        n: Optional[int] = None,
        max_tokens: Optional[int] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BeamResult:
        return self.generate_many(
            [
                self.make_request(
                    prompt=prompt,
                    messages=messages,
                    input_ids=input_ids,
                    n=n,
                    max_tokens=max_tokens,
                    chat_template_kwargs=chat_template_kwargs,
                )
            ]
        )[0]

    def submit(self, req: BeamRequest) -> None:
        self.inflight.submit(req)

    def has_work(self) -> bool:
        return self.inflight.has_work()

    def step(self) -> None:
        self.inflight.step()

    def run_burst(self, peek: Optional[Callable] = None) -> None:
        self.inflight.run_burst(peek)

    def flush_host(self) -> None:
        self.host_pool.flush()

    def generate_many(
        self,
        requests: List[BeamRequest],
        pull_more: Optional[Callable[..., List[BeamRequest]]] = None,
        fill_wait_s: float = 0.0,
    ) -> List[BeamResult]:
        if not requests:
            return []
        if self.profiler is not None:
            self.profiler.poll()
        self.ensure_cuda_graph()
        collected: Dict[str, BeamResult] = {}
        order: List[str] = []
        prev = self._on_complete

        def on_complete(req: BeamRequest, result: BeamResult) -> None:
            collected[req.rid] = result
            if prev is not None:
                prev(req, result)

        self._on_complete = on_complete
        try:
            with trace_range(f"flashrec.generate_many n={len(requests)}"):
                for req in requests:
                    self.submit(req)
                    order.append(req.rid)
                extra = []
                if pull_more is not None:
                    extra = list(
                        pull_more(
                            used_slots=self.inflight.used_slots()
                            + self.inflight.waiting_slots(),
                            used_reqs=len(self.inflight.waiting)
                            + len(self.inflight.running),
                            wait_s=float(fill_wait_s),
                        )
                        or []
                    )
                for req in extra:
                    self.submit(req)
                    order.append(req.rid)

                def peek() -> None:
                    if pull_more is None:
                        return
                    more = list(
                        pull_more(
                            used_slots=self.inflight.used_slots()
                            + self.inflight.waiting_slots(),
                            used_reqs=len(self.inflight.waiting)
                            + len(self.inflight.running),
                            wait_s=0.0,
                        )
                        or []
                    )
                    for req in more:
                        self.submit(req)
                        order.append(req.rid)

                self.run_burst(peek if pull_more is not None else None)
        finally:
            self.flush_host()
            self._on_complete = prev
        if self.profiler is not None:
            self.profiler.poll()
        return [collected[rid] for rid in order if rid in collected]

    def make_request(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        input_ids: Optional[Sequence[int]] = None,
        n: Optional[int] = None,
        max_tokens: Optional[int] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> BeamRequest:
        if input_ids is not None:
            ids = self._maybe_append_begin([int(x) for x in input_ids])
        elif messages is not None:
            ids = self.tokenize_chat(messages, chat_template_kwargs)
        elif prompt is not None:
            ids = self.tokenize_text(prompt)
        else:
            raise ValueError("provide prompt, messages, or input_ids")
        return BeamRequest(
            input_ids=ids,
            beam_width=int(n if n is not None else self.config.beam_width),
            max_new_tokens=int(
                max_tokens if max_tokens is not None else self.config.max_tokens
            ),
            messages=messages,
            ignore_eos=True,
            length_penalty=float(self.config.length_penalty),
            temperature=max(float(temperature or 0.0), 0.0),
        )

    def _prefill_for_loop(self, reqs: List[BeamRequest]) -> None:
        if not reqs:
            return

        def _run() -> None:
            if self.config.enable_prefill_batch:
                self._prefill_batch(reqs)
            else:
                for req in reqs:
                    self._prefill_batch([req])

        if not self._prefill_overlap():
            _run()
            return
        ps = self.pipeline.prefill_stream
        if self._last_prefill_ev is not None:
            ps.wait_event(self._last_prefill_ev)
        for store in (self.pipeline.expand_done, self.pipeline.remap_done):
            for ev in store.values():
                if ev is not None:
                    ps.wait_event(ev)
        with torch.cuda.stream(ps):
            _run()
        done = torch.cuda.Event()
        done.record(ps)
        self._last_prefill_ev = done
        for req in reqs:
            req.prefill_done = done

    def _free_reqs(self, reqs: List[BeamRequest]) -> None:
        if not reqs:
            return
        if self._prefill_overlap():
            self._wait_reqs_expand_on(self.pipeline.prefill_stream, reqs)
        else:
            self._wait_reqs_expand(reqs)
        for req in reqs:
            self._free_request(req)

    def _complete_reqs(self, reqs: List[BeamRequest]) -> None:
        if not reqs:
            return
        need_free = [
            r
            for r in reqs
            if r.prefill_pool_idx >= 0 or r.beam_pool_indices is not None
        ]
        if need_free:
            self._free_reqs(need_free)
        snap = self._snapshot_finalize(reqs)
        on_complete = self._on_complete

        def emit() -> None:
            results = self._emit_finalize(snap)
            if on_complete is None:
                return
            for req, result in zip(reqs, results):
                on_complete(req, result)

        # Inline emit so HTTP returns this tick. Host-pool detokenize delayed
        # EvalScope replacements until the rest of the 8-wide wave had finished.
        emit()

    def _match_prefix(self, ids: List[int]) -> Tuple[int, List[int]]:
        if self.prefix_cache is None or len(ids) <= 1:
            return 0, []
        hit_len, prefix_kv = self.prefix_cache.match_cpu(ids[:-1])
        return int(hit_len), prefix_kv

    def fill_prefix_kv(self, ids: List[int]) -> int:
        """Prefill ``ids`` into the radix cache and lock them (warmup pin)."""
        if self.prefix_cache is None or not ids:
            return 0
        n = len(ids)
        suffix_kv, suffix_cpu = self.runner.alloc_tokens_cpu(n)
        req_idx = self.runner.alloc_req_one()
        full_kv = torch.tensor(suffix_cpu, dtype=torch.int64, device=self.device)
        self.runner.write_prefill(req_idx, full_kv)
        table = self.runner.req_pool.req_to_token
        batch = ForwardBatch(
            input_ids=torch.tensor(ids, dtype=torch.int64, device=self.device),
            req_pool_indices=torch.tensor(
                [req_idx], dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.tensor([n], dtype=torch.int64, device=self.device),
            seq_lens_cpu=torch.tensor([n], dtype=torch.int64),
            positions=torch.arange(n, dtype=torch.int64, device=self.device),
            out_cache_loc=suffix_kv,
            is_prefill=True,
            extend_prefix_lens=[0],
            extend_seq_lens=[n],
            kv_indices=table[req_idx, :n].to(dtype=torch.int32),
        )
        self.runner.forward(batch)
        canon = self.prefix_cache.insert_cpu(ids, suffix_cpu)
        dups = [a for a, b in zip(suffix_cpu, canon) if a != b]
        if dups:
            self.runner.free_tokens(dups)
        self.prefix_cache.lock(ids)
        self.runner.free_req_slots([req_idx])
        return n

    def _prefill_batch(self, reqs: List[BeamRequest]) -> None:
        with trace_range(f"flashrec.prefill reqs={len(reqs)}"):
            prepared = []
            table = self.runner.req_pool.req_to_token
            kv_writes = []
            for req in reqs:
                ids = list(req.input_ids)
                prompt_len = len(ids)
                if prompt_len <= 0:
                    raise ValueError("empty prompt")
                hit_len, prefix_list = self._match_prefix(ids)
                req.cached_tokens = hit_len
                req.prompt_len = prompt_len
                req.seq_len = prompt_len
                suffix_ids = ids[hit_len:]
                suffix_kv, suffix_cpu = self.runner.alloc_tokens_cpu(len(suffix_ids))
                req_idx = self.runner.alloc_req_one()
                req.prefill_pool_idx = req_idx
                full_cpu = prefix_list + suffix_cpu
                kv_writes.append((full_cpu, table[req_idx, : len(full_cpu)]))
                prepared.append(
                    (req, ids, hit_len, suffix_ids, suffix_kv, None, full_cpu)
                )
            # One pinned H2D for every req's row instead of a pageable
            # torch.tensor(..., device=cuda) per req (host-blocking).
            self._stage.copy_lists(
                "prefill_full_kv", kv_writes, self.device, table.dtype
            )

            suffix_ids_cat = []
            suffix_kv_cat = []
            rows = []
            ext = []
            pfx = []
            pos_host = []
            seq_host = []
            for req, ids, hl, suffix_ids, suffix_kv, full_kv, full_cpu in prepared:
                suffix_ids_cat.extend(suffix_ids)
                suffix_kv_cat.append(suffix_kv)
                rows.append(req.prefill_pool_idx)
                e = len(suffix_ids)
                ext.append(e)
                pfx.append(hl)
                pos_host.extend(range(hl, hl + e))
                seq_host.append(req.prompt_len)
            ids_gpu, _ = self._stage.copy_list(
                "prefill_ids", suffix_ids_cat, self.device, torch.int64
            )
            rows_gpu, _ = self._stage.copy_list(
                "prefill_rows", rows, self.device, torch.int64
            )
            seq_gpu, seq_pin = self._stage.copy_list(
                "prefill_seq", seq_host, self.device, torch.int64
            )
            # One pinned H2D for positions instead of a GPU arange per request,
            # and one triton gather for kv_indices instead of a per-request
            # table slice + cast + cat.
            pos_gpu, _ = self._stage.copy_list(
                "prefill_pos", pos_host, self.device, torch.int64
            )
            kv_gpu = self.runner.gather_kv_indices(
                rows_gpu, seq_host, total=sum(seq_host)
            )
            batch = ForwardBatch(
                input_ids=ids_gpu,
                req_pool_indices=rows_gpu,
                seq_lens=seq_gpu,
                seq_lens_cpu=seq_pin,
                positions=pos_gpu,
                out_cache_loc=torch.cat(suffix_kv_cat),
                is_prefill=True,
                extend_prefix_lens=pfx,
                extend_seq_lens=ext,
                kv_indices=kv_gpu,
            )
            logprobs, cand_ids, _ = self.runner.forward(batch)
            precomputed = self._init_wave_precompute(prepared, logprobs, cand_ids)
            for i, (
                req,
                ids,
                hl,
                suffix_ids,
                suffix_kv,
                full_kv,
                full_cpu,
            ) in enumerate(prepared):
                self._init_after_prefill(
                    req,
                    ids,
                    full_kv,
                    full_cpu,
                    logprobs[i],
                    cand_ids,
                    hit_len=hl,
                    suffix_cpu=full_cpu[hl:],
                    prefix_list=full_cpu[:hl],
                    precomputed=precomputed[i] if precomputed is not None else None,
                )

    def _collect_completed(self, req: BeamRequest):
        completed = list(req.beam_list.completed)
        if not completed and req.beam_list.token_ids is not None:
            completed = req.beam_list.sequences_from_token_ids(
                req.beam_list.cum_logprobs,
                length_penalty=req.length_penalty,
                finish_reason=req.finish_reason,
            )
        completed.sort(key=lambda s: float(s.beam_score or 0.0), reverse=True)
        return completed[: req.beam_width]

    def _snapshot_finalize_dense(self, req: BeamRequest) -> Optional[dict]:
        """Array fast path for the common GenRec finalize: no per-request
        ``completed`` stubs, dense CPU ``token_ids`` of uniform length. Scores,
        ordering, packing (+boundary) and detokenize all run as one numpy pass
        instead of building three Python objects per beam — at beam=512 this
        was ~50ms of GPU-idle scheduler time per 8-wide wave."""
        bl = req.beam_list
        if (
            not _DENSE_FINALIZE
            or bl is None
            or bl.completed
            or bl.token_ids is None
            or bl.cur_len <= 0
            or bl.token_ids.device.type != "cpu"
        ):
            return None
        cur_len = int(bl.cur_len)
        ids = bl.token_ids[:, :cur_len].numpy()
        n = int(ids.shape[0])
        if n == 0:
            return None
        if bl.cum_logprobs is not None:
            lp_t = bl.cum_logprobs
            if lp_t.device.type != "cpu" or int(lp_t.shape[0]) != n:
                return None
            lp = lp_t.numpy().astype(np.float64, copy=False)
        else:
            lp = np.zeros(n, dtype=np.float64)
        scores = lp / (float(cur_len) ** float(req.length_penalty))
        k = min(int(req.beam_width), n)
        order = np.argsort(-scores, kind="stable")[:k]
        ids = ids[order]
        if self.boundary_ids is not None:
            mat = np.empty((k, cur_len + 2), dtype=np.int64)
            mat[:, 0] = int(self.boundary_ids[0])
            mat[:, 1:-1] = ids
            mat[:, -1] = int(self.boundary_ids[1])
        else:
            mat = ids
        finish = req.finish_reason
        if hasattr(finish, "to_json"):
            finish = finish.to_json()
        elif not isinstance(finish, dict):
            finish = {"type": "length"}
        lp_list = lp[order].tolist()
        score_list = scores[order].tolist()
        snap = {
            "prompt_tokens": int(req.prompt_len),
            "cached_tokens": int(req.cached_tokens),
            "packed": mat.tolist(),
            "scores": [(a, b, finish) for a, b in zip(lp_list, score_list)],
        }
        texts = self.tokenizer.decode_matrix(mat)
        if texts is not None:
            snap["texts"] = texts
        return snap

    def _snapshot_finalize(self, reqs: List[BeamRequest]) -> List[dict]:
        """Pull token ids onto CPU on copy_stream before the next prefill."""
        gpu_reqs = [
            r
            for r in reqs
            if r.beam_list is not None
            and not r.beam_list.completed
            and r.beam_list.token_ids is not None
            and getattr(r.beam_list.token_ids, "device", None) is not None
            and r.beam_list.token_ids.device.type == "cuda"
        ]
        if gpu_reqs:
            tensors: List[Optional[torch.Tensor]] = []
            for req in gpu_reqs:
                bl = req.beam_list
                tensors.append(bl.token_ids[:, : bl.cur_len])
                tensors.append(bl.cum_logprobs)
            copied = self._d2h_tensors(tensors, wait_reqs=gpu_reqs)
            idx = 0
            for req in gpu_reqs:
                req.beam_list.token_ids = copied[idx]
                if copied[idx + 1] is not None:
                    req.beam_list.cum_logprobs = copied[idx + 1]
                idx += 2
        snaps: List[dict] = []
        for req in reqs:
            snap = self._snapshot_finalize_dense(req)
            if snap is not None:
                snaps.append(snap)
                continue
            completed = self._collect_completed(req)
            packed: List[List[int]] = []
            scores: List[tuple] = []
            for seq in completed:
                tokens = list(seq.tokens)
                if self.boundary_ids is not None:
                    tokens = [
                        int(self.boundary_ids[0]),
                        *tokens,
                        int(self.boundary_ids[1]),
                    ]
                packed.append(tokens)
                score = seq.beam_score
                if score is None:
                    score = beam_score(
                        float(seq.cum_logprob),
                        max(len(seq.tokens), 1),
                        req.length_penalty,
                    )
                finish = seq.finish_reason
                if hasattr(finish, "to_json"):
                    finish = finish.to_json()
                elif not isinstance(finish, dict):
                    finish = {"type": "length"}
                scores.append((float(seq.cum_logprob), float(score), finish))
            snaps.append(
                {
                    "prompt_tokens": int(req.prompt_len),
                    "cached_tokens": int(req.cached_tokens),
                    "packed": packed,
                    "scores": scores,
                }
            )
        return snaps

    def _emit_finalize(self, snaps: List[dict]) -> List[BeamResult]:
        packed: List[List[int]] = []
        spans: List[Optional[tuple]] = []
        for snap in snaps:
            if "texts" in snap:
                spans.append(None)
                continue
            start = len(packed)
            packed.extend(snap["packed"])
            spans.append((start, len(packed)))
        with trace_range(f"flashrec.finalize reqs={len(snaps)}"):
            decoded = self.tokenizer.batch_decode(packed) if packed else []
        out: List[BeamResult] = []
        for snap, span in zip(snaps, spans):
            texts = snap["texts"] if span is None else decoded[span[0] : span[1]]
            # positional args: kwargs init is measurably slower at 4096 beams/wave
            sequences: List[BeamSequence] = [
                BeamSequence(tokens, lp, score, text, finish)
                for tokens, (lp, score, finish), text in zip(
                    snap["packed"], snap["scores"], texts
                )
            ]
            top = sequences[0] if sequences else BeamSequence(tokens=[], text="")
            out.append(
                BeamResult(
                    text=top.text,
                    output_ids=list(top.tokens),
                    sequences=sequences,
                    prompt_tokens=int(snap["prompt_tokens"]),
                    completion_tokens=sum(len(s.tokens) for s in sequences),
                    cached_tokens=int(snap["cached_tokens"]),
                )
            )
        return out

    def _init_wave_precompute(
        self,
        prepared: List[tuple],
        logprobs: torch.Tensor,
        cand_ids: Optional[torch.Tensor],
    ) -> Optional[List[tuple]]:
        """Batch the per-request init work of a homogeneous prefill wave.

        One topk/mask/select pass + one beam-row alloc + one table copy for
        the whole wave. Returns per-request ``(beam_list, rows, rows_cpu)``
        or None when requests are heterogeneous (caller falls back to the
        per-request path in ``_init_after_prefill``).
        """
        if not prepared or not isinstance(logprobs, torch.Tensor):
            return None
        if logprobs.dim() != 2 or int(logprobs.shape[0]) != len(prepared):
            return None
        reqs = [p[0] for p in prepared]
        bw = int(reqs[0].beam_width)
        cand = int(reqs[0].beam_candidates)
        temp = float(reqs[0].temperature)
        limits = [self.generation_token_limit(r.max_new_tokens) for r in reqs]
        max_new = max(limits[0], int(reqs[0].max_new_tokens))
        for r, limit in zip(reqs, limits):
            if (
                int(r.beam_width) != bw
                or int(r.beam_candidates) != cand
                or float(r.temperature) != temp
                or max(limit, int(r.max_new_tokens)) != max_new
                or limit <= 1
            ):
                return None
        bls = init_from_prefill_batch(
            logprobs,
            beam_width=bw,
            beam_candidates=cand,
            max_new_tokens=max_new,
            prompt_lens=[int(r.prompt_len) for r in reqs],
            valid_path=self.valid_path,
            candidate_token_ids=(
                cand_ids if isinstance(cand_ids, torch.Tensor) else None
            ),
            temperature=temp,
        )
        n = len(reqs)
        all_rows = self.runner.alloc_req_slots(n * bw)
        all_rows_cpu = list(self.runner.req_pool.last_alloc_cpu)
        self.runner.copy_prefill_to_beams_many(
            [int(r.prefill_pool_idx) for r in reqs],
            all_rows.view(n, bw),
            [int(r.prompt_len) for r in reqs],
        )
        return [
            (
                bls[i],
                all_rows[i * bw : (i + 1) * bw],
                all_rows_cpu[i * bw : (i + 1) * bw],
            )
            for i in range(n)
        ]

    def _init_after_prefill(
        self,
        req: BeamRequest,
        ids: List[int],
        full_kv: torch.Tensor,
        full_cpu: List[int],
        logprobs: torch.Tensor,
        cand_ids: Optional[torch.Tensor],
        hit_len: int = 0,
        suffix_cpu: Optional[List[int]] = None,
        prefix_list: Optional[List[int]] = None,
        precomputed: Optional[tuple] = None,
    ) -> None:
        if self.prefix_cache is not None:
            hit = max(int(hit_len), 0)
            # Lock the matched prefix only on the critical path. Insert the
            # prompt into radix on finish (SGLang cache_finished) so unique
            # suffix pages can be reused without a  per-prefill trie walk.
            lock_ids = ids[:hit] if hit > 0 else []
            if lock_ids:
                self.prefix_cache.lock(lock_ids)
                req.radix_lock_tokens = list(lock_ids)
            else:
                req.radix_lock_tokens = None
            req.prefill_kv_cpu = list(full_cpu)
        else:
            req.prefill_kv_cpu = None
            req.prefill_kv_cpu_all = list(full_cpu)
        if precomputed is not None:
            # Wave-batched path: selection, beam rows and the prefill row copy
            # were done once for the whole wave (limit > 1 guaranteed there).
            req.beam_list, req.beam_pool_indices, req.beam_pool_indices_cpu = (
                precomputed
            )
            req.expanded = True
            return
        limit = self.generation_token_limit(req.max_new_tokens)
        req.beam_list = init_from_prefill(
            logprobs,
            beam_width=req.beam_width,
            beam_candidates=req.beam_candidates,
            max_new_tokens=max(limit, req.max_new_tokens),
            prompt_len=req.prompt_len,
            valid_path=self.valid_path,
            candidate_token_ids=(
                cand_ids if isinstance(cand_ids, torch.Tensor) else None
            ),
            temperature=req.temperature,
        )
        if limit <= 1:
            req.finished = True
            req.finish_reason = FinishReason(type="length", length=req.max_new_tokens)
            req.beam_list.completed = req.beam_list.sequences_from_token_ids(
                req.beam_list.cum_logprobs,
                length_penalty=req.length_penalty,
                finish_reason=req.finish_reason,
            )
            return
        beam_rows = self.runner.alloc_req_slots(req.beam_width)
        beam_rows_cpu = list(self.runner.req_pool.last_alloc_cpu)
        self.runner.copy_prefill_to_beams(
            req.prefill_pool_idx, beam_rows, req.prompt_len
        )
        req.beam_pool_indices = beam_rows
        req.beam_pool_indices_cpu = beam_rows_cpu
        req.expanded = True

    def _expand_groups(
        self,
        groups: List[List[BeamRequest]],
        logprobs: torch.Tensor,
        cand_ids: Optional[torch.Tensor],
    ) -> None:
        offset = 0
        for group in groups:
            n_rows = sum(int(r.beam_width) for r in group)
            sl = logprobs[offset : offset + n_rows]
            self._expand_decode(group, sl, cand_ids)
            offset += n_rows

    def _ordered_live(
        self, reqs: List[BeamRequest]
    ) -> Tuple[List[BeamRequest], List[List[BeamRequest]]]:
        live = [r for r in reqs if not r.finished]
        groups = group_by_beam_depth(
            live,
            lambda r: int(r.beam_width),
            lambda r: (
                int(r.beam_list.generated_len()) if r.beam_list is not None else 0
            ),
        )
        ordered = [r for group in groups for r in group]
        return ordered, groups

    def _decode_tick(self, reqs: List[BeamRequest]) -> None:
        """One scheduler tick: process prior pipe result, launch one pipe (or serial)."""
        live = [r for r in reqs if not r.finished]
        if not live:
            return
        pipe = self.pipeline
        is_fin = lambda r: bool(r.finished)
        if pipe.enabled and (pipe.active or len(live) >= 2):
            if not pipe.active:
                pipe.try_materialize(live, is_fin)
            if pipe.active:
                pipe.sync_pipes_from_running(live, is_fin)
                pipe.try_dematerialize(
                    is_fin, wait_events=self._wait_all_pipeline_events
                )
                if pipe.active:
                    self._decode_one_pipe()
                    return
        self._decode_one_step(live)

    def _decode_one_pipe(self) -> None:
        pipe = self.pipeline

        def process(launch: _DecodeLaunch) -> None:
            self._process_pipe_result(launch)

        batch = pipe.get_next(process, lambda r: bool(r.finished))
        if batch is None:
            # Do not drain_all or serial-forward leftover live reqs. That path
            # launched the full conc=8 / 400-row batch and killed 3-pipe overlap.
            return
        launch = self._launch_pipe_forward(batch, pipe.current_pipe_id)
        pipe.enqueue(pipe.current_pipe_id, launch)

    def _launch_pipe_forward(
        self, reqs: List[BeamRequest], pipe_id: Optional[int]
    ) -> _DecodeLaunch:
        self._wait_pipe_ready(pipe_id, pop=True)
        ordered, _ = self._ordered_live(reqs)
        for req in ordered:
            ev = getattr(req, "prefill_done", None)
            if ev is not None:
                self._wait_event(ev)
                req.prefill_done = None
        with trace_range(f"flashrec.pipe_fwd pipe={pipe_id} reqs={len(ordered)}"):
            logprobs, cands, fused = self._forward_decode(ordered)
        replay_done = None
        if torch.cuda.is_available():
            replay_done = torch.cuda.Event()
            replay_done.record()
        return _DecodeLaunch(
            reqs=ordered,
            logprobs=logprobs,
            cand_ids=cands,
            pipe_id=pipe_id,
            replay_done=replay_done,
            fused=fused,
        )

    def _run_expand_overlap(
        self,
        apply_fn: Callable[[], None],
        replay_done: Optional[torch.cuda.Event],
        pipe_id: Optional[int],
    ) -> None:
        """Expand on expand_stream; KV remap on copy_stream (SGLang beam_copy)."""
        expand_stream = self.pipeline.expand_stream
        overlap = (
            expand_stream is not None
            and torch.cuda.is_available()
            and isinstance(expand_stream, torch.cuda.Stream)
        )

        def _apply_with_pipe() -> None:
            prev = self._expand_pipe_id
            self._expand_pipe_id = pipe_id
            try:
                apply_fn()
            finally:
                self._expand_pipe_id = prev

        if not overlap:
            if replay_done is not None:
                self._wait_event(replay_done)
            _apply_with_pipe()
            return
        self._pending_remaps.clear()
        self._defer_remap = True
        try:
            if replay_done is not None:
                expand_stream.wait_event(replay_done)
            with torch.cuda.stream(expand_stream):
                _apply_with_pipe()
        finally:
            self._defer_remap = False
        ev = torch.cuda.Event()
        ev.record(expand_stream)
        self.pipeline.expand_done[pipe_id] = ev
        copy_stream = self.pipeline.copy_stream
        remap_stream = copy_stream if copy_stream is not None else expand_stream
        if self._pending_remaps:
            remap_stream.wait_event(ev)
            with torch.cuda.stream(remap_stream):
                self._flush_pending_remaps()
            remap_ev = torch.cuda.Event()
            remap_ev.record(remap_stream)
            self.pipeline.remap_done[pipe_id] = remap_ev
        else:
            self.pipeline.remap_done[pipe_id] = ev

    def _process_pipe_result(self, launch: _DecodeLaunch) -> None:
        def _apply() -> None:
            if launch.fused is not None:
                self._apply_fused(launch.reqs, launch.fused)
                return
            groups = group_by_beam_depth(
                launch.reqs,
                lambda r: int(r.beam_width),
                lambda r: (
                    int(r.beam_list.generated_len()) if r.beam_list is not None else 0
                ),
            )
            self._expand_groups(groups, launch.logprobs, launch.cand_ids)

        self._run_expand_overlap(_apply, launch.replay_done, launch.pipe_id)

    def _decode_one_step(self, reqs: List[BeamRequest]) -> None:
        """Serial decode: one forward over all live reqs (conc=1 / pipeline off)."""
        live = [r for r in reqs if not r.finished]
        if not live:
            return
        ordered, groups = self._ordered_live(live)
        with trace_range(
            f"flashrec.decode_step reqs={len(ordered)} groups={len(groups)}"
        ):
            self._wait_pipe_ready(None, pop=True)
            logprobs, cands, fused = self._forward_decode(ordered)
            replay_done = None
            if torch.cuda.is_available():
                replay_done = torch.cuda.Event()
                replay_done.record()

            def _apply() -> None:
                if fused is not None:
                    self._apply_fused(ordered, fused)
                    return
                self._expand_groups(groups, logprobs, cands)

            self._run_expand_overlap(_apply, replay_done, None)

    def _forward_decode(self, reqs: List[BeamRequest]):
        device = self.device
        tok_srcs = [
            req.beam_list.last_tokens.to(device=device, dtype=torch.int64)
            for req in reqs
        ]
        # Wave cache: pool_rows persists across decode steps within the same wave.
        # _wave_seq_base holds seq_len at wave start; step counter gives O(1)
        # position/seq_lens computation without rebuilding Python lists or H2D.
        # Key on rid, not id(): CPython recycles addresses, so a freed request's
        # id() can reappear on a new one and make it inherit the previous wave's
        # _wave_step/_wave_seq_base -- decode then starts mid-sequence.
        reqs_key = tuple(r.rid for r in reqs)
        wave_hit = self._wave_reqs_key == reqs_key and self._wave_pool_rows is not None
        if wave_hit:
            pool_rows = self._wave_pool_rows
            n_rows = self._wave_n_rows
            self._wave_step += 1
        else:
            row_srcs = [req.beam_pool_indices.to(device=device) for req in reqs]
            n_rows = int(sum(int(t.numel()) for t in tok_srcs))
            pool_rows = self._stage.copy_rows("dec_rows", row_srcs, device, torch.int64)
            seq_base: List[int] = []
            for req in reqs:
                seq_base.extend([int(req.seq_len)] * int(req.beam_width))
            seq_base_gpu, _ = self._stage.copy_list(
                "_wave_base", seq_base, device, torch.int64
            )
            self._wave_pool_rows = pool_rows
            self._wave_seq_base = seq_base_gpu
            self._wave_seq_base_sum = int(sum(seq_base))
            self._wave_reqs_key = reqs_key
            self._wave_n_rows = n_rows
            self._wave_step = 0
        step = self._wave_step
        pack = bool(getattr(self.config, "enable_decode_pack", True))
        if not pack:
            last_tokens = torch.cat(tok_srcs)
        out_cache = self.runner.alloc_tokens(n_rows)
        out_cache_cpu = self.runner.kv_pool.last_alloc_cpu
        table = self.runner.req_pool.req_to_token
        # Batched write_decode: column index = seq_base + step (positions where new KV is written)
        col_idx = self._wave_seq_base[:n_rows] + step
        table[pool_rows.to(dtype=torch.int64), col_idx] = out_cache.to(
            dtype=table.dtype
        ).view(-1)
        offset = 0
        for req in reqs:
            bw = int(req.beam_width)
            req.beam_list.record_owned_decode_kv_cpu(
                out_cache_cpu[offset : offset + bw]
            )
            offset += bw
        for req in reqs:
            req.seq_len += 1
        graph = self.runner.graph
        packed = (
            graph.buffers_for(n_rows)
            if graph is not None and graph.can_replay(n_rows)
            else None
        )
        with torch.inference_mode():
            if packed is not None:
                _, buf = packed
                raw = n_rows
                if pack:
                    last_tokens = self._stage.copy_rows(
                        "dec_tok",
                        tok_srcs,
                        device,
                        torch.int64,
                        dest=buf["input_ids"][:raw],
                    )
                else:
                    buf["input_ids"][:raw].copy_(
                        last_tokens.view(-1)[:raw], non_blocking=True
                    )
                buf["req_pool"][:raw].copy_(pool_rows[:raw], non_blocking=True)
                buf["out_loc"][:raw].copy_(out_cache.view(-1)[:raw], non_blocking=True)
                # positions = base + step; seq_lens = base + step + 1
                torch.add(self._wave_seq_base[:raw], step, out=buf["positions"][:raw])
                torch.add(
                    self._wave_seq_base[:raw], step + 1, out=buf["seq_lens"][:raw]
                )
                buf["seq_lens_cpu"][:raw].copy_(
                    buf["seq_lens"][:raw], non_blocking=True
                )
                seq_after_total = self._wave_seq_base_sum + raw * (step + 1)
                with trace_range(f"flashrec.gather_kv rows={raw}"):
                    kv_indices = self.runner.gather_kv_indices(
                        pool_rows,
                        buf["seq_lens"][:raw],
                        out=buf["kv_indices"],
                        total=seq_after_total,
                    )
                want_expand = self._can_graph_expand(reqs, n_rows)
                if want_expand:
                    self._pack_graph_expand(reqs, buf)
                batch = ForwardBatch(
                    input_ids=buf["input_ids"][:raw],
                    req_pool_indices=buf["req_pool"][:raw],
                    seq_lens=buf["seq_lens"][:raw],
                    seq_lens_cpu=buf["seq_lens_cpu"][:raw],
                    positions=buf["positions"][:raw],
                    out_cache_loc=buf["out_loc"][:raw],
                    is_prefill=False,
                    extend_prefix_lens=None,
                    extend_seq_lens=[1] * n_rows,
                    kv_indices=kv_indices,
                    buffers_ready=True,
                    want_expand=want_expand,
                )
            else:
                if pack:
                    last_tokens = self._stage.copy_rows(
                        "dec_tok", tok_srcs, device, torch.int64
                    )
                positions_gpu = self._wave_seq_base + step
                seq_gpu = self._wave_seq_base + (step + 1)
                seq_after_total = self._wave_seq_base_sum + n_rows * (step + 1)
                with trace_range(f"flashrec.gather_kv rows={n_rows}"):
                    kv_indices = self.runner.gather_kv_indices(
                        pool_rows, seq_gpu, total=seq_after_total
                    )
                batch = ForwardBatch(
                    input_ids=last_tokens,
                    req_pool_indices=pool_rows,
                    seq_lens=seq_gpu,
                    seq_lens_cpu=seq_gpu,
                    positions=positions_gpu,
                    out_cache_loc=out_cache,
                    is_prefill=False,
                    extend_prefix_lens=None,
                    extend_seq_lens=[1] * n_rows,
                    kv_indices=kv_indices,
                )
            with trace_range(f"flashrec.decode_fwd reqs={len(reqs)}"):
                return self.runner.forward(batch)

    def _expand_decode(
        self,
        reqs: List[BeamRequest],
        logprobs: torch.Tensor,
        cand_ids: Optional[torch.Tensor],
    ) -> None:
        with trace_range(f"flashrec.expand reqs={len(reqs)}"):
            self._expand_decode_body(reqs, logprobs, cand_ids)

    def _expand_decode_body(
        self,
        reqs: List[BeamRequest],
        logprobs: torch.Tensor,
        cand_ids: Optional[torch.Tensor],
    ) -> None:
        offset = 0
        use_fused = (
            self.config.enable_fused_expand
            and self.valid_path.active
            and self.valid_path.mode == "trie"
            # Fused kernels need the dense allow/next tables; large (CSR-only)
            # tries take the eager per-request path below.
            and self.valid_path._allow_table is not None
            and all(r.ignore_eos for r in reqs)
            # Fused CUDA select has no Gumbel-noise hook; sampled requests take
            # the eager per-request path below.
            and all(r.temperature <= 0.0 for r in reqs)
        )
        if use_fused and len(reqs) >= 1:
            try:
                self._expand_fused(reqs, logprobs, cand_ids)
                return
            except Exception:
                logger.debug("fused expand fallback", exc_info=True)
        for req in reqs:
            bw = int(req.beam_width)
            topk = min(req.beam_candidates, int(logprobs.shape[1]))
            row = logprobs[offset : offset + bw]
            # Temperature rescaling must see the full row (renormalization is
            # only exact over the whole candidate vocabulary), so it happens
            # before the per-row top-k truncation.
            row = apply_temperature(row, req.temperature)
            if row.shape[1] <= topk:
                vals, idx = torch.sort(row, dim=1, descending=True)
            else:
                vals, idx = torch.topk(row, topk, dim=1, largest=True, sorted=True)
            tokens = cand_ids[idx] if isinstance(cand_ids, torch.Tensor) else idx
            self._apply_expand(req, tokens, vals)
            offset += bw

    def _ws_for_expand(self) -> GenrecFusedWorkspace:
        pid = self._expand_pipe_id
        if pid is None:
            return self._fused_ws
        ws = self._fused_ws_pipe.get(pid)
        if ws is None:
            ws = GenrecFusedWorkspace()
            self._fused_ws_pipe[pid] = ws
        return ws

    def _expand_fused(
        self,
        reqs: List[BeamRequest],
        logprobs: torch.Tensor,
        cand_ids: Optional[torch.Tensor],
    ) -> None:
        n = len(reqs)
        bw = int(reqs[0].beam_width)
        topk = min(reqs[0].beam_candidates, int(logprobs.shape[1]))
        stacked = logprobs.view(n, bw, -1)
        if stacked.shape[-1] > topk:
            vals, idx = torch.topk(stacked, topk, dim=-1, largest=True, sorted=True)
        else:
            vals, idx = torch.sort(stacked, dim=-1, descending=True)
        tokens = cand_ids[idx] if isinstance(cand_ids, torch.Tensor) else idx
        ws = self._ws_for_expand()
        cum = ws.get("cum_in", (n, bw), torch.float32, self.device)
        width = int(reqs[0].beam_list.token_ids.shape[-1])
        token_ids = pick_pingpong_stack(
            ws,
            "tok",
            [r.beam_list.token_ids for r in reqs],
            (n, bw, width),
            torch.int64,
            self.device,
        )
        has_nodes = reqs[0].beam_list.node_ids is not None
        node_ids = None
        if has_nodes:
            node_ids = pick_pingpong_stack(
                ws,
                "node",
                [r.beam_list.node_ids for r in reqs],
                (n, bw),
                torch.int64,
                self.device,
            )
        for i, r in enumerate(reqs):
            cum[i].copy_(
                r.beam_list.cum_logprobs.to(device=self.device, dtype=torch.float32)
            )
        self.valid_path._ensure_gpu_cache(self.device)
        cols = [int(r.beam_list.cur_len) for r in reqs]
        col_t = torch.tensor(cols, dtype=torch.int32, device=self.device)
        limits = [self.generation_token_limit(r.max_new_tokens) for r in reqs]
        will_finish = [
            r.beam_list.generated_len() + 1 >= lim for r, lim in zip(reqs, limits)
        ]
        fused = genrec_mask_topk_expand(
            cum,
            vals.to(dtype=torch.float32),
            tokens.to(dtype=torch.int64),
            select_k=bw,
            node_ids=node_ids,
            allow_table=self.valid_path._allow_table,
            token_base=int(self.valid_path.token_base or 0),
            token_ids=token_ids,
            col=col_t,
            next_node=self.valid_path._next_node,
            invalid_node=int(self.valid_path._invalid_node),
            apply_mask=True,
            apply_expand=True,
            apply_advance=node_ids is not None,
            inplace=False,
            workspace=ws,
        )
        for i, req in enumerate(reqs):
            req.beam_list.cum_logprobs = fused.vals[i].clone()
            req.beam_list.last_tokens = fused.tokens[i]
            req.beam_list.token_ids = fused.token_ids[i]
            req.beam_list.cur_len = cols[i] + 1
            req.beam_list.dense_authoritative = True
            if fused.node_ids is not None:
                req.beam_list.node_ids = fused.node_ids[i]
            parents = fused.parents[i]
            if will_finish[i]:
                req.finished = True
                req.finish_reason = FinishReason(
                    type="length", length=req.max_new_tokens
                )
            else:
                self._remap_req(req, parents)

    def _apply_expand(
        self, req: BeamRequest, tokens: torch.Tensor, vals: torch.Tensor
    ) -> None:
        limit = self.generation_token_limit(req.max_new_tokens)
        will_finish = req.beam_list.generated_len() + 1 >= limit
        parents = expand_step(
            req.beam_list,
            tokens,
            vals,
            beam_width=req.beam_width,
            valid_path=self.valid_path,
            stop_token_ids=req.stop_token_ids,
            ignore_eos=req.ignore_eos,
            will_finish=will_finish,
            temperature=req.temperature,
        )
        if will_finish or parents is None:
            req.finished = True
            req.finish_reason = FinishReason(type="length", length=req.max_new_tokens)
        else:
            self._remap_req(req, parents)

    def _flush_pending_remaps(self) -> None:
        for req, parents in self._pending_remaps:
            self._remap_req(req, parents)
        self._pending_remaps.clear()

    def _remap_req(self, req: BeamRequest, parents_rel: torch.Tensor) -> None:
        if self._defer_remap:
            self._pending_remaps.append((req, parents_rel))
            return
        bw = int(req.beam_width)
        n_owned = req.beam_list.owned_decode_kv_count()
        n_decode = int(n_owned) // max(bw, 1) if n_owned > 0 else 0
        seq_len = int(req.prompt_len) + max(n_decode, 0)
        remap_by_parents(
            self.runner.req_pool.req_to_token,
            req.beam_pool_indices,
            parents_rel,
            req.prompt_len,
            seq_len,
        )

    def _radix_cache_finished(self, req: BeamRequest) -> None:
        """Insert prompt KV into radix after decode (SGLang cache_finished)."""
        full = getattr(req, "prefill_kv_cpu", None)
        ids = list(req.input_ids or [])
        req.prefill_kv_cpu = None
        if self.prefix_cache is None or not full or len(ids) < 2:
            if full:
                start = int(req.cached_tokens)
                if start < len(full):
                    self.runner.free_tokens(full[start:])
            return
        if len(full) != len(ids):
            start = int(req.cached_tokens)
            if start < len(full):
                self.runner.free_tokens(full[start:])
            return
        if req.radix_lock_tokens:
            self.prefix_cache.unlock(req.radix_lock_tokens)
            req.radix_lock_tokens = None
        ins_tok = ids[:-1]
        ins_kv = full[:-1]
        max_ins = 128
        to_free: List[int] = [int(full[-1])]
        if len(ins_tok) > max_ins:
            to_free.extend(int(x) for x in full[max_ins:-1])
            ins_tok = ins_tok[:max_ins]
            ins_kv = ins_kv[:max_ins]
        cap = 49152
        extra = int(self.prefix_cache.num_cached_tokens) - cap
        if extra > 0:
            # Amortized eviction: one big evict(extra + 4096) walks thousands
            # of Python trie nodes and freezes every in-flight request for
            # 200ms+ (the periodic P99 wave). Each completion inserts <=128
            # tokens, so a bounded evict per completion keeps up while capping
            # a single stall at ~1-2ms.
            self.prefix_cache.evict(min(extra, 512))
        if ins_tok:
            canon = self.prefix_cache.insert_cpu(ins_tok, ins_kv)
            to_free.extend(int(a) for a, b in zip(ins_kv, canon) if int(a) != int(b))
        if to_free:
            self.runner.free_tokens(to_free)

    def _free_request(self, req: BeamRequest) -> None:
        if req.beam_list is None:
            return
        owned_cpu = req.beam_list.take_owned_decode_kv_cpu()
        if owned_cpu:
            self.runner.free_tokens(owned_cpu)
        if self.prefix_cache is not None and getattr(req, "prefill_kv_cpu", None):
            self._radix_cache_finished(req)
        else:
            if self.prefix_cache is not None and req.radix_lock_tokens:
                self.prefix_cache.unlock(req.radix_lock_tokens)
                req.radix_lock_tokens = None
            if req.prefill_pool_idx >= 0 and req.prompt_len > 0:
                start = int(req.cached_tokens) if self.prefix_cache is not None else 0
                if start < int(req.prompt_len):
                    full = getattr(req, "prefill_kv_cpu_all", None)
                    if full is not None:
                        self.runner.free_tokens(full[start : int(req.prompt_len)])
                        req.prefill_kv_cpu_all = None
                    else:
                        table = self.runner.req_pool.req_to_token
                        prefill_kv = table[
                            int(req.prefill_pool_idx), start : int(req.prompt_len)
                        ]
                        self.runner.free_tokens(prefill_kv.to(dtype=torch.int64))
        if req.prefill_pool_idx >= 0:
            self.runner.free_req_slots([req.prefill_pool_idx])
            req.prefill_pool_idx = -1
        if req.beam_pool_indices is not None:
            if req.beam_pool_indices_cpu is not None:
                self.runner.free_req_slots(req.beam_pool_indices_cpu)
                req.beam_pool_indices_cpu = None
            else:
                self.runner.free_req_slots(req.beam_pool_indices)
            req.beam_pool_indices = None
