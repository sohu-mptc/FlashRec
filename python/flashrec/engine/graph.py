"""CUDA Graph capture/replay for decode.

FlashInfer ``plan()`` runs *outside* the graph (SGLang replay_prepare). The
captured region is the model forward, optional restricted LM-head, and optional
row top-k + fused trie expand. Replay pads to the next captured batch size.

When cascade attention is enabled, a second graph per ``bs`` captures
``MultiLevelCascadeAttentionWrapper.run`` (plus expand if the paged expand
graph was captured). Replay selects ``graphs`` vs ``graphs_cascade``.

Outputs that later kernels/CPU still need (vals/parents/tokens) are cloned so
two pipes that pad to the same ``bs`` do not clobber each other. Token/node
planes ping-pong between ``exp_tok_in``/``exp_tok_out`` when both expand graphs
were captured; the scheduler steals those views when ``n_live >= 2`` and clones
a plane when another pipe still aliases the next write buffer, or when a
single-request (``n=1``) wave would otherwise keep a view into the static
buffer that the next conc=1 replay overwrites.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from flashrec.core import ForwardBatch
from flashrec.kernel.beam_trie import GenrecFusedResult, call_genrec_cuda

logger = logging.getLogger(__name__)

LogitsFn = Callable[[torch.Tensor, torch.Tensor], None]


def fill_topk_tokens(
    cand_ids: torch.Tensor, idx: torch.Tensor, out: torch.Tensor
) -> None:
    """Write ``cand_ids[idx]`` into ``out``. Safe to capture inside a CUDA graph.

    Advanced indexing ``cand_ids[idx]`` with a leading size-1 batch (the
    single-request ``bs = beam_width`` graph) can freeze to capture-time
    values, so every beam row is written as the same SID. ``gather`` +
    ``copy_`` into a static buffer re-executes on replay.
    """
    out.copy_(cand_ids.gather(0, idx.reshape(-1)).view(idx.shape))


@dataclass
class LevelCaptureSpec:
    level: int
    logits_fn: LogitsFn
    logprobs_k: int
    expand_spec: ExpandCaptureSpec


@dataclass
class ExpandCaptureSpec:
    beam_width: int
    cand: int
    select_k: int
    width: int
    cand_ids: torch.Tensor
    allow_table: torch.Tensor
    next_node: torch.Tensor
    token_base: int
    invalid_node: int


class DecodeGraphRunner:
    def __init__(
        self,
        device: torch.device,
        max_bs: int,
        max_seq_len: int,
        capture_bs: List[int],
        attn,
    ):
        self.device = device
        self.max_bs = max_bs
        self.max_seq_len = max_seq_len
        self.capture_bs = sorted(b for b in capture_bs if 0 < b <= max_bs)
        self.attn = attn
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.graphs_expand: Dict[int, torch.cuda.CUDAGraph] = {}
        self.graphs_expand_flip: Dict[int, torch.cuda.CUDAGraph] = {}
        self.graphs_cascade: Dict[int, torch.cuda.CUDAGraph] = {}
        self.graphs_cascade_expand: Dict[int, torch.cuda.CUDAGraph] = {}
        self.graphs_cascade_expand_flip: Dict[int, torch.cuda.CUDAGraph] = {}
        self.static: Dict[int, dict] = {}
        self._pool = None
        self.enabled = device.type == "cuda"
        self.raw_bs = 0
        self.captures_logprobs = False
        self._logprobs_k: Optional[int] = None
        self.expand_spec: Optional[ExpandCaptureSpec] = None
        self.expand_bw: Optional[int] = None
        self._shared_logprobs: Optional[torch.Tensor] = None
        # Per-level graphs: (bs, level) -> graph capturing model_fwd + level lm_head + expand
        self.graphs_level: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.graphs_level_flip: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.static_level: Dict[Tuple[int, int], dict] = {}
        self.level_specs: Dict[int, LevelCaptureSpec] = {}

    def capture(
        self,
        model_fn: Callable[[ForwardBatch], torch.Tensor],
        prepare_fn: Callable[[ForwardBatch], None],
        logits_fn: Optional[LogitsFn] = None,
        logprobs_k: Optional[int] = None,
        expand_spec: Optional[ExpandCaptureSpec] = None,
    ) -> None:
        if not self.enabled or not self.capture_bs:
            return
        use_logits = (
            logits_fn is not None and logprobs_k is not None and int(logprobs_k) > 0
        )
        self.captures_logprobs = bool(use_logits)
        self._logprobs_k = int(logprobs_k) if use_logits else None
        self.expand_spec = expand_spec
        self.expand_bw = (
            int(expand_spec.beam_width) if expand_spec is not None else None
        )
        extra = f" logprobs_k={self._logprobs_k}" if use_logits else ""
        if expand_spec is not None:
            extra += f" expand_bw={expand_spec.beam_width} C={expand_spec.cand}"
        logger.info(
            "CUDA graph capture start: bs=%s%s",
            list(reversed(self.capture_bs)),
            extra,
        )
        for bs in reversed(self.capture_bs):
            try:
                self._capture_one(
                    bs,
                    model_fn,
                    prepare_fn,
                    logits_fn if use_logits else None,
                    self._logprobs_k,
                    expand_spec,
                )
            except Exception as exc:
                logger.warning("CUDA graph capture failed for bs=%s: %s", bs, exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def capture_levels(
        self,
        model_fn: Callable[[ForwardBatch], torch.Tensor],
        prepare_fn: Callable[[ForwardBatch], None],
        level_specs: Dict[int, LevelCaptureSpec],
    ) -> None:
        """Capture per-level graphs: model_fwd + level lm_head + level expand.

        Must be called *after* ``capture()`` so the model-only graphs and
        memory pool already exist. Idempotent: a later call with the same
        ``(bs, level)`` set is a no-op so warmup / ``generate_many`` cannot
        recapture into the shared graph pool (that path dropped ``s_b``).
        """
        if not self.enabled or not self.capture_bs or not level_specs:
            return
        wanted = [
            (bs, level)
            for level, lspec in level_specs.items()
            for bs in self.capture_bs
            if int(lspec.expand_spec.beam_width) > 0
            and bs % int(lspec.expand_spec.beam_width) == 0
        ]
        if wanted and all(key in self.graphs_level for key in wanted):
            return
        self.level_specs = dict(level_specs)
        for level, lspec in level_specs.items():
            for bs in reversed(self.capture_bs):
                if (bs, level) in self.graphs_level:
                    continue
                try:
                    self._capture_level_one(bs, level, model_fn, prepare_fn, lspec)
                except Exception as exc:
                    logger.warning(
                        "per-level graph capture failed bs=%d level=%d: %s",
                        bs, level, exc,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        captured = sorted(self.graphs_level.keys())
        if captured:
            logger.info("per-level CUDA graphs captured: %s", captured)

    def _capture_level_one(
        self,
        bs: int,
        level: int,
        model_fn,
        prepare_fn,
        lspec: LevelCaptureSpec,
    ) -> None:
        device = self.device
        max_idx = bs * self.max_seq_len
        fi = self.attn.graph_bufs(bs) if hasattr(self.attn, "graph_bufs") else None
        kv_indices = (
            fi["indices"]
            if fi is not None and fi.get("indices") is not None
            else torch.zeros(max_idx, dtype=torch.int32, device=device)
        )
        k = int(lspec.logprobs_k)
        buf: dict = {
            "input_ids": torch.zeros(bs, dtype=torch.int64, device=device),
            "req_pool": torch.zeros(bs, dtype=torch.int64, device=device),
            "seq_lens": torch.ones(bs, dtype=torch.int64, device=device),
            "seq_lens_cpu": torch.ones(bs, dtype=torch.int64, pin_memory=True),
            "positions": torch.zeros(bs, dtype=torch.int64, device=device),
            "out_loc": torch.zeros(bs, dtype=torch.int64, device=device),
            "kv_indices": kv_indices,
            "hidden": None,
            "logprobs": torch.empty(bs, k, dtype=torch.float32, device=device),
        }
        espec = lspec.expand_spec
        want_expand = self._alloc_expand_bufs(buf, bs, espec) is not None
        if not want_expand:
            return
        batch = ForwardBatch(
            input_ids=buf["input_ids"],
            req_pool_indices=buf["req_pool"],
            seq_lens=buf["seq_lens"],
            seq_lens_cpu=buf["seq_lens_cpu"],
            positions=buf["positions"],
            out_cache_loc=buf["out_loc"],
            is_prefill=False,
            extend_prefix_lens=[0] * bs,
            extend_seq_lens=[1] * bs,
            kv_indices=buf["kv_indices"][:bs],
        )
        logits_fn = lspec.logits_fn

        self.attn.begin_graph_decode(bs)
        try:
            prepare_fn(batch)

            def _run_level():
                hidden = model_fn(batch)
                logits_fn(hidden, buf["logprobs"])
                return hidden

            def _run_level_expand():
                _run_level()
                self._run_expand(buf, espec, flip=False)

            for _ in range(2):
                _run_level()
            torch.cuda.synchronize()

            for _ in range(2):
                _run_level_expand()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            kwargs = {"pool": self._pool} if self._pool is not None else {}
            with torch.cuda.graph(g, **kwargs):
                _run_level_expand()
            torch.cuda.synchronize()
            if self._pool is None:
                try:
                    self._pool = g.pool()
                except Exception:
                    self._pool = torch.cuda.graph_pool_handle()

            key = (bs, level)
            self.graphs_level[key] = g
            buf["batch"] = batch
            # Per-level graphs run once per request (L0/L1/L3 are different
            # keys). Ping-pong would only flip *between* requests; the flip
            # replay was observed to leave the newly written SID column as
            # token 0 (``<s_a>!<s_c>``). Always pack tok_in → tok_out.
            buf["exp_pingpong"] = False
            buf["exp_phase"] = 0
            self.static_level[key] = buf
            logger.info(
                "captured per-level CUDA graph bs=%d level=%d +expand k=%d",
                bs, level, k,
            )
        finally:
            self.attn.end_graph_decode()

    def can_replay_level(self, n_rows: int, level: int) -> bool:
        bs = self.pad_bs(n_rows)
        if bs is None:
            return False
        espec = self.level_specs.get(level)
        if espec is None:
            return False
        bw = int(espec.expand_spec.beam_width)
        return (
            bw > 0
            and int(n_rows) % bw == 0
            and (bs, level) in self.graphs_level
        )

    def replay_level(
        self,
        batch: ForwardBatch,
        level: int,
        prepare_fn: Callable[[ForwardBatch], None],
        skip_copy: bool = False,
    ) -> Tuple[torch.Tensor, GenrecFusedResult]:
        raw_bs = int(batch.n_rows)
        bs = self.pad_bs(raw_bs)
        key = (bs, level)
        if key not in self.graphs_level:
            raise RuntimeError(f"no per-level CUDA graph for bs={raw_bs} level={level}")
        buf = self.static_level[key]
        sb = buf["batch"]
        if not skip_copy:
            buf["input_ids"][:raw_bs].copy_(
                batch.input_ids.view(-1)[:raw_bs], non_blocking=True
            )
            buf["seq_lens"][:raw_bs].copy_(
                batch.seq_lens.view(-1)[:raw_bs], non_blocking=True
            )
            buf["positions"][:raw_bs].copy_(
                batch.positions.view(-1)[:raw_bs], non_blocking=True
            )
            buf["out_loc"][:raw_bs].copy_(
                batch.out_cache_loc.view(-1)[:raw_bs], non_blocking=True
            )
            buf["req_pool"][:raw_bs].copy_(
                batch.req_pool_indices.view(-1)[:raw_bs], non_blocking=True
            )
            sl_cpu = batch.seq_lens_cpu.view(-1)[:raw_bs]
            if sl_cpu.device.type != "cpu":
                sl_cpu = sl_cpu.detach().to("cpu")
            buf["seq_lens_cpu"][:raw_bs].copy_(sl_cpu)
            nidx = int(batch.kv_indices.numel()) if batch.kv_indices is not None else 0
            if nidx > 0:
                src = batch.kv_indices.view(-1)[:nidx]
                if src.dtype != torch.int32:
                    src = src.to(dtype=torch.int32)
                if src.data_ptr() != buf["kv_indices"].data_ptr():
                    buf["kv_indices"][:nidx].copy_(src, non_blocking=True)
        else:
            nidx = int(batch.kv_indices.numel()) if batch.kv_indices is not None else 0
        pad = bs - raw_bs
        dirty_rows = int(buf.get("dirty_rows", bs))
        last_raw = int(buf.get("last_raw_bs", 0))
        skip_pad = pad > 0 and raw_bs >= last_raw and dirty_rows <= raw_bs
        if pad > 0 and not skip_pad and dirty_rows > raw_bs:
            end = min(dirty_rows, bs)
            buf["input_ids"][raw_bs:end].zero_()
            buf["seq_lens"][raw_bs:end].fill_(1)
            buf["seq_lens_cpu"][raw_bs:end].fill_(1)
            buf["positions"][raw_bs:end].zero_()
            buf["out_loc"][raw_bs:end].zero_()
            buf["req_pool"][raw_bs:end].zero_()
        buf["dirty_rows"] = raw_bs
        buf["last_raw_bs"] = raw_bs
        total = nidx + pad
        kv_dirty = int(buf.get("kv_dirty", int(buf["kv_indices"].numel())))
        last_nidx = int(buf.get("last_nidx", 0))
        skip_kv_pad = pad > 0 and nidx >= last_nidx and kv_dirty <= nidx
        if pad > 0 and not skip_kv_pad and kv_dirty > nidx:
            buf["kv_indices"][
                nidx : max(total, min(kv_dirty, int(buf["kv_indices"].numel())))
            ].zero_()
        buf["kv_dirty"] = nidx
        buf["last_nidx"] = nidx
        sb.kv_indices = buf["kv_indices"][:total]
        self.attn.begin_graph_decode(bs)
        try:
            prepare_fn(sb)
            phase = int(buf.get("exp_phase", 0)) if buf.get("exp_pingpong") else 0
            if phase == 1 and key in self.graphs_level_flip:
                self.graphs_level_flip[key].replay()
            else:
                self.graphs_level[key].replay()
        finally:
            self.attn.end_graph_decode()
        self.raw_bs = raw_bs
        lspec = self.level_specs[level]
        bw = int(lspec.expand_spec.beam_width)
        n_live = raw_bs // bw
        # Always clone: a stolen view into the static plane is overwritten by
        # the next request that pads to the same (bs, level), which shows up
        # as a missing SID column (token 0) on wave 2+.
        fused = self._clone_fused(buf, n_live, steal=False)
        logprobs = buf["logprobs"][:raw_bs]
        return logprobs, fused

    def _alloc_expand_bufs(
        self, buf: dict, bs: int, spec: ExpandCaptureSpec
    ) -> Optional[int]:
        bw = int(spec.beam_width)
        if bw <= 0 or bs % bw != 0:
            return None
        n = bs // bw
        C = int(spec.cand)
        ksel = int(spec.select_k)
        L = int(spec.width)
        device = self.device
        buf["exp_n"] = n
        buf["exp_cum"] = torch.zeros(n, bw, dtype=torch.float32, device=device)
        buf["exp_nodes"] = torch.zeros(n, bw, dtype=torch.int64, device=device)
        buf["exp_tok_in"] = torch.zeros(n, bw, L, dtype=torch.int64, device=device)
        buf["exp_tok_out"] = torch.zeros(n, bw, L, dtype=torch.int64, device=device)
        buf["exp_col"] = torch.zeros(n, dtype=torch.int32, device=device)
        buf["exp_do"] = torch.ones(n, dtype=torch.uint8, device=device)
        buf["exp_node_out"] = torch.zeros(n, bw, dtype=torch.int64, device=device)
        buf["exp_vals"] = torch.empty(n, ksel, dtype=torch.float32, device=device)
        buf["exp_parents"] = torch.empty(n, ksel, dtype=torch.int64, device=device)
        buf["exp_tokens"] = torch.empty(n, ksel, dtype=torch.int64, device=device)
        buf["exp_indices"] = torch.empty(n, ksel, dtype=torch.int64, device=device)
        buf["exp_scratch"] = torch.empty(n, bw * C, dtype=torch.float32, device=device)
        buf["topk_lp"] = torch.empty(n, bw, C, dtype=torch.float32, device=device)
        buf["topk_idx"] = torch.empty(n, bw, C, dtype=torch.int64, device=device)
        buf["topk_tok"] = torch.empty(n, bw, C, dtype=torch.int64, device=device)
        buf["exp_phase"] = 0
        buf["exp_pingpong"] = False
        return n

    def expand_io(
        self, buf: dict, n_live: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(tok_src, tok_dst, node_src, node_dst) for the next expand replay."""
        n = int(n_live)
        phase = int(buf.get("exp_phase", 0)) if buf.get("exp_pingpong") else 0
        if phase == 0:
            return (
                buf["exp_tok_in"][:n],
                buf["exp_tok_out"],
                buf["exp_nodes"][:n],
                buf["exp_node_out"],
            )
        return (
            buf["exp_tok_out"][:n],
            buf["exp_tok_in"],
            buf["exp_node_out"][:n],
            buf["exp_nodes"],
        )

    def _run_expand(
        self, buf: dict, spec: ExpandCaptureSpec, *, flip: bool = False
    ) -> None:
        n = int(buf["exp_n"])
        bw = int(spec.beam_width)
        C = int(spec.cand)
        logprobs = buf["logprobs"]
        stacked = logprobs.view(n, bw, -1)
        torch.topk(
            stacked,
            C,
            dim=-1,
            largest=True,
            sorted=True,
            out=(buf["topk_lp"], buf["topk_idx"]),
        )
        fill_topk_tokens(spec.cand_ids, buf["topk_idx"], buf["topk_tok"])
        if flip:
            tok_in, tok_out = buf["exp_tok_out"], buf["exp_tok_in"]
            node_in, node_out = buf["exp_node_out"], buf["exp_nodes"]
        else:
            tok_in, tok_out = buf["exp_tok_in"], buf["exp_tok_out"]
            node_in, node_out = buf["exp_nodes"], buf["exp_node_out"]
        ok = call_genrec_cuda(
            buf["exp_vals"],
            buf["exp_parents"],
            buf["exp_tokens"],
            buf["exp_indices"],
            buf["exp_scratch"],
            buf["exp_cum"],
            buf["topk_lp"],
            buf["topk_tok"],
            node_in,
            spec.allow_table,
            tok_in,
            tok_out,
            spec.next_node,
            node_out,
            buf["exp_do"],
            spec.token_base,
            spec.invalid_node,
            buf["exp_col"],
        )
        if not ok:
            raise RuntimeError("call_genrec_cuda failed during graph capture/replay")

    def _capture_one(
        self,
        bs: int,
        model_fn,
        prepare_fn,
        logits_fn: Optional[LogitsFn],
        logprobs_k: Optional[int],
        expand_spec: Optional[ExpandCaptureSpec],
    ) -> None:
        device = self.device
        max_idx = bs * self.max_seq_len
        fi = self.attn.graph_bufs(bs) if hasattr(self.attn, "graph_bufs") else None
        kv_indices = (
            fi["indices"]
            if fi is not None and fi.get("indices") is not None
            else torch.zeros(max_idx, dtype=torch.int32, device=device)
        )
        buf = {
            "input_ids": torch.zeros(bs, dtype=torch.int64, device=device),
            "req_pool": torch.zeros(bs, dtype=torch.int64, device=device),
            "seq_lens": torch.ones(bs, dtype=torch.int64, device=device),
            "seq_lens_cpu": torch.ones(bs, dtype=torch.int64, pin_memory=True),
            "positions": torch.zeros(bs, dtype=torch.int64, device=device),
            "out_loc": torch.zeros(bs, dtype=torch.int64, device=device),
            "kv_indices": kv_indices,
            "hidden": None,
            "logprobs": None,
        }
        if logits_fn is not None and logprobs_k:
            k = int(logprobs_k)
            if self._shared_logprobs is None or self._shared_logprobs.shape[0] < bs:
                self._shared_logprobs = torch.empty(
                    bs, k, dtype=torch.float32, device=device
                )
            buf["logprobs"] = self._shared_logprobs[:bs]
        want_expand = (
            expand_spec is not None
            and buf["logprobs"] is not None
            and self._alloc_expand_bufs(buf, bs, expand_spec) is not None
        )
        batch = ForwardBatch(
            input_ids=buf["input_ids"],
            req_pool_indices=buf["req_pool"],
            seq_lens=buf["seq_lens"],
            seq_lens_cpu=buf["seq_lens_cpu"],
            positions=buf["positions"],
            out_cache_loc=buf["out_loc"],
            is_prefill=False,
            extend_prefix_lens=[0] * bs,
            extend_seq_lens=[1] * bs,
            kv_indices=buf["kv_indices"][:bs],
        )
        self.attn.begin_graph_decode(bs)
        try:
            prepare_fn(batch)

            def _run_lmhead():
                hidden = model_fn(batch)
                if logits_fn is not None and buf["logprobs"] is not None:
                    logits_fn(hidden, buf["logprobs"])
                return hidden

            def _run_expand_full(flip: bool = False):
                hidden = _run_lmhead()
                self._run_expand(buf, expand_spec, flip=flip)
                return hidden

            for _ in range(2):
                hidden = _run_lmhead()
            torch.cuda.synchronize()
            if want_expand:
                try:
                    for _ in range(2):
                        hidden = _run_expand_full(False)
                    torch.cuda.synchronize()
                    g_ex = torch.cuda.CUDAGraph()
                    kwargs = {"pool": self._pool} if self._pool is not None else {}
                    with torch.cuda.graph(g_ex, **kwargs):
                        hidden = _run_expand_full(False)
                    torch.cuda.synchronize()
                    if self._pool is None:
                        try:
                            self._pool = g_ex.pool()
                        except Exception:
                            self._pool = torch.cuda.graph_pool_handle()
                    self.graphs_expand[bs] = g_ex
                    try:
                        for _ in range(2):
                            hidden = _run_expand_full(True)
                        torch.cuda.synchronize()
                        g_flip = torch.cuda.CUDAGraph()
                        kwargs = {"pool": self._pool} if self._pool is not None else {}
                        with torch.cuda.graph(g_flip, **kwargs):
                            hidden = _run_expand_full(True)
                        torch.cuda.synchronize()
                        self.graphs_expand_flip[bs] = g_flip
                        buf["exp_pingpong"] = True
                        logger.info(
                            "captured CUDA graph bs=%d +lm_head +expand pingpong", bs
                        )
                    except Exception as exc:
                        logger.warning(
                            "CUDA graph expand ping-pong capture failed for bs=%s: %s",
                            bs,
                            exc,
                        )
                        logger.info("captured CUDA graph bs=%d +lm_head +expand", bs)
                except Exception as exc:
                    logger.warning(
                        "CUDA graph expand capture failed for bs=%s: %s", bs, exc
                    )

            g = torch.cuda.CUDAGraph()
            kwargs = {"pool": self._pool} if self._pool is not None else {}
            with torch.cuda.graph(g, **kwargs):
                hidden = _run_lmhead()
            torch.cuda.synchronize()
            if self._pool is None:
                try:
                    self._pool = g.pool()
                except Exception:
                    self._pool = torch.cuda.graph_pool_handle()
            buf["hidden"] = hidden
            buf["batch"] = batch
            self.graphs[bs] = g
            self.static[bs] = buf
            tag = " +lm_head" if buf["logprobs"] is not None else ""
            if bs in self.graphs_expand:
                tag += " +expand-alt"
                if buf.get("exp_pingpong"):
                    tag += "+pp"
            logger.info("captured CUDA graph bs=%d%s", bs, tag)
        finally:
            self.attn.end_graph_decode()
        if bs in self.graphs:
            self._capture_cascade_graphs(
                bs,
                buf,
                batch,
                model_fn,
                prepare_fn,
                logits_fn,
                expand_spec,
            )

    def _capture_cascade_graphs(
        self,
        bs: int,
        buf: dict,
        batch: ForwardBatch,
        model_fn,
        prepare_fn,
        logits_fn: Optional[LogitsFn],
        expand_spec: Optional[ExpandCaptureSpec],
    ) -> None:
        """Capture a second graph that runs cascade attention on the same static bufs."""
        if not getattr(self.attn, "supports_cascade", False):
            return
        if not hasattr(self.attn, "begin_graph_cascade_decode"):
            return
        if int(bs) not in getattr(self.attn, "_cascade_graph_wrappers", {}):
            return
        dummy = self.attn.dummy_cascade_meta(bs)
        self.attn.begin_graph_cascade_decode(bs)
        prev_cascade = batch.cascade
        try:
            batch.cascade = dummy
            prepare_fn(batch)

            def _run_lmhead():
                hidden = model_fn(batch)
                if logits_fn is not None and buf["logprobs"] is not None:
                    logits_fn(hidden, buf["logprobs"])
                return hidden

            def _run_expand_full(flip: bool = False):
                hidden = _run_lmhead()
                if expand_spec is None:
                    return hidden
                self._run_expand(buf, expand_spec, flip=flip)
                return hidden

            for _ in range(2):
                hidden = _run_lmhead()
            torch.cuda.synchronize()
            want_expand = bs in self.graphs_expand and expand_spec is not None
            if want_expand:
                try:
                    for _ in range(2):
                        hidden = _run_expand_full(False)
                    torch.cuda.synchronize()
                    g_ex = torch.cuda.CUDAGraph()
                    kwargs = {"pool": self._pool} if self._pool is not None else {}
                    with torch.cuda.graph(g_ex, **kwargs):
                        hidden = _run_expand_full(False)
                    torch.cuda.synchronize()
                    self.graphs_cascade_expand[bs] = g_ex
                    if buf.get("exp_pingpong") and bs in self.graphs_expand_flip:
                        try:
                            for _ in range(2):
                                hidden = _run_expand_full(True)
                            torch.cuda.synchronize()
                            g_flip = torch.cuda.CUDAGraph()
                            kwargs = {
                                "pool": self._pool
                            } if self._pool is not None else {}
                            with torch.cuda.graph(g_flip, **kwargs):
                                hidden = _run_expand_full(True)
                            torch.cuda.synchronize()
                            self.graphs_cascade_expand_flip[bs] = g_flip
                        except Exception as exc:
                            logger.warning(
                                "CUDA graph cascade expand ping-pong capture "
                                "failed for bs=%s: %s",
                                bs,
                                exc,
                            )
                except Exception as exc:
                    logger.warning(
                        "CUDA graph cascade expand capture failed for bs=%s: %s",
                        bs,
                        exc,
                    )

            g = torch.cuda.CUDAGraph()
            kwargs = {"pool": self._pool} if self._pool is not None else {}
            with torch.cuda.graph(g, **kwargs):
                hidden = _run_lmhead()
            torch.cuda.synchronize()
            buf["hidden"] = hidden
            self.graphs_cascade[bs] = g
            tag = " +cascade"
            if buf["logprobs"] is not None:
                tag += " +lm_head"
            if bs in self.graphs_cascade_expand:
                tag += " +expand"
                if bs in self.graphs_cascade_expand_flip:
                    tag += "+pp"
            logger.info("captured CUDA graph bs=%d%s", bs, tag)
        except Exception as exc:
            logger.warning("CUDA graph cascade capture failed for bs=%s: %s", bs, exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            self.attn.end_graph_cascade_decode()
            batch.cascade = prev_cascade

    def pad_bs(self, n_rows: int) -> Optional[int]:
        if not self.enabled or not self.capture_bs:
            return None
        n = int(n_rows)
        if n <= 0 or n > self.capture_bs[-1]:
            return None
        # Skip buckets whose capture failed: a failed bs would otherwise force
        # every batch padding into it to run eager forever, even when a larger
        # bucket captured fine. Before capture() runs, ``graphs`` is empty and
        # the smallest configured bucket is returned unchanged.
        if not self.graphs:
            return self.capture_bs[bisect.bisect_left(self.capture_bs, n)]
        for bs in self.capture_bs[bisect.bisect_left(self.capture_bs, n) :]:
            if bs in self.graphs:
                return bs
        return None

    def can_replay(self, n_rows: int) -> bool:
        bs = self.pad_bs(n_rows)
        return bs is not None and bs in self.graphs

    def can_replay_expand(self, n_rows: int) -> bool:
        bs = self.pad_bs(n_rows)
        bw = self.expand_bw
        return (
            bs is not None
            and bw is not None
            and int(n_rows) % int(bw) == 0
            and bs in self.graphs_expand
        )

    def can_replay_cascade(self, n_rows: int, n_reqs: Optional[int] = None) -> bool:
        bs = self.pad_bs(n_rows)
        if bs is None or bs not in self.graphs_cascade:
            return False
        if n_reqs is None:
            return True
        max_reqs = int(getattr(self.attn, "_max_cascade_reqs", 0) or 0)
        if max_reqs <= 0:
            return False
        need = int(n_reqs) + (1 if int(n_rows) < int(bs) else 0)
        return need <= max_reqs

    def can_replay_cascade_expand(self, n_rows: int) -> bool:
        bs = self.pad_bs(n_rows)
        bw = self.expand_bw
        return (
            bs is not None
            and bw is not None
            and int(n_rows) % int(bw) == 0
            and bs in self.graphs_cascade_expand
        )

    def buffers_for(self, n_rows: int) -> Optional[tuple[int, dict]]:
        bs = self.pad_bs(n_rows)
        if bs is None or bs not in self.static:
            return None
        return bs, self.static[bs]

    def _clone_fused(
        self, buf: dict, n_live: int, *, steal: Optional[bool] = None
    ) -> GenrecFusedResult:
        sl = slice(0, int(n_live))
        pingpong = bool(buf.get("exp_pingpong"))
        phase = int(buf.get("exp_phase", 0)) if pingpong else 0
        if pingpong:
            tok = buf["exp_tok_out"] if phase == 0 else buf["exp_tok_in"]
            nodes = buf["exp_node_out"] if phase == 0 else buf["exp_nodes"]
            token_plane = tok[sl]
            node_plane = nodes[sl]
        else:
            token_plane = buf["exp_tok_out"][sl]
            node_plane = buf["exp_node_out"][sl]
        # n=1 has no other request to ping-pong against; stealing the graph
        # plane lets the next conc=1 replay overwrite the only live SID table.
        if steal is None:
            steal = pingpong and int(n_live) >= 2
        if steal:
            token_ids = token_plane
            node_ids = node_plane
        else:
            token_ids = token_plane.clone()
            node_ids = node_plane.clone()
        return GenrecFusedResult(
            vals=buf["exp_vals"][sl].clone(),
            parents=buf["exp_parents"][sl].clone(),
            tokens=buf["exp_tokens"][sl].clone(),
            indices=buf["exp_indices"][sl].clone(),
            token_ids=token_ids,
            node_ids=node_ids,
        )

    def replay(
        self,
        batch: ForwardBatch,
        prepare_fn: Callable[[ForwardBatch], None],
        skip_copy: bool = False,
        want_expand: bool = False,
        cascade: bool = False,
    ) -> Tuple[torch.Tensor, Optional[GenrecFusedResult]]:
        raw_bs = int(batch.n_rows)
        bs = self.pad_bs(raw_bs)
        if cascade:
            if bs is None or bs not in self.graphs_cascade:
                raise RuntimeError(f"no cascade CUDA graph for bs={raw_bs}")
        elif bs is None or bs not in self.graphs:
            raise RuntimeError(f"no CUDA graph for bs={raw_bs}")
        if cascade:
            use_expand = bool(want_expand) and bs in self.graphs_cascade_expand
        else:
            use_expand = bool(want_expand) and bs in self.graphs_expand
        buf = self.static[bs]
        sb = buf["batch"]
        if not skip_copy:
            buf["input_ids"][:raw_bs].copy_(
                batch.input_ids.view(-1)[:raw_bs], non_blocking=True
            )
            buf["seq_lens"][:raw_bs].copy_(
                batch.seq_lens.view(-1)[:raw_bs], non_blocking=True
            )
            buf["positions"][:raw_bs].copy_(
                batch.positions.view(-1)[:raw_bs], non_blocking=True
            )
            buf["out_loc"][:raw_bs].copy_(
                batch.out_cache_loc.view(-1)[:raw_bs], non_blocking=True
            )
            buf["req_pool"][:raw_bs].copy_(
                batch.req_pool_indices.view(-1)[:raw_bs], non_blocking=True
            )
            sl_cpu = batch.seq_lens_cpu.view(-1)[:raw_bs]
            if sl_cpu.device.type != "cpu":
                sl_cpu = sl_cpu.detach().to("cpu")
            # copy_ converts dtype in place; an explicit .to() would allocate
            # a CPU temp per step whenever the producer dtype differs.
            buf["seq_lens_cpu"][:raw_bs].copy_(sl_cpu)
            nidx = int(batch.kv_indices.numel()) if batch.kv_indices is not None else 0
            if nidx > 0:
                src = batch.kv_indices.view(-1)[:nidx]
                if src.dtype != torch.int32:
                    src = src.to(dtype=torch.int32)
                if src.data_ptr() != buf["kv_indices"].data_ptr():
                    buf["kv_indices"][:nidx].copy_(src, non_blocking=True)
        else:
            nidx = int(batch.kv_indices.numel()) if batch.kv_indices is not None else 0
        pad = bs - raw_bs
        # Fast path: when raw_bs hasn't shrunk since the last replay at this
        # padded bs, pad rows are still valid zeros — skip 6 tiny kernels.
        dirty_rows = int(buf.get("dirty_rows", bs))
        last_raw = int(buf.get("last_raw_bs", 0))
        skip_pad = pad > 0 and raw_bs >= last_raw and dirty_rows <= raw_bs
        if pad > 0 and not skip_pad and dirty_rows > raw_bs:
            end = min(dirty_rows, bs)
            buf["input_ids"][raw_bs:end].zero_()
            buf["seq_lens"][raw_bs:end].fill_(1)
            buf["seq_lens_cpu"][raw_bs:end].fill_(1)
            buf["positions"][raw_bs:end].zero_()
            buf["out_loc"][raw_bs:end].zero_()
            buf["req_pool"][raw_bs:end].zero_()
        buf["dirty_rows"] = raw_bs
        buf["last_raw_bs"] = raw_bs
        total = nidx + pad
        kv_dirty = int(buf.get("kv_dirty", int(buf["kv_indices"].numel())))
        last_nidx = int(buf.get("last_nidx", 0))
        skip_kv_pad = pad > 0 and nidx >= last_nidx and kv_dirty <= nidx
        if pad > 0 and not skip_kv_pad and kv_dirty > nidx:
            buf["kv_indices"][
                nidx : max(total, min(kv_dirty, int(buf["kv_indices"].numel())))
            ].zero_()
        buf["kv_dirty"] = nidx
        buf["last_nidx"] = nidx
        sb.kv_indices = buf["kv_indices"][:total]
        prev_cascade = sb.cascade
        if cascade:
            sb.cascade = batch.cascade
            self.attn.begin_graph_cascade_decode(bs)
        else:
            self.attn.begin_graph_decode(bs)
        try:
            prepare_fn(sb)
            if use_expand:
                phase = int(buf.get("exp_phase", 0)) if buf.get("exp_pingpong") else 0
                if cascade:
                    flip_graphs = self.graphs_cascade_expand_flip
                    expand_graphs = self.graphs_cascade_expand
                else:
                    flip_graphs = self.graphs_expand_flip
                    expand_graphs = self.graphs_expand
                if phase == 1 and bs in flip_graphs:
                    flip_graphs[bs].replay()
                else:
                    expand_graphs[bs].replay()
            elif cascade:
                self.graphs_cascade[bs].replay()
            else:
                self.graphs[bs].replay()
        finally:
            if cascade:
                self.attn.end_graph_cascade_decode()
                sb.cascade = prev_cascade
            else:
                self.attn.end_graph_decode()
        self.raw_bs = raw_bs
        fused = None
        if use_expand and self.expand_bw:
            n_live = int(raw_bs) // int(self.expand_bw)
            fused = self._clone_fused(buf, n_live)
            if buf.get("exp_pingpong"):
                buf["exp_phase"] = 1 - int(buf.get("exp_phase", 0))
        logprobs = buf.get("logprobs")
        if logprobs is not None:
            if fused is not None:
                return logprobs[:raw_bs], fused
            return logprobs[:raw_bs].clone(), None
        return buf["hidden"][:raw_bs], fused
