"""CUDA Graph capture/replay for decode.

FlashInfer ``plan()`` runs *outside* the graph (SGLang replay_prepare). The
captured region is the model forward, optional restricted LM-head, and optional
row top-k + fused trie expand. Replay pads to the next captured batch size.

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
        self.static: Dict[int, dict] = {}
        self._pool = None
        self.enabled = device.type == "cuda"
        self.raw_bs = 0
        self.captures_logprobs = False
        self._logprobs_k: Optional[int] = None
        self.expand_spec: Optional[ExpandCaptureSpec] = None
        self.expand_bw: Optional[int] = None

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
            buf["logprobs"] = torch.empty(
                bs, int(logprobs_k), dtype=torch.float32, device=device
            )
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

    def pad_bs(self, n_rows: int) -> Optional[int]:
        if not self.enabled or not self.capture_bs:
            return None
        n = int(n_rows)
        if n <= 0 or n > self.capture_bs[-1]:
            return None
        return self.capture_bs[bisect.bisect_left(self.capture_bs, n)]

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

    def buffers_for(self, n_rows: int) -> Optional[tuple[int, dict]]:
        bs = self.pad_bs(n_rows)
        if bs is None or bs not in self.static:
            return None
        return bs, self.static[bs]

    def _clone_fused(self, buf: dict, n_live: int) -> GenrecFusedResult:
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
    ) -> Tuple[torch.Tensor, Optional[GenrecFusedResult]]:
        raw_bs = int(batch.n_rows)
        bs = self.pad_bs(raw_bs)
        if bs is None or bs not in self.graphs:
            raise RuntimeError(f"no CUDA graph for bs={raw_bs}")
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
            buf["seq_lens_cpu"][:raw_bs].copy_(sl_cpu.to(dtype=torch.int64))
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
        if pad > 0:
            buf["input_ids"][raw_bs:bs].zero_()
            buf["seq_lens"][raw_bs:bs].fill_(1)
            buf["seq_lens_cpu"][raw_bs:bs].fill_(1)
            buf["positions"][raw_bs:bs].zero_()
            buf["out_loc"][raw_bs:bs].zero_()
            buf["req_pool"][raw_bs:bs].zero_()
        total = nidx + pad
        if pad > 0:
            buf["kv_indices"][nidx:total].zero_()
        sb.kv_indices = buf["kv_indices"][:total]
        sb.extend_seq_lens = [1] * bs
        self.attn.begin_graph_decode(bs)
        try:
            prepare_fn(sb)
            if use_expand:
                phase = int(buf.get("exp_phase", 0)) if buf.get("exp_pingpong") else 0
                if phase == 1 and bs in self.graphs_expand_flip:
                    self.graphs_expand_flip[bs].replay()
                else:
                    self.graphs_expand[bs].replay()
            else:
                self.graphs[bs].replay()
        finally:
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
