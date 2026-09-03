"""Model runner: load Qwen3, KV pool, extend/decode, optional CUDA graph."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from flashrec.attention.flashinfer import AttentionBackend
from flashrec.config import BeamRecConfig
from flashrec.core import ForwardBatch
from flashrec.engine.graph import DecodeGraphRunner, ExpandCaptureSpec
from flashrec.kernel.beam_trie import GenrecFusedResult
from flashrec.kvcache.pool import ReqToTokenPool, TokenToKVPool
from flashrec.kvcache.radix import PrefixCache
from flashrec.logits import RestrictedLMHead
from flashrec.models.qwen3 import Qwen3ForCausalLM
from flashrec.models.weight import load_hf_config, load_weights
from flashrec.sid_layout import resolve_sid_layout

logger = logging.getLogger(__name__)


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "fp8_e4m3": torch.float8_e4m3fn,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }.get(name, torch.bfloat16)


class ModelEngine:
    def __init__(self, config: BeamRecConfig):
        resolve_sid_layout(config)
        self.config = config
        self.device = torch.device(
            f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.compute_dtype = _dtype(config.compute_dtype)
        kv_dtype = (
            _dtype(config.kv_cache_dtype)
            if config.quantization == "fp8"
            else self.compute_dtype
        )
        if kv_dtype not in (
            torch.float8_e4m3fn,
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            kv_dtype = self.compute_dtype

        self.model_config = load_hf_config(config.model_path)
        cfg = self.model_config
        free_mem = 0
        if self.device.type == "cuda":
            free_mem, _ = torch.cuda.mem_get_info(self.device)

        # One KV pool. Prefer native FP8 pages (SGLang --kv-cache-dtype fp8_e4m3).
        token_budget = 65536
        if free_mem > 0:
            bytes_per = (
                2 * cfg.num_key_value_heads * cfg.head_dim * cfg.num_hidden_layers
            )
            if kv_dtype == torch.float8_e4m3fn:
                bytes_per *= 1
            elif kv_dtype == torch.float16 or kv_dtype == torch.bfloat16:
                bytes_per *= 2
            else:
                bytes_per *= 4
            token_budget = max(
                int(free_mem * config.mem_fraction_static * 0.45) // max(bytes_per, 1),
                4096,
            )

        try:
            self.kv_pool = TokenToKVPool(
                num_layers=cfg.num_hidden_layers,
                num_tokens=token_budget,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                dtype=kv_dtype,
                device=self.device,
            )
        except Exception:
            logger.warning(
                "KV alloc dtype=%s failed, falling back to %s",
                kv_dtype,
                self.compute_dtype,
            )
            kv_dtype = self.compute_dtype
            self.kv_pool = TokenToKVPool(
                num_layers=cfg.num_hidden_layers,
                num_tokens=token_budget,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                dtype=kv_dtype,
                device=self.device,
            )

        self.req_pool = ReqToTokenPool(
            max_reqs=max(
                int(config.max_running_requests) * max(int(config.beam_width), 1) + 8,
                1024,
            ),
            max_seq_len=int(config.max_seq_len),
            device=self.device,
        )
        self.prefix_cache = (
            PrefixCache(self.kv_pool, self.device) if config.enable_radix else None
        )
        self.attn = AttentionBackend(
            self.kv_pool,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            backend=config.attention_backend,
            fi_backend=getattr(config, "flashinfer_variant", "fa2"),
        )
        self.attn.q_dtype = self.compute_dtype
        self.model = Qwen3ForCausalLM(
            cfg,
            self.attn,
            self.device,
            self.compute_dtype,
            fused_rms_fp8=bool(getattr(config, "enable_fused_rms_fp8", True)),
            fused_silu_fp8=bool(getattr(config, "enable_fused_silu_fp8", True)),
            fused_qk_rope_kv=bool(getattr(config, "enable_fused_qk_rope_kv", True)),
        )
        load_weights(
            self.model,
            config.model_path,
            self.device,
            self.compute_dtype,
            quantize_fp8=config.quantization == "fp8",
        )
        self.model.eval()
        # Compiled clone for prefill only: decode replays CUDA graphs, and the
        # capture path must stay on the eager module. Dynamic shapes because
        # every wave has a different extend-token count.
        self._compiled_model = None
        if bool(getattr(config, "enable_torch_compile", False)):
            try:
                self._compiled_model = torch.compile(
                    self.model,
                    dynamic=True,
                    fullgraph=False,
                    mode=getattr(config, "torch_compile_mode", None),
                )
                logger.info(
                    "torch.compile enabled for prefill (mode=%s)",
                    getattr(config, "torch_compile_mode", None) or "default",
                )
            except Exception:
                logger.exception("torch.compile setup failed; eager prefill")
                self._compiled_model = None
        self.lm_head = RestrictedLMHead(
            config.parsed_sid_token_ids(),
            enabled=config.enable_restricted_lm_head,
        )
        try:
            self.lm_head.bind(self.model.lm_head())
        except Exception:
            logger.debug("restricted lm_head bind deferred", exc_info=True)
        self.graph: Optional[DecodeGraphRunner] = None
        self._graph_ready = False
        self.profiler = None
        if config.enable_cuda_graph and self.device.type == "cuda":
            logger.info("CUDA graph capture deferred to the decode worker thread")

        logger.info(
            "ModelEngine ready: layers=%d hidden=%d tokens=%d kv_dtype=%s",
            cfg.num_hidden_layers,
            cfg.hidden_size,
            token_budget,
            self.kv_pool.dtype,
        )

    def alloc_tokens(self, n: int) -> torch.Tensor:
        loc = self.kv_pool.alloc(n)
        if loc is None and self.prefix_cache is not None:
            need = max(int(n), min(4096, int(self.prefix_cache.num_cached_tokens)))
            self.prefix_cache.evict(need)
            loc = self.kv_pool.alloc(n)
        if loc is None:
            raise RuntimeError(f"out of KV tokens (need {n})")
        return loc

    def alloc_tokens_cpu(self, n: int) -> tuple[torch.Tensor, List[int]]:
        loc = self.alloc_tokens(n)
        return loc, list(self.kv_pool.last_alloc_cpu)

    def alloc_req_slots(self, n: int) -> torch.Tensor:
        idx = self.req_pool.alloc(n)
        if idx is None:
            raise RuntimeError(f"out of req slots (need {n})")
        return idx

    def alloc_req_one(self) -> int:
        t = self.alloc_req_slots(1)
        cpu = self.req_pool.last_alloc_cpu
        if cpu:
            return int(cpu[0])
        return int(t.reshape(-1)[0].item())

    def free_tokens(self, indices) -> None:
        self.kv_pool.free(indices)

    def free_req_slots(self, indices) -> None:
        self.req_pool.free(indices)

    def write_prefill(self, req_idx: int, kv_indices: torch.Tensor) -> None:
        n = int(kv_indices.numel())
        if n <= 0:
            return
        self.req_pool.req_to_token[int(req_idx), :n] = kv_indices.to(
            dtype=self.req_pool.req_to_token.dtype
        )

    def copy_prefill_to_beams(
        self, src: int, beam_rows: torch.Tensor, seq_len: int
    ) -> None:
        table = self.req_pool.req_to_token
        src_i = int(src)
        table[beam_rows.to(dtype=torch.int64), :seq_len] = table[
            src_i : src_i + 1, :seq_len
        ]

    def copy_prefill_to_beams_many(
        self, srcs: List[int], beam_rows: torch.Tensor, seq_lens: List[int]
    ) -> None:
        """Batched ``copy_prefill_to_beams``: one gather/scatter for a wave.

        Copies ``max(seq_lens)`` columns for every request. Columns past a
        request's own seq_len carry stale values that are never read (readers
        slice ``[:seq_len]``) and decode overwrites them in step order.
        """
        table = self.req_pool.req_to_token
        n, bw = beam_rows.shape
        m = max(int(s) for s in seq_lens)
        dst = beam_rows.to(dtype=torch.int64).reshape(-1)
        src = torch.as_tensor(
            [int(s) for s in srcs], dtype=torch.int64, device=table.device
        ).repeat_interleave(bw)
        table[dst, :m] = table[src, :m]

    def gather_kv_indices(
        self, rows: torch.Tensor, seq_lens, out=None, total: Optional[int] = None
    ) -> torch.Tensor:
        from flashrec.attention.kv_indices import gather_kv_indices as _gather

        return _gather(self.req_pool.req_to_token, rows, seq_lens, out=out, total=total)

    def _model_fwd(self, batch: ForwardBatch) -> torch.Tensor:
        return self.model(batch)

    def last_token_hidden(
        self, hidden: torch.Tensor, batch: ForwardBatch
    ) -> torch.Tensor:
        if not batch.is_prefill:
            return hidden
        ext = batch.extend_seq_lens
        if not ext:
            return hidden
        t = torch.as_tensor(list(ext), device=hidden.device, dtype=torch.int64)
        return hidden[torch.cumsum(t, dim=0) - 1]

    def ensure_cuda_graph(
        self, expand_spec: Optional[ExpandCaptureSpec] = None
    ) -> None:
        """Capture FlashInfer decode graphs on the *current* thread/stream.

        Must run on the worker that will replay. Failed captures are skipped;
        eager decode remains correct.
        """
        if self._graph_ready:
            return
        self._graph_ready = True
        if not self.config.enable_cuda_graph:
            return
        if self.device.type != "cuda" or self.attn._fi_decode is None:
            logger.info("CUDA graph skipped (no CUDA FlashInfer decode)")
            return
        capture_bs = self.config.resolved_cuda_graph_sizes()
        if not capture_bs:
            return
        torch.cuda.set_device(self.device)
        try:
            self.attn.init_graph_wrappers(capture_bs, int(self.config.max_seq_len))
        except Exception as exc:
            logger.warning("FlashInfer CUDA-graph wrappers failed: %s", exc)
            return
        runner = DecodeGraphRunner(
            device=self.device,
            max_bs=int(self.config.cuda_graph_max_bs),
            max_seq_len=int(self.config.max_seq_len),
            capture_bs=capture_bs,
            attn=self.attn,
        )
        logits_fn = None
        logprobs_k = None
        if self.lm_head.enabled:
            try:
                self.lm_head.bind(self.model.lm_head())
            except Exception:
                logger.debug(
                    "restricted lm_head bind before graph failed", exc_info=True
                )
            if self.lm_head.ready and self.lm_head.num_tokens:

                def _logits_fn(hidden: torch.Tensor, out: torch.Tensor) -> None:
                    self.lm_head.compute_into(hidden, self.model.lm_head(), out)

                logits_fn = _logits_fn
                logprobs_k = int(self.lm_head.num_tokens)
        with torch.inference_mode():
            runner.capture(
                self._model_fwd,
                self.attn.prepare,
                logits_fn=logits_fn,
                logprobs_k=logprobs_k,
                expand_spec=expand_spec if self.config.enable_graph_expand else None,
            )
        if runner.graphs:
            self.graph = runner
            logger.info(
                "CUDA graph ready: captured=%s lm_head=%s k=%s expand=%s",
                sorted(runner.graphs),
                runner.captures_logprobs,
                logprobs_k,
                sorted(runner.graphs_expand) if runner.graphs_expand else False,
            )
        else:
            logger.warning("CUDA graph capture produced no graphs; eager decode")

    def forward(
        self, batch: ForwardBatch
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[GenrecFusedResult]
    ]:
        if self.profiler is not None:
            self.profiler.on_forward(is_prefill=bool(batch.is_prefill))
        try:
            with torch.inference_mode():
                use_graph = (
                    self.graph is not None
                    and not batch.is_prefill
                    and self.graph.can_replay(batch.n_rows)
                )
                if use_graph:
                    want_expand = bool(getattr(batch, "want_expand", False)) and (
                        self.graph.can_replay_expand(batch.n_rows)
                    )
                    out, fused = self.graph.replay(
                        batch,
                        self.attn.prepare,
                        skip_copy=bool(batch.buffers_ready),
                        want_expand=want_expand,
                    )
                    if fused is not None:
                        return None, self.lm_head.token_ids, fused
                    if self.graph.captures_logprobs:
                        return out, self.lm_head.token_ids, None
                    hidden = out
                else:
                    self.attn.prepare(batch)
                    if batch.is_prefill and self._compiled_model is not None:
                        hidden = self._compiled_model(batch)
                    else:
                        hidden = self.model(batch)
                last = self.last_token_hidden(hidden, batch)
                logprobs, cands = self.lm_head.compute(last, self.model.lm_head())
                return logprobs, cands, None
        finally:
            if self.profiler is not None:
                self.profiler.after_forward(is_prefill=bool(batch.is_prefill))
