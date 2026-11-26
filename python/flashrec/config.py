"""Engine config. Ablation flags default to production-on."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union


def parse_int_list(value: Optional[Union[str, Sequence[int]]]) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        return None
    if ":" in text and "," not in text:
        start, end = text.split(":", 1)
        lo, hi = int(start), int(end)
        if hi < lo:
            raise ValueError(f"empty token range: {text}")
        return list(range(lo, hi + 1))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_sid_spec(value: str) -> Tuple[str, str, str]:
    """Parse compact SID layout ``START:END/SIZE,...[@BEGIN,END]``.

    Returns ``(token_range, codebook_sizes, boundary_tokens)`` as strings.
    Boundary tokens are the last codebook when its size is 2 (RecIF 4-level
    ``...,2``); otherwise they are the two ids immediately after ``END``.
    Optional ``@BEGIN,END`` overrides that pair. SoHu-0.6B's packed 512×3
    window keeps ``<|sid_begin|>`` / ``<|sid_end|>`` outside the window.
    """
    text = str(value).strip()
    if "/" not in text:
        raise ValueError(
            "sid spec must be START:END/SIZE,... e.g. 151669:176246/8192,8192,8192,2"
        )
    range_s, sizes_s = text.split("/", 1)
    range_s = range_s.strip()
    sizes_s = sizes_s.strip()
    boundary_override: Optional[str] = None
    if "@" in sizes_s:
        sizes_s, boundary_override = sizes_s.split("@", 1)
        sizes_s = sizes_s.strip()
        boundary_override = boundary_override.strip()
    if ":" not in range_s or "," in range_s:
        raise ValueError(f"sid spec range must be START:END, got {range_s!r}")
    start_s, end_s = range_s.split(":", 1)
    start, end = int(start_s), int(end_s)
    if end < start:
        raise ValueError(f"empty token range: {range_s}")
    sizes = [int(x.strip()) for x in sizes_s.split(",") if x.strip()]
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError("sid spec codebook sizes must be positive integers")
    need = sum(sizes)
    n_ids = end - start + 1
    if need > n_ids:
        raise ValueError(f"sid spec codebook sum ({need}) exceeds range size ({n_ids})")
    if boundary_override:
        if boundary_override.lower() == "none":
            # No begin/end wrapper: the codebook tokens are the whole
            # generation (MiniOneRec emits 3 tokens, nothing around them).
            b0 = b1 = None
        else:
            parts = [p.strip() for p in boundary_override.split(",") if p.strip()]
            if len(parts) != 2:
                raise ValueError(
                    f"sid spec @boundary must be BEGIN,END or none, got {boundary_override!r}"
                )
            b0, b1 = int(parts[0]), int(parts[1])
    elif sizes[-1] == 2:
        b0 = start + need - 2
        b1 = start + need - 1
    else:
        b0, b1 = end + 1, end + 2
    sizes_norm = ",".join(str(s) for s in sizes)
    boundary = "" if b0 is None else f"{b0},{b1}"
    return f"{start}:{end}", sizes_norm, boundary


def _sid_ids_key(value: Union[str, Sequence[int]]) -> Tuple:
    """Equality key for a SID id field without expanding huge ranges."""
    if isinstance(value, str):
        text = value.strip()
        if ":" in text and "," not in text:
            start_s, end_s = text.split(":", 1)
            lo, hi = int(start_s), int(end_s)
            return ("range", lo, hi)
        ids = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    else:
        ids = tuple(int(x) for x in value)
    if len(ids) > 1 and ids[-1] == ids[0] + len(ids) - 1:
        return ("range", ids[0], ids[-1])
    return ("list", ids)


def same_sid_ids(
    left: Union[str, Sequence[int]], right: Union[str, Sequence[int]]
) -> bool:
    return _sid_ids_key(left) == _sid_ids_key(right)


@dataclass
class FlashRecConfig:
    model_path: str
    mem_fraction_static: float = 0.8
    # "fp8" (W8A8 per-channel). "nvfp4" is reserved for an official checkpoint.
    quantization: Optional[str] = "fp8"
    kv_cache_dtype: str = "fp8_e4m3"
    compute_dtype: str = "bfloat16"
    attention_backend: str = "flashinfer"
    flashinfer_variant: str = "fa2"
    cuda_graph_max_bs: int = 800
    gpu_id: int = 0
    max_running_requests: int = 64
    max_seq_len: int = 4096

    # SID (semantic id) vocabulary. Production serving sets sid_vocab_file;
    # token range / codebook sizes / boundary are then inferred from the
    # tokenizer (<s_a_0> codebooks + <|sid_begin|> / <|sid_end|>). Left
    # unset, the engine runs unconstrained (full-vocab lm_head, no SID trie).
    # sid "START:END/SIZE,..." and the three split fields override inference.
    sid: Optional[str] = None
    # Closed range "start:end" or comma list. Inferred, or set to override.
    sid_token_range: Optional[Union[str, Sequence[int]]] = None
    sid_vocab_file: Optional[str] = None
    # Per-level codebook sizes, e.g. "8192,8192,8192" for a depth-3 SID.
    sid_codebook_sizes: Optional[Union[str, Sequence[int]]] = None
    # "begin,end" token ids wrapping a SID. Inferred, or set to override.
    sid_boundary_tokens: Optional[Union[str, Sequence[int]]] = None
    system_prompt: Optional[str] = None
    system_prompt_file: Optional[str] = None
    # Optional radix-warmup probes. Two distinct user texts; their LCP is
    # pinned. Unset = pin chat template + system prompt only.
    warmup_user_a: Optional[str] = None
    warmup_user_b: Optional[str] = None
    # End-to-end warmup generations run by the serve worker after CUDA-graph
    # capture and before the HTTP port opens. 0 disables.
    warmup_iters: int = 10

    beam_width: int = 50
    max_tokens: int = 5
    length_penalty: float = 1.0
    # None: follow cuda_graph_max_bs (slot budget exists to fill the largest graph).
    batch_slots: Optional[int] = None
    batch_wait_ms: int = 4
    batch_wait_max_ms: int = 10
    # 0 = pipeline off; N>=1 enables N-stage decode pipelining.
    pipeline_stages: int = 0
    # Admission sizing: keep waiting (bounded by batch_wait_max_ms) until
    # target_batch_requests are queued; stop admitting new requests into a
    # wave at max_batch_requests.
    target_batch_requests: int = 8
    max_batch_requests: int = 16
    decode_pack_min_requests: int = 6
    decode_pack_ratio: float = 0.75

    enable_radix: bool = True
    enable_cuda_graph: bool = True
    enable_restricted_lm_head: bool = True
    enable_fused_expand: bool = True
    enable_graph_expand: bool = True
    # Unary finish skips GPU decode for forced trie suffixes. Each appended
    # token contributes zero to the cumulative log probability.
    enable_unary_finish: bool = True
    # FR-legal normalizes over each parent's legal trie children. The default
    # FR-full normalizes over the current SID codebook before catalog masking.
    # The different normalizers can change the ranking across parents.
    enable_sparse_lm_head: bool = False
    sparse_max_outdeg: int = 256
    # Fused CUDA scorer for the sparse path: reads lm_head rows in place rather
    # than materializing [rows, outdeg, H], and pads to the trie's static max
    # outdeg so no host sync is needed to size the plane. Falls back to the
    # gather + matmul path when the JIT module is unavailable.
    enable_fused_sparse_score: bool = True
    # Coverage-prefix beam budget (the "budget" in trie-budget beam search):
    # each decode step keeps only the smallest model-ranked prefix of
    # survivors covering `tau` of the reachable leaf mass. Commit 86e5129
    # removed this rule — it cut wide beams (200/500/1000 returned
    # 192/480/928) and cost recall without buying throughput — so it is OFF
    # by default here and exists only so the same commit can run both arms of
    # that A/B. Nothing in production should set it.
    enable_trie_budget: bool = False
    # Floor (NOT a cap) of the per-level coverage prefix. The prefix itself is
    # decided by trie_budget_tau over the model-ranked candidates, clamped to
    # [max(n0, n_min), beam_width]. The old cap semantics truncated n=128/512
    # first levels to 64 unranked survivors and collapsed recall.
    trie_budget_n0: int = 64
    trie_budget_tau: float = 0.9
    trie_budget_n_max: int = 0  # 0 = beam_width
    trie_budget_n_min: int = 16
    # Leaf flush enumerates catalog suffixes in DFS order with tied parent
    # scores — not a model ranking — so it is opt-in. 0 = off.
    trie_flush_leaves: int = 0
    enable_prefill_batch: bool = True
    enable_warmup: bool = True
    enable_decode_pack: bool = True
    # Beam-shared attention (BSA) reuses prompt KV reads across beam rows
    # through FlashInfer multi-level cascade attention.
    # When enabled, decode tries a dedicated cascade CUDA graph (padded
    # MultiLevelCascade wrapper); missing/oversized batches stay on eager
    # cascade. cascade_min_rows gates per batch: batches with fewer total
    # beam rows keep the paged-graph path.
    enable_cascade_attention: bool = False
    cascade_min_rows: int = 128
    # Per-level CUDA graphs return before cascade metadata is built. Turn
    # this off when the experiment needs decode to reach paged or cascade
    # attention; the per-codebook lm_head itself stays available.
    enable_level_graph: bool = True
    # Per-head Q/K RMSNorm. "auto" derives it from the checkpoint (on for
    # Qwen3, off for Qwen2); "on"/"off" force it. MiniOneRec is Qwen2 and its
    # two published HR@10 levels (0.158 vs 0.05) differ exactly on this flag,
    # so the override exists to run the A/B rather than to pick a side.
    qk_norm: str = "auto"
    enable_kv_admission: bool = True
    # row: the slot budget is the sum of beam widths.
    # request: every request charges admission_request_rows against that same
    # budget (0 charges the real width, which collapses to row accounting).
    admission_unit: str = "row"
    admission_request_rows: int = 0
    # Deadline-aware admission gate (P1-2). OFF by default: with
    # admission_slo_budget_ms<=0 the gate never drops and select_jobs_lpm
    # output is byte-for-byte unchanged. The gate is orthogonal to
    # admission_unit -- it reasons about wall-clock feasibility (queue age +
    # estimated service time vs the SLO budget), not the slot charge -- so the
    # same rule applies to both the row and request accounting arms.
    admission_slo_budget_ms: int = 0  # per-request end-to-end SLO budget; 0 = gate off
    admission_deadline_slack_ms: int = 0  # safety margin added to the estimate
    admission_svc_base_ms: float = 0.0  # service-time estimator intercept
    admission_svc_per_row_ms: float = 0.0  # estimator slope in beam width (job.n)
    admission_svc_queue_ahead_ms: float = (
        0.0  # estimator slope in rows ahead (0 = ignore)
    )
    admission_deadline_source: str = (
        "server"  # "server" (global budget) | "client" (per-request)
    )
    enable_fused_rms_fp8: bool = True
    enable_fused_silu_fp8: bool = True
    enable_fused_qk_rope_kv: bool = True
    enable_next_norm_fusion: bool = False
    enable_per_codebook_lm_head: bool = True
    schedule_policy: str = "lpm"
    # Promote jobs waiting longer than this ahead of the LPM prefix ranking.
    # 0 disables aging (pure LPM, which can starve short prompts).
    lpm_aging_ms: int = 300
    # torch.compile the prefill model forward (decode stays on CUDA graphs).
    enable_torch_compile: bool = False
    torch_compile_mode: Optional[str] = None
    host_worker_threads: int = 4

    # Note: the server has no authentication; only bind a public address on a
    # trusted network or behind a proxy.
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    # uvicorn access log formats one line per request on the event-loop
    # thread; off by default for serving throughput.
    access_log: bool = False
    # Reject new requests with 503 once this many jobs are queued (0 = no
    # cap). Bounds memory and tail latency under overload.
    max_queue_len: int = 4096
    cuda_graph_capture_sizes: List[int] = field(
        default_factory=lambda: [50, 100, 150, 200, 250, 300, 350, 400, 800]
    )
    # Full-heap collection can stall HTTP ingress and decode for hundreds of ms.
    idle_gc_full: bool = True

    def __post_init__(self) -> None:
        self.apply_sid_spec()
        unit = str(self.admission_unit or "row").lower()
        if unit not in ("row", "request"):
            raise ValueError(f"admission_unit must be row or request, got {unit!r}")
        self.admission_unit = unit
        request_rows = int(self.admission_request_rows)
        if request_rows < 0:
            raise ValueError("admission_request_rows must be non-negative")
        self.admission_request_rows = request_rows
        if unit == "request" and request_rows > self.resolved_batch_slots():
            raise ValueError(
                "admission_request_rows cannot exceed the slot budget "
                f"({request_rows} > {self.resolved_batch_slots()})"
            )
        source = str(self.admission_deadline_source or "server").lower()
        if source not in ("server", "client"):
            raise ValueError(
                f"admission_deadline_source must be server or client, got {source!r}"
            )
        self.admission_deadline_source = source
        self.admission_slo_budget_ms = max(int(self.admission_slo_budget_ms), 0)
        self.admission_deadline_slack_ms = max(int(self.admission_deadline_slack_ms), 0)
        for name in (
            "admission_svc_base_ms",
            "admission_svc_per_row_ms",
            "admission_svc_queue_ahead_ms",
        ):
            val = float(getattr(self, name))
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val!r}")
            setattr(self, name, val)

    def deadline_gate_on(self) -> bool:
        """True when the deadline-aware admission gate is active."""
        return int(self.admission_slo_budget_ms) > 0

    def apply_sid_spec(self) -> None:
        """Expand ``sid`` into token range / codebook sizes / boundary tokens."""
        if not self.sid:
            return
        token_range, codebook_sizes, boundary = parse_sid_spec(self.sid)

        def _fill(
            name: str,
            current: Optional[Union[str, Sequence[int]]],
            inferred: str,
        ) -> None:
            if current is None:
                setattr(self, name, inferred)
                return
            if not same_sid_ids(current, inferred):
                raise ValueError(
                    f"{name}={current!r} conflicts with --sid {self.sid!r} "
                    f"(inferred {inferred})"
                )

        _fill("sid_token_range", self.sid_token_range, token_range)
        _fill("sid_codebook_sizes", self.sid_codebook_sizes, codebook_sizes)
        _fill("sid_boundary_tokens", self.sid_boundary_tokens, boundary)

    def parsed_sid_token_ids(self) -> Optional[List[int]]:
        return parse_int_list(self.sid_token_range)

    @property
    def enable_pipeline(self) -> bool:
        return int(self.pipeline_stages) > 0

    def resolved_batch_slots(self) -> int:
        if self.batch_slots is not None:
            return max(int(self.batch_slots), 1)
        return max(int(self.cuda_graph_max_bs), 1)

    def resolved_cuda_graph_sizes(self) -> List[int]:
        """Capture sizes, widened to multiples of the beam width.

        A beam wave is always k*n rows, and the fused-expand graph can only be
        captured when bs % n == 0 (DecodeGraphRunner._alloc_expand_bufs). With
        the plain list a wide beam (n=512) matches no multiple, so expand falls
        back to eager launches and the batch pads to the next larger size.
        """
        max_bs = max(int(self.cuda_graph_max_bs), 1)
        sizes = {int(b) for b in self.cuda_graph_capture_sizes if 0 < int(b) <= max_bs}
        bw = int(self.beam_width)
        if bw > 0:
            # k concurrent beams, bounded by what the slot budget can admit and
            # by the soft admit cap.
            k_max = min(self.resolved_batch_slots() // bw, self.soft_admit_max_reqs())
            for k in range(1, max(k_max, 1) + 1):
                if bw * k <= max_bs:
                    sizes.add(bw * k)
        return sorted(sizes)

    def preferred_batch_sizes(self) -> List[int]:
        target = max(int(self.target_batch_requests), 1)
        cap = max(int(self.max_batch_requests), target)
        return sorted({target, cap})

    def target_admit_reqs(self) -> int:
        return max(int(self.target_batch_requests), 1)

    def soft_admit_max_reqs(self) -> int:
        return max(int(self.max_batch_requests), self.target_admit_reqs())

    def parsed_codebook_sizes(self) -> Optional[List[int]]:
        return parse_int_list(self.sid_codebook_sizes)

    def parsed_boundary_token_ids(self) -> Optional[List[int]]:
        ids = parse_int_list(self.sid_boundary_tokens)
        if ids is not None and len(ids) != 2:
            raise ValueError("sid_boundary_tokens must be begin,end")
        return ids

    def system_prompt_text(self) -> Optional[str]:
        if self.system_prompt:
            return self.system_prompt
        if self.system_prompt_file:
            path = Path(self.system_prompt_file)
            if path.is_file():
                return path.read_text(encoding="utf-8").strip()
        return None


BeamRecConfig = FlashRecConfig
