"""CLI for FlashRec offline generate / HTTP serve."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from flashrec.config import BeamRecConfig, parse_int_list
from flashrec.scheduler.scheduler import BeamRecEngine


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flashrec",
        description="FlashRec: beam-search engine for generative recommendation",
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--messages-json", default=None)
    p.add_argument("--beam-width", "--n", dest="beam_width", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=5)
    p.add_argument("--quantization", default="fp8")
    p.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    # Production: --model-path + --sid-vocab-file; layout inferred from tokenizer.
    # --sid / the three split flags override inference.
    p.add_argument(
        "--sid-vocab-file",
        default=None,
        help="Valid-SID catalog (JSON). With --model-path, infers SID layout "
        "from the tokenizer. Unset = unconstrained full-vocab decode.",
    )
    p.add_argument(
        "--sid",
        default=None,
        help="Optional layout override START:END/SIZE,... when the tokenizer "
        "does not use <s_a_0> / <|sid_begin|>.",
    )
    p.add_argument("--sid-token-range", default=None)
    p.add_argument("--sid-codebook-sizes", default=None)
    p.add_argument("--sid-boundary-tokens", default=None)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--system-prompt-file", default=None)
    p.add_argument(
        "--warmup-user-a",
        default=None,
        help="Radix-warmup user text A. Set with --warmup-user-b to pin a "
        "shared user-head; unset pins template + system prompt only.",
    )
    p.add_argument("--warmup-user-b", default=None)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--length-penalty", type=float, default=1.0)
    p.add_argument("--cuda-graph-max-bs", type=int, default=800)
    p.add_argument("--mem-fraction-static", type=float, default=0.8)
    p.add_argument("--attention-backend", default="flashinfer")
    p.add_argument("--flashinfer-variant", default="fa2")
    p.add_argument("--disable-radix", action="store_true")
    p.add_argument("--disable-warmup", action="store_true")
    p.add_argument("--disable-cuda-graph", action="store_true")
    p.add_argument("--disable-prefill-batch", action="store_true")
    p.add_argument("--disable-fused-expand", action="store_true")
    p.add_argument("--disable-graph-expand", action="store_true")
    p.add_argument("--disable-decode-pack", action="store_true")
    p.add_argument("--disable-fused-rms-fp8", action="store_true")
    p.add_argument("--disable-fused-silu-fp8", action="store_true")
    p.add_argument("--disable-fused-qk-rope-kv", action="store_true")
    # 0 = pipeline off (default); N>=1 = N-stage decode pipelining.
    p.add_argument("--pipeline-stages", type=int, default=0)
    # Default: follow --cuda-graph-max-bs.
    p.add_argument("--batch-slots", type=int, default=None)
    p.add_argument("--batch-wait-ms", type=int, default=4)
    p.add_argument("--batch-wait-max-ms", type=int, default=10)
    # Keep waiting (up to --batch-wait-max-ms) until this many requests are
    # queued; stop admitting new requests into a wave at the max.
    p.add_argument("--target-batch-requests", type=int, default=8)
    p.add_argument("--max-batch-requests", type=int, default=16)
    p.add_argument("--max-running-requests", type=int, default=64)
    p.add_argument("--cuda-graph-capture-sizes", default=None)
    p.add_argument("--decode-pack-min-requests", type=int, default=6)
    p.add_argument("--decode-pack-ratio", type=float, default=0.75)
    p.add_argument("--schedule-policy", default="lpm", choices=("lpm", "fcfs"))
    p.add_argument("--lpm-aging-ms", type=int, default=300)
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--torch-compile-mode", default=None)
    p.add_argument("--host-worker-threads", type=int, default=4)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--serve", action="store_true")
    # No authentication: only bind a public address on a trusted network.
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="info")
    return p


def config_from_args(args: argparse.Namespace) -> BeamRecConfig:
    cfg = BeamRecConfig(
        model_path=args.model_path,
        mem_fraction_static=args.mem_fraction_static,
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        attention_backend=args.attention_backend,
        flashinfer_variant=args.flashinfer_variant,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        gpu_id=args.gpu_id,
        max_running_requests=args.max_running_requests,
        sid=args.sid,
        sid_token_range=args.sid_token_range,
        sid_vocab_file=args.sid_vocab_file,
        sid_codebook_sizes=args.sid_codebook_sizes,
        sid_boundary_tokens=args.sid_boundary_tokens,
        system_prompt=args.system_prompt,
        system_prompt_file=args.system_prompt_file,
        warmup_user_a=args.warmup_user_a,
        warmup_user_b=args.warmup_user_b,
        max_seq_len=args.max_seq_len,
        length_penalty=args.length_penalty,
        beam_width=args.beam_width,
        max_tokens=args.max_tokens,
        enable_radix=not args.disable_radix,
        enable_warmup=not args.disable_warmup,
        enable_cuda_graph=not args.disable_cuda_graph,
        enable_prefill_batch=not args.disable_prefill_batch,
        enable_fused_expand=not args.disable_fused_expand,
        enable_graph_expand=not args.disable_graph_expand,
        enable_decode_pack=not args.disable_decode_pack,
        enable_fused_rms_fp8=not args.disable_fused_rms_fp8,
        enable_fused_silu_fp8=not args.disable_fused_silu_fp8,
        enable_fused_qk_rope_kv=not args.disable_fused_qk_rope_kv,
        pipeline_stages=args.pipeline_stages,
        batch_slots=args.batch_slots,
        batch_wait_ms=args.batch_wait_ms,
        batch_wait_max_ms=args.batch_wait_max_ms,
        target_batch_requests=args.target_batch_requests,
        max_batch_requests=args.max_batch_requests,
        decode_pack_min_requests=args.decode_pack_min_requests,
        decode_pack_ratio=args.decode_pack_ratio,
        schedule_policy=args.schedule_policy,
        lpm_aging_ms=args.lpm_aging_ms,
        enable_torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
        host_worker_threads=args.host_worker_threads,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    sizes = parse_int_list(args.cuda_graph_capture_sizes)
    if sizes:
        cfg.cuda_graph_capture_sizes = sizes
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = config_from_args(args)
    if args.serve:
        from flashrec.server.api import serve

        serve(cfg)
        return 0
    engine = BeamRecEngine(cfg)
    messages = json.loads(args.messages_json) if args.messages_json else None
    result = engine.generate(
        prompt=args.prompt,
        messages=messages,
        n=args.beam_width,
        max_tokens=args.max_tokens,
    )
    payload = {
        "text": result.text,
        "output_ids": result.output_ids,
        "meta_info": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cached_tokens": result.cached_tokens,
            "beam_results": [
                {
                    "text": s.text,
                    "output_ids": s.tokens,
                    "meta_info": {
                        "finish_reason": s.finish_reason,
                        "sequence_score": s.beam_score,
                    },
                }
                for s in result.sequences
            ],
        },
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
