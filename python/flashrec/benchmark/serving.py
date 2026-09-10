"""Online serving throughput for FlashRec wide-beam chat completions.

Mirrors the usage shape of ``python -m sglang.benchmark.serving``, but targets
FlashRec's ``POST /v1/chat/completions`` with ``n`` = beam width.

Examples::

    python -m flashrec.benchmark.serving \\
      --base-url http://127.0.0.1:8000 --n 50 --max-concurrency 32 --num-prompts 200

    python -m flashrec.benchmark.serving \\
      --base-url http://127.0.0.1:8000 --dataset-name file --dataset-path prompts.jsonl \\
      --n 128 --max-concurrency 16 --profile
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from flashrec.benchmark.utils import (
    dumps_json,
    loads_json,
    percentile,
    start_profile,
    stop_profile,
    wait_for_endpoint,
)

DEFAULT_PROMPT = (
    "The user has watched the following videos: "
    "<sid_0><sid_1><sid_2>. Recommend the next video."
)


@dataclass
class RequestResult:
    success: bool = False
    latency: float = 0.0
    error: str = ""
    n_choices: int = 0
    start_time: float = 0.0


@dataclass
class BenchMetrics:
    completed: int = 0
    failed: int = 0
    wall_seconds: float = 0.0
    qps: float = 0.0
    latency_mean_s: float = 0.0
    latency_p50_s: float = 0.0
    latency_p90_s: float = 0.0
    latency_p95_s: float = 0.0
    latency_p99_s: float = 0.0
    latencies: list[float] = field(default_factory=list)


def _build_prompts(
    *,
    dataset_name: str,
    dataset_path: Optional[Path],
    num_prompts: int,
    seed: int,
    input_len: int,
) -> list[str]:
    if dataset_name == "file":
        if dataset_path is None:
            raise ValueError("--dataset-path is required for dataset-name=file")
        prompts: list[str] = []
        text = dataset_path.read_text(encoding="utf-8")
        # JSONL of {"prompt": "..."} / {"messages":[...]} / plain string lines,
        # or a JSON list of the same.
        if text.lstrip().startswith("["):
            items = json.loads(text)
            for item in items:
                prompts.append(_item_to_prompt(item))
        else:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    prompts.append(_item_to_prompt(json.loads(line)))
                except json.JSONDecodeError:
                    prompts.append(line)
        if not prompts:
            raise RuntimeError(f"no prompts in {dataset_path}")
        rng = random.Random(seed)
        while len(prompts) < num_prompts:
            prompts.append(prompts[rng.randrange(len(prompts))])
        return prompts[:num_prompts]

    if dataset_name == "sharegpt":
        # Lightweight stand-in: ShareGPT-shaped files are rare in GenRec setups.
        # Prefer --dataset-name file with RecIF-exported prompts.
        raise ValueError(
            "dataset-name=sharegpt is not bundled; export prompts to JSONL and "
            "use --dataset-name file --dataset-path <path>"
        )

    # random: fixed GenRec template, optionally padded for longer prefill.
    rng = random.Random(seed)
    base = DEFAULT_PROMPT
    if input_len > 0:
        # Approximate token count with whitespace-separated filler tokens.
        filler = " ".join(f"tok{rng.randint(0, 9999)}" for _ in range(input_len))
        base = f"{filler}\n{DEFAULT_PROMPT}"
    return [base for _ in range(num_prompts)]


def _item_to_prompt(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        raise TypeError(f"unsupported prompt item type: {type(item)}")
    if "prompt" in item:
        return str(item["prompt"])
    messages = item.get("messages")
    if isinstance(messages, list) and messages:
        # Use last user message content when present.
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    return "".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                return str(content or "")
        content = messages[-1].get("content") if isinstance(messages[-1], dict) else ""
        return str(content or "")
    raise ValueError("prompt item needs 'prompt' or 'messages'")


async def _request_one(
    session: Any,
    url: str,
    prompt: str,
    *,
    n: int,
    max_tokens: int,
    temperature: float,
    model: str,
    timeout: float,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "n": n,
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = dumps_json(payload)
    out = RequestResult()
    async with semaphore:
        out.start_time = time.perf_counter()
        try:
            async with session.post(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            ) as resp:
                raw = await resp.read()
                if resp.status >= 400:
                    out.error = f"HTTP {resp.status}: {raw[:200]!r}"
                    out.latency = time.perf_counter() - out.start_time
                    return out
                data = loads_json(raw)
        except Exception as err:  # noqa: BLE001 — surface any client failure
            out.error = str(err)
            out.latency = time.perf_counter() - out.start_time
            return out
    out.latency = time.perf_counter() - out.start_time
    if isinstance(data, dict):
        choices = data.get("choices") or []
        out.n_choices = len(choices) if isinstance(choices, list) else 0
        out.success = True
    else:
        out.error = "response is not a JSON object"
    return out


async def _run_async(args: argparse.Namespace, prompts: list[str]) -> BenchMetrics:
    try:
        import aiohttp
    except ImportError as err:
        raise SystemExit(
            "aiohttp is required for flashrec.benchmark.serving; "
            "install with: pip install 'flashrec[eval]' or pip install aiohttp"
        ) from err

    base = args.base_url.rstrip("/")
    url = f"{base}/v1/chat/completions"
    semaphore = asyncio.Semaphore(max(args.max_concurrency, 1))
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=max(args.max_concurrency * 2, 16))

    warmup = max(args.warmup, 0)
    warmup_prompts = prompts[:warmup]
    timed_prompts = prompts[warmup:] if warmup else prompts
    if not timed_prompts:
        timed_prompts = prompts
        warmup_prompts = []

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for i, prompt in enumerate(warmup_prompts):
            result = await _request_one(
                session,
                url,
                prompt,
                n=args.n,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                model=args.model,
                timeout=args.request_timeout,
                semaphore=semaphore,
            )
            status = "ok" if result.success else f"fail:{result.error}"
            print(f"  warmup {i + 1}/{len(warmup_prompts)} {status}", flush=True)

        if args.profile:
            print("[profile] start_profile", flush=True)
            start_profile(base, num_steps=args.profile_steps)

        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                _request_one(
                    session,
                    url,
                    prompt,
                    n=args.n,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    model=args.model,
                    timeout=args.request_timeout,
                    semaphore=semaphore,
                )
            )
            for prompt in timed_prompts
        ]
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

        if args.profile:
            print("[profile] stop_profile", flush=True)
            stop_profile(base)

    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    latencies = sorted(r.latency for r in ok)
    n_ok = len(ok)
    metrics = BenchMetrics(
        completed=n_ok,
        failed=len(fail),
        wall_seconds=wall,
        qps=(n_ok / wall) if wall > 0 else 0.0,
        latency_mean_s=(sum(latencies) / n_ok) if n_ok else float("nan"),
        latency_p50_s=percentile(latencies, 50) if latencies else float("nan"),
        latency_p90_s=percentile(latencies, 90) if latencies else float("nan"),
        latency_p95_s=percentile(latencies, 95) if latencies else float("nan"),
        latency_p99_s=percentile(latencies, 99) if latencies else float("nan"),
        latencies=latencies,
    )
    for r in fail[:5]:
        print(f"  error: {r.error}", flush=True)
    if len(fail) > 5:
        print(f"  ... {len(fail) - 5} more errors", flush=True)
    return metrics


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.wait_health > 0 and not wait_for_endpoint(
        args.base_url, timeout_sec=args.wait_health
    ):
        raise SystemExit(f"server not healthy at {args.base_url}")

    total = args.num_prompts + max(args.warmup, 0)
    prompts = _build_prompts(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        num_prompts=total,
        seed=args.seed,
        input_len=args.random_input_len,
    )
    print(
        f"[setup] url={args.base_url} n={args.n} conc={args.max_concurrency} "
        f"prompts={args.num_prompts} warmup={args.warmup} "
        f"dataset={args.dataset_name}",
        flush=True,
    )
    metrics = asyncio.run(_run_async(args, prompts))
    summary = {
        "backend": "flashrec",
        "base_url": args.base_url,
        "beam_size": args.n,
        "max_concurrency": args.max_concurrency,
        "num_prompts": args.num_prompts,
        "warmup": args.warmup,
        "dataset_name": args.dataset_name,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "completed": metrics.completed,
        "failed": metrics.failed,
        "wall_seconds": round(metrics.wall_seconds, 4),
        "qps": round(metrics.qps, 4),
        "latency_mean_s": round(metrics.latency_mean_s, 6),
        "latency_p50_s": round(metrics.latency_p50_s, 6),
        "latency_p90_s": round(metrics.latency_p90_s, 6),
        "latency_p95_s": round(metrics.latency_p95_s, 6),
        "latency_p99_s": round(metrics.latency_p99_s, 6),
    }
    print(
        "[done] completed={c} failed={f} wall={w:.1f}s qps={q:.3f} "
        "p50={p50:.3f}s p90={p90:.3f}s p95={p95:.3f}s p99={p99:.3f}s".format(
            c=metrics.completed,
            f=metrics.failed,
            w=metrics.wall_seconds,
            q=metrics.qps,
            p50=metrics.latency_p50_s,
            p90=metrics.latency_p90_s,
            p95=metrics.latency_p95_s,
            p99=metrics.latency_p99_s,
        ),
        flush=True,
    )
    if args.result_filename:
        path = Path(args.result_filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {path}", flush=True)
    return summary


def cli_main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="flashrec")
    p.add_argument("--n", type=int, default=50, help="beam width / candidates")
    p.add_argument("--max-tokens", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--wait-health", type=int, default=0, help="seconds; 0 skips")
    p.add_argument(
        "--dataset-name",
        choices=("random", "file", "sharegpt"),
        default="random",
    )
    p.add_argument("--dataset-path", type=Path, default=None)
    p.add_argument(
        "--random-input-len",
        type=int,
        default=0,
        help="extra filler tokens prepended for dataset-name=random",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="call /start_profile before timed requests and /stop_profile after",
    )
    p.add_argument("--profile-steps", type=int, default=None)
    p.add_argument(
        "--result-filename",
        default=None,
        help="optional path to write summary JSON",
    )
    args = p.parse_args()
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
