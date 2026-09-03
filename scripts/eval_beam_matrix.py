#!/usr/bin/env python3
"""RecIF video eval client for FlashRec and SGLang beam search.

Measures wall-clock throughput plus Recall / NDCG / MRR / hit / invalid_rate
at a given beam width and request concurrency.

Protocols:
  FlashRec / vLLM / TRT-LLM: POST /v1/chat/completions with ``n`` = beam width
  SGLang (PR #31626):    POST /generate with sampling_params.beam_width
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

SID_TOKEN_RE = re.compile(r"<s_[abc]_\d+>")
SID_BLOCK_RE = re.compile(r"<\|sid_begin\|>(.*?)<\|sid_end\|>")
RECIF_CODE_BASE = 8192
KS_BASE = (1, 4, 8, 16, 32, 50, 128, 512)


def normalize_sid(value: Any) -> str:
    return "".join(SID_TOKEN_RE.findall(str(value)))


def compact_sid_to_tokens(sid_value: Any) -> list[str]:
    text = str(sid_value).strip()
    explicit = SID_TOKEN_RE.findall(text)
    if explicit:
        return explicit
    if not text.isdigit():
        return []
    encoded = int(text)
    first, remainder = divmod(encoded, RECIF_CODE_BASE * RECIF_CODE_BASE)
    second, third = divmod(remainder, RECIF_CODE_BASE)
    values = (first, second, third)
    if any(v < 0 or v >= RECIF_CODE_BASE for v in values):
        return []
    return [f"<s_a_{first}>", f"<s_b_{second}>", f"<s_c_{third}>"]


def extract_sid(text: str) -> str:
    toks = SID_TOKEN_RE.findall(text or "")[:3]
    return "".join(toks)


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        converted.append({"role": message.get("role"), "content": content})
    return converted


def load_samples(
    data_dir: Path,
    task: str,
    sample_size: int,
    sample_mode: str,
    seed: int,
    sample_offset: int,
) -> list[dict[str, Any]]:
    import pandas as pd

    parquet_path = data_dir / task / f"{task}_test.parquet"
    frame = pd.read_parquet(parquet_path, columns=["metadata", "messages"])
    n_rows = len(frame)
    if n_rows == 0:
        raise ValueError(f"empty parquet: {parquet_path}")

    if sample_mode == "random":
        rng = random.Random(seed)
        order = list(range(n_rows))
        rng.shuffle(order)
        indices = order[:sample_size]
    else:
        if not (0 <= sample_offset < n_rows):
            raise ValueError(f"sample_offset={sample_offset} outside {n_rows} rows")
        indices = list(range(sample_offset, min(sample_offset + sample_size, n_rows)))

    samples: list[dict[str, Any]] = []
    for row_idx in indices:
        row = frame.iloc[row_idx]
        metadata_raw, messages_raw = row["metadata"], row["messages"]
        metadata = (
            json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
        )
        messages = (
            json.loads(messages_raw) if isinstance(messages_raw, str) else messages_raw
        )
        if not isinstance(metadata, dict) or not isinstance(messages, list):
            continue
        converted = _convert_messages(messages)
        answer = str(metadata.get("answer", ""))
        blocks = SID_BLOCK_RE.findall(answer)
        ground_truth = list(
            dict.fromkeys(sid for sid in (normalize_sid(b) for b in blocks) if sid)
        )
        if converted and ground_truth:
            samples.append(
                {
                    "id": str(int(row_idx)),
                    "messages": converted,
                    "ground_truth_sids": ground_truth,
                }
            )
        if len(samples) >= sample_size:
            break
    if not samples:
        raise RuntimeError("no valid RecIF samples after filtering")
    return samples


def load_catalog(mapping_path: Path) -> set[str]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    catalog: set[str] = set()
    for packed in mapping:
        text = str(packed)
        if "," in text:
            parts = text.split(",")
            if len(parts) >= 3:
                catalog.add(f"<s_a_{parts[0]}><s_b_{parts[1]}><s_c_{parts[2]}>")
                continue
        toks = compact_sid_to_tokens(packed)
        if len(toks) == 3:
            catalog.add("".join(toks))
    return catalog


def load_samples_json(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    for item in raw:
        gt = item.get("ground_truth_sids") or []
        messages = item.get("messages") or []
        if messages and gt:
            samples.append(
                {
                    "id": str(item.get("id")),
                    "messages": messages,
                    "ground_truth_sids": list(gt),
                }
            )
    if not samples:
        raise RuntimeError(f"no samples in {path}")
    return samples


def _choice_score(ch: Mapping[str, Any]) -> float | None:
    ext = ch.get("sglext")
    if isinstance(ext, Mapping):
        score = ext.get("sequence_score")
        if score is not None:
            return float(score)
    score = ch.get("sequence_score")
    return None if score is None else float(score)


def uses_sglang_generate(engine: str, protocol: str) -> bool:
    if protocol == "generate":
        return True
    if protocol == "chat":
        return False
    return engine.replace("-", "_").endswith("sglang") or engine == "sglang"


def render_chat_prompt(messages: list[dict[str, Any]], tokenizer: Any) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def parse_sglang_generate(
    payload: Mapping[str, Any], n: int
) -> list[tuple[str, float | None]]:
    out: list[tuple[str, float | None]] = []
    beams = (payload.get("meta_info") or {}).get("beam_results") or []
    if isinstance(beams, list) and beams:
        for item in beams:
            if not isinstance(item, dict):
                continue
            txt = item.get("text") or ""
            score = None
            meta = item.get("meta_info")
            if isinstance(meta, dict) and meta.get("sequence_score") is not None:
                score = float(meta["sequence_score"])
            out.append((extract_sid(str(txt)), score))
    elif "text" in payload:
        text = payload["text"]
        texts = text if isinstance(text, list) else [text]
        out = [(extract_sid(str(item)), None) for item in texts]
    return out[:n]


def _post_json(
    url: str,
    body: Mapping[str, Any],
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    data = json.dumps(body).encode()
    last_err: Exception | None = None
    for attempt in range(max(retries, 1)):
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.load(resp)
            if not isinstance(payload, dict):
                raise json.JSONDecodeError("expected object", "", 0)
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            last_err = err
            err_s = str(err)
            if "111" in err_s or "Connection refused" in err_s:
                break
            time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise RuntimeError(f"request failed after {retries} tries: {last_err}")


def request_beams(
    server_url: str,
    messages: list[dict[str, Any]],
    n: int,
    max_tokens: int,
    timeout: float,
    retries: int,
    *,
    engine: str = "",
    protocol: str = "auto",
    tokenizer: Any = None,
) -> list[tuple[str, float | None]]:
    base = server_url.rstrip("/")
    if uses_sglang_generate(engine, protocol):
        if tokenizer is None:
            raise ValueError("SGLang /generate needs a tokenizer (--model-path)")
        payload = _post_json(
            f"{base}/generate",
            {
                "text": render_chat_prompt(messages, tokenizer),
                "sampling_params": {
                    "beam_width": n,
                    "n": n,
                    "max_new_tokens": max_tokens,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
            },
            timeout,
            retries,
        )
        return parse_sglang_generate(payload, n)

    payload = _post_json(
        f"{base}/v1/chat/completions",
        {
            "model": "onerec",
            "messages": messages,
            "n": n,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
        retries,
    )
    out: list[tuple[str, float | None]] = []
    for ch in payload.get("choices") or []:
        if not isinstance(ch, Mapping):
            continue
        txt = (ch.get("message") or {}).get("content") or ""
        out.append((extract_sid(str(txt)), _choice_score(ch)))
    return out


def hit_recall(beams: list[str], gt: set[str], k: int) -> tuple[float, float]:
    top = beams[:k]
    hits = sum(1 for s in top if s in gt)
    return (1.0 if hits else 0.0), hits / max(len(gt), 1)


def ndcg_at_k(beams: list[str], gt: set[str], k: int) -> float:
    dcg = 0.0
    for i, sid in enumerate(beams[:k], start=1):
        if sid in gt:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(k, len(gt))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def mrr_at_k(beams: list[str], gt: set[str], k: int) -> float:
    for i, sid in enumerate(beams[:k], start=1):
        if sid in gt:
            return 1.0 / i
    return 0.0


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--server-url", required=True)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--samples-json", type=Path, default=None)
    ap.add_argument("--catalog", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--task", default="video")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--sample-mode", choices=("random", "slice"), default="random")
    ap.add_argument("--sample-offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument(
        "--protocol",
        choices=("auto", "chat", "generate"),
        default="auto",
        help="auto: SGLang uses POST /generate; others use /v1/chat/completions",
    )
    ap.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="tokenizer path; required for SGLang /generate chat template",
    )
    args = ap.parse_args()

    tokenizer = None
    if uses_sglang_generate(args.engine, args.protocol):
        if args.model_path is None:
            raise ValueError("--model-path is required for SGLang /generate")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_path), trust_remote_code=True
        )

    if args.catalog is not None:
        catalog_path = args.catalog
    elif args.data_dir is not None:
        mapping_name = "sid2iid.json" if args.task == "product" else "sid2pid.json"
        catalog_path = args.data_dir / mapping_name
    else:
        raise ValueError("need --catalog or --data-dir")
    catalog = load_catalog(catalog_path)
    extra = max(args.warmup, 0)
    if args.samples_json is not None:
        samples = load_samples_json(args.samples_json)
        if len(samples) > args.sample_size + extra:
            samples = samples[: args.sample_size + extra]
    else:
        if args.data_dir is None:
            raise ValueError("need --samples-json or --data-dir")
        samples = load_samples(
            args.data_dir,
            args.task,
            args.sample_size + extra,
            args.sample_mode,
            args.seed,
            args.sample_offset,
        )
    warmup_samples = samples[:extra] if extra else []
    timed_samples = samples[extra : extra + args.sample_size]
    if len(timed_samples) < args.sample_size:
        timed_samples = samples[: args.sample_size]
        warmup_samples = timed_samples[:extra]

    ks = tuple(k for k in KS_BASE if k <= args.n)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[setup] engine={args.engine} n={args.n} conc={args.concurrency} "
        f"samples={len(timed_samples)} warmup={len(warmup_samples)} "
        f"catalog={len(catalog)} url={args.server_url}",
        flush=True,
    )

    def run_one(sample: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        beams = request_beams(
            args.server_url,
            sample["messages"],
            args.n,
            args.max_tokens,
            args.request_timeout,
            args.retries,
            engine=args.engine,
            protocol=args.protocol,
            tokenizer=tokenizer,
        )
        dt = time.perf_counter() - t0
        gt = set(sample["ground_truth_sids"])
        invalid = 0
        ranked: list[tuple[str, float | None]] = []
        for sid, score in beams:
            valid = (
                bool(SID_TOKEN_RE.findall(sid)) and len(SID_TOKEN_RE.findall(sid)) == 3
            )
            if not valid or sid not in catalog:
                invalid += 1
            ranked.append((sid, score))
        return {
            "id": sample["id"],
            "gt": gt,
            "gt_count": len(gt),
            "beams": ranked,
            "invalid": invalid,
            "latency": dt,
        }

    warmup_fail = 0
    for i, sample in enumerate(warmup_samples):
        try:
            run_one(sample)
            print(f"  warmup {i + 1}/{len(warmup_samples)}", flush=True)
        except Exception as err:
            warmup_fail += 1
            print(f"  warmup {i + 1} failed: {err}", flush=True)
    if warmup_samples and warmup_fail == len(warmup_samples):
        raise RuntimeError("all warmup requests failed; aborting cell")

    results: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as ex:
        futs = {ex.submit(run_one, s): s for s in timed_samples}
        done = 0
        errors = 0
        for fut in cf.as_completed(futs):
            done += 1
            try:
                results.append(fut.result())
            except Exception as err:
                errors += 1
                print(f"  error {done}/{len(timed_samples)}: {err}", flush=True)
            if done % 25 == 0 or done == len(timed_samples):
                print(f"  {done}/{len(timed_samples)} done errors={errors}", flush=True)
    wall = time.perf_counter() - t_start
    order = {s["id"]: i for i, s in enumerate(timed_samples)}
    results.sort(key=lambda r: order.get(r["id"], 10**9))
    n = len(results)
    if n == 0:
        raise RuntimeError("all requests failed")

    latencies = sorted(r["latency"] for r in results)
    qps = n / wall if wall > 0 else 0.0
    agg = {f"hit@{k}": 0.0 for k in ks}
    agg.update({f"recall@{k}": 0.0 for k in ks})
    agg.update({f"ndcg@{k}": 0.0 for k in ks})
    agg.update({f"mrr@{k}": 0.0 for k in ks})
    invalid_total = 0
    cand_total = 0
    psm_path = args.out_dir / "per_sample_metrics.csv"
    cand_path = args.out_dir / "candidates.csv"
    lat_path = args.out_dir / "latencies.jsonl"

    with (
        cand_path.open("w", newline="", encoding="utf-8") as cfh,
        psm_path.open("w", newline="", encoding="utf-8") as pfh,
        lat_path.open("w", encoding="utf-8") as lfh,
    ):
        cw = csv.writer(cfh)
        cw.writerow(["sample_id", "rank", "sid", "is_ground_truth", "score"])
        pcols = [
            "sample_id",
            "candidate_count",
            "unique_candidate_count",
            "invalid_rate",
            "ground_truth_count",
            "latency_s",
        ]
        for k in ks:
            pcols += [f"hit@{k}", f"recall@{k}", f"ndcg@{k}", f"mrr@{k}"]
        pw = csv.writer(pfh)
        pw.writerow(pcols)
        for r in results:
            sids = [s for s, _ in r["beams"]]
            gt = r["gt"]
            invalid_total += r["invalid"]
            cand_total += max(len(sids), 1)
            for rank, (sid, score) in enumerate(r["beams"], start=1):
                cw.writerow(
                    [
                        r["id"],
                        rank,
                        sid,
                        1 if sid in gt else 0,
                        score if score is not None else "",
                    ]
                )
            prow = [
                r["id"],
                len(sids),
                len(set(sids)),
                round(r["invalid"] / max(len(sids), 1), 6),
                r["gt_count"],
                round(r["latency"], 6),
            ]
            for k in ks:
                h, rec = hit_recall(sids, gt, k)
                nd = ndcg_at_k(sids, gt, k)
                mr = mrr_at_k(sids, gt, k)
                agg[f"hit@{k}"] += h / n
                agg[f"recall@{k}"] += rec / n
                agg[f"ndcg@{k}"] += nd / n
                agg[f"mrr@{k}"] += mr / n
                prow += [h, rec, nd, mr]
            pw.writerow(prow)
            lfh.write(json.dumps({"id": r["id"], "latency_s": r["latency"]}) + "\n")

    summary = {
        "engine": args.engine,
        "server_url": args.server_url,
        "task": args.task,
        "beam_size": args.n,
        "concurrency": args.concurrency,
        "samples": n,
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "wall_seconds": round(wall, 4),
        "qps": round(qps, 4),
        "latency_mean_s": round(sum(latencies) / n, 6),
        "latency_p50_s": round(_percentile(latencies, 50), 6),
        "latency_p95_s": round(_percentile(latencies, 95), 6),
        "latency_p99_s": round(_percentile(latencies, 99), 6),
        "invalid_rate": round(invalid_total / max(cand_total, 1), 6),
        "metrics": {k: round(v, 6) for k, v in agg.items()},
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rec32 = agg.get("recall@32", agg.get(f"recall@{ks[-1]}", 0.0))
    nd32 = agg.get("ndcg@32", agg.get(f"ndcg@{ks[-1]}", 0.0))
    print(
        "[done] {engine} n={n} c={c} samples={s} wall={w:.1f}s qps={q:.3f} "
        "recall@32={r:.5f} ndcg@32={d:.5f} invalid={inv:.4f} p50={p50:.3f}s".format(
            engine=args.engine,
            n=args.n,
            c=args.concurrency,
            s=n,
            w=wall,
            q=qps,
            r=rec32,
            d=nd32,
            inv=summary["invalid_rate"],
            p50=summary["latency_p50_s"],
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
