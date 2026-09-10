#!/usr/bin/env python3
"""Emit FlashRec RecIF matrix report (QPS / latency / quality).

Reads ``matrix_summary.json`` if present, otherwise globs
``*/n*_c*/summary.json`` (prefers ``flashrec/`` cells), and writes
``MATRIX_REPORT.md`` + ``matrix_summary.json`` next to the run.

    python benchmark/recif/summarize_compare.py results/onerec_beam_conc_bench_<stamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(root.glob("flashrec/n*_c*/summary.json"))
    if not paths:
        paths = sorted(root.glob("*/n*_c*/summary.json"))
    for path in paths:
        cell = json.loads(path.read_text(encoding="utf-8"))
        engine = str(cell.get("engine") or path.parent.parent.name)
        if engine != "flashrec" and path.parent.parent.name != "flashrec":
            # Skip leftover baseline cells from older compare runs.
            continue
        metrics = cell.get("metrics") or {}
        rows.append(
            {
                "engine": "flashrec",
                "beam": cell.get("beam_size"),
                "conc": cell.get("concurrency"),
                "samples": cell.get("samples"),
                "qps": cell.get("qps"),
                "wall_s": cell.get("wall_seconds"),
                "p50_s": cell.get("latency_p50_s"),
                "p90_s": cell.get("latency_p90_s"),
                "p99_s": cell.get("latency_p99_s"),
                "invalid_rate": cell.get("invalid_rate"),
                "duplicate_rate": cell.get("duplicate_rate"),
                "mean_unique_candidates": cell.get("mean_unique_candidates"),
                "collapsed_rate": cell.get("collapsed_rate"),
                "recall@32": metrics.get("recall@32"),
                "ndcg@32": metrics.get("ndcg@32"),
                "mrr@32": metrics.get("mrr@32"),
                "hit@32": metrics.get("hit@32"),
                "path": str(path.parent.relative_to(root)),
            }
        )
    rows.sort(
        key=lambda r: (int(r["beam"] or 0), int(r["conc"] or 0), str(r.get("path") or ""))
    )
    return rows


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render(rows: list[dict[str, Any]], *, source: str = "") -> str:
    lines = [
        "# OneRec × RecIF video — FlashRec",
        "",
    ]
    if source:
        lines += [f"- output: `{source}`", f"- cells: {len(rows)}", ""]
    lines += [
        "| beam | conc | samples | QPS | wall(s) | p50(s) | p90(s) | p99(s) | recall@32 | ndcg@32 | invalid_rate | dup_rate | mean_uniq | collapsed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {beam} | {conc} | {samples} | {qps} | {wall} | {p50} | {p90} | {p99} | {rec} | {ndcg} | {inv} | {dup} | {uniq} | {col} |".format(
                beam=r["beam"],
                conc=r["conc"],
                samples=r["samples"],
                qps=_fmt(r.get("qps"), 3),
                wall=_fmt(r.get("wall_s"), 1),
                p50=_fmt(r.get("p50_s"), 3),
                p90=_fmt(r.get("p90_s"), 3),
                p99=_fmt(r.get("p99_s"), 3),
                rec=_fmt(r.get("recall@32"), 5),
                ndcg=_fmt(r.get("ndcg@32"), 5),
                inv=_fmt(r.get("invalid_rate"), 4),
                dup=_fmt(r.get("duplicate_rate"), 4),
                uniq=_fmt(r.get("mean_unique_candidates"), 1),
                col=_fmt(r.get("collapsed_rate"), 4),
            )
        )
    lines += [
        "",
        "## 说明",
        "",
        "- FlashRec 默认 SID trie + FP8；指标来自 `python -m flashrec.benchmark.recif`。",
        "- concurrency=1 时存在 unique-beam 塌缩，不要单独引用那一格的 RecIF 质量。",
        "- `dup_rate` = 平均 `(n - n_unique) / n`；`collapsed` = unique≤1 的样本占比。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "matrix_dir",
        type=Path,
        help="benchmark/recif 矩阵脚本的输出目录",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="写入路径（默认 <matrix_dir>/MATRIX_REPORT.md）",
    )
    args = parser.parse_args()
    root = args.matrix_dir.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    rows = _load_rows(root)
    (root / "matrix_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    text = render(rows, source=str(root))
    out = args.output or (root / "MATRIX_REPORT.md")
    out.write_text(text, encoding="utf-8")
    print(text, end="" if text.endswith("\n") else "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
