#!/usr/bin/env python3
"""Fold per-variant RecIF cells into one ablation report.

    python benchmark/recif/summarize_ablation.py results/ablation_<stamp>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted(root.glob("*/flashrec/n*_c*/summary.json")):
        variant = summary.parent.parent.parent.name
        cell = json.loads(summary.read_text(encoding="utf-8"))
        metrics = cell.get("metrics") or {}
        rows.append(
            {
                "variant": variant,
                "beam": int(cell.get("beam_size") or 0),
                "conc": int(cell.get("concurrency") or 0),
                "samples": cell.get("samples"),
                "qps": cell.get("qps"),
                "p50_s": cell.get("latency_p50_s"),
                "p90_s": cell.get("latency_p90_s"),
                "p99_s": cell.get("latency_p99_s"),
                "invalid_rate": cell.get("invalid_rate"),
                "duplicate_rate": cell.get("duplicate_rate"),
                "mean_unique_candidates": cell.get("mean_unique_candidates"),
                "collapsed_rate": cell.get("collapsed_rate"),
                "recall@32": metrics.get("recall@32"),
                "ndcg@32": metrics.get("ndcg@32"),
                "path": str(summary.parent.relative_to(root)),
            }
        )
    rows.sort(key=lambda r: (r["variant"], r["beam"], r["conc"]))
    return rows


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _pivot(rows: list[dict[str, Any]], field: str) -> tuple[list[str], list[tuple[int, int]], dict]:
    variants = sorted({r["variant"] for r in rows})
    cells = sorted({(r["beam"], r["conc"]) for r in rows})
    table: dict[tuple[str, int, int], Any] = {}
    for r in rows:
        table[(r["variant"], r["beam"], r["conc"])] = r.get(field)
    return variants, cells, table


def _md_pivot(
    rows: list[dict[str, Any]],
    field: str,
    digits: int,
    *,
    baseline: str = "baseline",
    speedup: bool = False,
) -> list[str]:
    variants, cells, table = _pivot(rows, field)
    header = "| variant | " + " | ".join(f"n={b} c={c}" for b, c in cells) + " |"
    sep = "|---|" + "|".join("---:" for _ in cells) + "|"
    lines = [header, sep]
    for variant in variants:
        vals = []
        for beam, conc in cells:
            raw = table.get((variant, beam, conc))
            if speedup:
                base = table.get((baseline, beam, conc))
                try:
                    b = float(base)
                    v = float(raw)
                    vals.append(_fmt(v / b, digits) + "×" if b > 0 else "")
                except (TypeError, ValueError):
                    vals.append("")
            else:
                vals.append(_fmt(raw, digits))
        lines.append("| " + variant + " | " + " | ".join(vals) + " |")
    return lines


def render(rows: list[dict[str, Any]], *, source: str = "") -> str:
    lines = ["# FlashRec RecIF ablation", ""]
    if source:
        lines += [f"- output: `{source}`", f"- cells: {len(rows)}", ""]
    lines += ["## QPS", ""]
    lines += _md_pivot(rows, "qps", 2)
    lines += ["", "## QPS vs baseline", ""]
    lines += _md_pivot(rows, "qps", 2, speedup=True)
    lines += ["", "## recall@32", ""]
    lines += _md_pivot(rows, "recall@32", 4)
    lines += ["", "## latency p50 (s)", ""]
    lines += _md_pivot(rows, "p50_s", 3)
    lines += ["", "## invalid_rate / unique beams", ""]
    lines += [
        "| variant | beam | conc | QPS | p50(s) | recall@32 | invalid | dup | mean_uniq | collapsed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {variant} | {beam} | {conc} | {qps} | {p50} | {rec} | {inv} | {dup} | {uniq} | {col} |".format(
                variant=r["variant"],
                beam=r["beam"],
                conc=r["conc"],
                qps=_fmt(r["qps"], 2),
                p50=_fmt(r["p50_s"], 3),
                rec=_fmt(r["recall@32"], 4),
                inv=_fmt(r["invalid_rate"], 4),
                dup=_fmt(r["duplicate_rate"], 4),
                uniq=_fmt(r["mean_unique_candidates"], 2),
                col=_fmt(r["collapsed_rate"], 4),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = _load_rows(root)
    payload = {"root": str(root), "cells": rows}
    (root / "ablation_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    report = render(rows, source=str(root))
    (root / "ABLATION_REPORT.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    by: dict[str, int] = defaultdict(int)
    for r in rows:
        by[r["variant"]] += 1
    print("variants:", dict(by))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
