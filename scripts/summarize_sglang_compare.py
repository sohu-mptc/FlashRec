#!/usr/bin/env python3
"""Emit FlashRec vs SGLang QPS / latency / speedup from a matrix run.

Reads ``matrix_summary.json`` produced by
``scripts/run_sglang_flashrec_matrix.sh`` (or globs ``*/n*_c*/summary.json``)
and writes ``SPEED_COMPARE.md`` next to it.

    python scripts/summarize_sglang_compare.py results/onerec_beam_conc_matrix_<stamp>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_rows(root: Path) -> list[dict[str, Any]]:
    summary = root / "matrix_summary.json"
    if summary.is_file():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/n*_c*/summary.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        metrics = cell.get("metrics") or {}
        rows.append(
            {
                "engine": cell.get("engine"),
                "beam": cell.get("beam_size"),
                "conc": cell.get("concurrency"),
                "samples": cell.get("samples"),
                "qps": cell.get("qps"),
                "wall_s": cell.get("wall_seconds"),
                "p50_s": cell.get("latency_p50_s"),
                "p99_s": cell.get("latency_p99_s"),
                "invalid_rate": cell.get("invalid_rate"),
                "recall@32": metrics.get("recall@32"),
                "path": str(path.parent.relative_to(root)),
            }
        )
    return rows


def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        engine = str(row.get("engine") or "")
        try:
            beam = int(row["beam"])
            conc = int(row["conc"])
        except (KeyError, TypeError, ValueError):
            continue
        out[(engine, beam, conc)] = row
    return out


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _speedup(mini_qps: Any, sgl_qps: Any) -> float | None:
    try:
        mini = float(mini_qps)
        sgl = float(sgl_qps)
    except (TypeError, ValueError):
        return None
    if sgl <= 0 or mini <= 0:
        return None
    return mini / sgl


def render(rows: list[dict[str, Any]], *, source: str = "") -> str:
    by_cell = _index(rows)
    beams = sorted({beam for _, beam, _ in by_cell})
    concs = sorted({conc for _, _, conc in by_cell})
    speedups: list[float] = []
    qps_lines = [
        "| beam | conc | FlashRec QPS | SGLang QPS | 加速比 | FlashRec samples | SGLang samples |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lat_lines = [
        "| beam | conc | FlashRec p50 (s) | SGLang p50 (s) | FlashRec p99 (s) | SGLang p99 (s) | FlashRec wall (s) | SGLang wall (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    missing: list[str] = []
    for beam in beams:
        for conc in concs:
            mini = by_cell.get(("flashrec", beam, conc))
            sgl = by_cell.get(("sglang", beam, conc))
            if mini is None and sgl is None:
                continue
            ratio = _speedup(
                None if mini is None else mini.get("qps"),
                None if sgl is None else sgl.get("qps"),
            )
            if ratio is not None:
                speedups.append(ratio)
            if sgl is None:
                missing.append(f"sglang n={beam} conc={conc}")
            if mini is None:
                missing.append(f"flashrec n={beam} conc={conc}")
            ratio_s = "—" if ratio is None else f"{ratio:.2f}×"
            qps_lines.append(
                "| {beam} | {conc} | {mq} | {sq} | {sp} | {mn} | {sn} |".format(
                    beam=beam,
                    conc=conc,
                    mq=_fmt(None if mini is None else mini.get("qps"), 3),
                    sq=_fmt(None if sgl is None else sgl.get("qps"), 3),
                    sp=ratio_s,
                    mn="" if mini is None else mini.get("samples"),
                    sn="" if sgl is None else sgl.get("samples"),
                )
            )
            lat_lines.append(
                "| {beam} | {conc} | {mp50} | {sp50} | {mp99} | {sp99} | {mw} | {sw} |".format(
                    beam=beam,
                    conc=conc,
                    mp50=_fmt(None if mini is None else mini.get("p50_s"), 3),
                    sp50=_fmt(None if sgl is None else sgl.get("p50_s"), 3),
                    mp99=_fmt(None if mini is None else mini.get("p99_s"), 3),
                    sp99=_fmt(None if sgl is None else sgl.get("p99_s"), 3),
                    mw=_fmt(None if mini is None else mini.get("wall_s"), 1),
                    sw=_fmt(None if sgl is None else sgl.get("wall_s"), 1),
                )
            )

    geo = (
        math.exp(sum(math.log(x) for x in speedups) / len(speedups))
        if speedups
        else None
    )
    lines = [
        "# FlashRec vs SGLang 速度对比",
        "",
    ]
    if source:
        lines += [
            f"- 数据源：`{source}`",
            f"- 成对格子：{len(speedups)}（两侧都有 QPS 才算加速比）",
            "",
        ]
    if speedups:
        lines += [
            f"已完成格子上 FlashRec 相对 SGLang 的吞吐为 **{min(speedups):.2f}×–{max(speedups):.2f}×**"
            f"（几何平均 **{geo:.2f}×**）。",
            "",
        ]
    lines += ["## QPS", "", *qps_lines, "", "## 延迟 / 墙钟", "", *lat_lines, ""]
    if missing:
        lines += [
            "## 缺失格子",
            "",
            "以下格子没有 `summary.json`，加速比留空：",
            "",
            *[f"- `{name}`" for name in missing],
            "",
        ]
    lines += [
        "## 说明",
        "",
        "- 加速比 = FlashRec QPS / SGLang QPS，同一 `beam × concurrency`。",
        "- FlashRec 默认 SID trie；SGLang 为本仓库矩阵脚本启动的开放词表 beam。",
        "- 本表只比速度。FlashRec 在 concurrency=1 时存在 beam 塌缩，不要用那一格的 RecIF 质量。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "matrix_dir",
        type=Path,
        help="run_sglang_flashrec_matrix.sh 的输出目录",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="写入路径（默认 <matrix_dir>/SPEED_COMPARE.md）",
    )
    args = parser.parse_args()
    root = args.matrix_dir.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    rows = _load_rows(root)
    text = render(rows, source=str(root))
    out = args.output or (root / "SPEED_COMPARE.md")
    out.write_text(text, encoding="utf-8")
    print(text, end="" if text.endswith("\n") else "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
