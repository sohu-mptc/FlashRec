#!/usr/bin/env python3
"""Build a flashrec SID catalog from OpenOneRec RecIF packed mappings.

RecIF keys are packed integers ``a*8192^2 + b*8192 + c``; ``--sid-vocab-file``
wants comma keys ``"a,b,c"``. Default output is 4-level ``"a,b,c,1"`` where
level 4 is a size-2 codebook {<|sid_begin|>, <|sid_end|>} and code 1 ==
<|sid_end|>. Serve with ``--model-path`` plus ``--sid-vocab-file``; the
engine infers layout from the tokenizer (OneRec-1.7B logs
``Inferred --sid 151669:176246/8192,8192,8192,2``) and generates
[a, b, c, sid_end] with the end-token logprob in sequence_score.

Usage:
  python scripts/convert_recif_catalog.py --data-dir /path/to/benchmark_data
  python scripts/convert_recif_catalog.py sid2pid.json out.json [--levels 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "catalogs"
DEFAULT_BASE = 8192

TASK_FILES = {
    "video": ("sid2pid.json", "sid2pid_beamrec_l{levels}.json"),
    "product": ("sid2iid.json", "sid2iid_beamrec_l{levels}.json"),
}


def parse_sid_codes(
    key: object, *, base: int = DEFAULT_BASE
) -> Optional[tuple[int, int, int]]:
    """Decode a RecIF packed key or an already-split ``a,b,c[,end]`` key."""
    text = str(key).strip()
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        try:
            codes = [int(x) for x in parts]
        except ValueError:
            return None
        if len(codes) < 3:
            return None
        a, b, c = codes[0], codes[1], codes[2]
        if 0 <= a < base and 0 <= b < base and 0 <= c < base:
            return (a, b, c)
        return None
    try:
        packed = int(text)
    except ValueError:
        return None
    if not (0 <= packed < base**3):
        return None
    a = packed // (base * base)
    b = (packed // base) % base
    c = packed % base
    return (a, b, c)


def iter_unique_sids(
    mapping: object, *, base: int = DEFAULT_BASE
) -> tuple[list[tuple[int, int, int]], int, int]:
    if not isinstance(mapping, dict):
        raise ValueError("catalog JSON root must be an object of SID keys")
    seen: set[tuple[int, int, int]] = set()
    bad = 0
    for key in mapping:
        codes = parse_sid_codes(key, base=base)
        if codes is None:
            bad += 1
            continue
        seen.add(codes)
    return sorted(seen), len(mapping), bad


def format_sid_key(codes: tuple[int, int, int], *, levels: int) -> str:
    a, b, c = codes
    if levels == 4:
        return f"{a},{b},{c},1"
    return f"{a},{b},{c}"


def write_sid2vid(
    sids: Iterable[tuple[int, int, int]],
    out_path: Path,
    *,
    levels: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("{")
        first = True
        for codes in sids:
            if not first:
                f.write(",")
            f.write(f'"{format_sid_key(codes, levels=levels)}":1')
            first = False
        f.write("}")


def convert_file(
    src: Path,
    out: Path,
    *,
    levels: int = 4,
    base: int = DEFAULT_BASE,
) -> tuple[int, int, int]:
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    sids, n_in, bad = iter_unique_sids(data, base=base)
    write_sid2vid(sids, out, levels=levels)
    return n_in, len(sids), bad


def _task_jobs(task: str) -> Iterator[tuple[str, str, str]]:
    names = ("video", "product") if task == "both" else (task,)
    for name in names:
        src_name, out_tmpl = TASK_FILES[name]
        yield name, src_name, out_tmpl


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert RecIF packed SID catalogs to flashrec sid2vid JSON."
    )
    ap.add_argument("src", nargs="?", help="sid2pid.json or sid2iid.json")
    ap.add_argument("out", nargs="?", help="output sid2vid JSON")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="RecIF benchmark_data directory (reads sid2pid.json / sid2iid.json)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--task",
        choices=("video", "product", "both"),
        default="video",
        help="which RecIF mapping to convert when using --data-dir",
    )
    ap.add_argument(
        "--levels",
        type=int,
        default=4,
        choices=(3, 4),
        help="3 = plain SID trie; 4 = append sid_end level (default)",
    )
    ap.add_argument(
        "--codebook-base",
        type=int,
        default=DEFAULT_BASE,
        help="RecIF packing base / per-level codebook size (default: 8192)",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    jobs: list[tuple[Path, Path]] = []

    if args.src is not None:
        if args.out is None:
            print(
                "convert_recif_catalog.py: out path required with src",
                file=sys.stderr,
            )
            return 2
        jobs.append((Path(args.src), Path(args.out)))
    elif args.data_dir is not None:
        out_dir = args.out_dir or DEFAULT_OUT_DIR
        for _name, src_name, out_tmpl in _task_jobs(args.task):
            src = args.data_dir / src_name
            out = out_dir / out_tmpl.format(levels=args.levels)
            jobs.append((src, out))
    else:
        print(
            "convert_recif_catalog.py: provide src/out or --data-dir",
            file=sys.stderr,
        )
        return 2

    for src, out in jobs:
        if not src.is_file():
            print(f"missing RecIF mapping: {src}", file=sys.stderr)
            return 1
        n_in, n_out, bad = convert_file(
            src, out, levels=args.levels, base=args.codebook_base
        )
        print(
            f"{src}: {n_in} keys in, {n_out} unique SIDs out, "
            f"{bad} bad keys, levels={args.levels} -> {out}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
