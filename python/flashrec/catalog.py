"""Build a FlashRec SID catalog from OpenOneRec RecIF packed mappings.

RecIF keys are packed integers ``a*8192^2 + b*8192 + c``; ``--sid-vocab-file``
wants comma keys ``"a,b,c"``. Layer count is inferred from the source file:

* packed integers → unpack in base ``--codebook-base`` (RecIF is 3 layers; a
  value ``>= base^3`` grows the depth). Classic RecIF 3-layer packs also append
  ``,1`` so the catalog is 4-level ``"a,b,c,1"`` (last codebook is
  ``{<|sid_begin|>, <|sid_end|>}``, code 1 == ``<|sid_end|>``).
* comma keys → keep every segment (``"a,b,c"`` stays 3-level; ``"a,b,c,d"``
  stays 4-level with the real last code).

stdlib only: ``flashrec --catalog`` must not import torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

DEFAULT_OUT_DIR = Path("data") / "catalogs"
DEFAULT_BASE = 8192
RECIF_PACKED_DEPTH = 3
MAX_PACKED_DEPTH = 16
SID_END_CODE = 1

TASK_FILES = {
    "video": ("sid2pid.json", "sid2pid_beamrec_l{levels}.json"),
    "product": ("sid2iid.json", "sid2iid_beamrec_l{levels}.json"),
}

ConvertResult = Tuple[Path, int, int, int, int]


def unpack_packed(
    packed: int, *, depth: int, base: int = DEFAULT_BASE
) -> Optional[tuple[int, ...]]:
    if depth < 1 or not (0 <= packed < base**depth):
        return None
    codes: list[int] = []
    value = packed
    for _ in range(depth):
        codes.append(value % base)
        value //= base
    return tuple(reversed(codes))


def packed_depth(max_packed: int, *, base: int = DEFAULT_BASE) -> int:
    """Smallest depth that can hold ``max_packed``, at least RecIF's 3 layers."""
    if max_packed < 0:
        raise ValueError(f"packed SID must be non-negative, got {max_packed}")
    depth = RECIF_PACKED_DEPTH
    limit = base**depth
    while max_packed >= limit:
        depth += 1
        if depth > MAX_PACKED_DEPTH:
            raise ValueError(
                f"packed SID {max_packed} exceeds depth {MAX_PACKED_DEPTH} "
                f"at base {base}"
            )
        limit *= base
    return depth


def parse_sid_codes(
    key: object,
    *,
    base: int = DEFAULT_BASE,
    packed_layers: int = RECIF_PACKED_DEPTH,
) -> Optional[tuple[int, ...]]:
    """Decode a RecIF packed key or a comma-separated codebook path."""
    text = str(key).strip()
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            return None
        try:
            codes = tuple(int(x) for x in parts)
        except ValueError:
            return None
        if any(c < 0 or c >= base for c in codes):
            return None
        return codes
    try:
        packed = int(text)
    except ValueError:
        return None
    return unpack_packed(packed, depth=packed_layers, base=base)


def iter_unique_sids(
    mapping: object, *, base: int = DEFAULT_BASE
) -> tuple[list[tuple[int, ...]], int, int, bool]:
    if not isinstance(mapping, dict):
        raise ValueError("catalog JSON root must be an object of SID keys")
    packed_values: list[int] = []
    comma_keys: list[object] = []
    other = 0
    for key in mapping:
        text = str(key).strip()
        if "," in text:
            comma_keys.append(key)
            continue
        try:
            packed_values.append(int(text))
        except ValueError:
            other += 1

    depth = (
        packed_depth(max(packed_values), base=base)
        if packed_values
        else RECIF_PACKED_DEPTH
    )
    seen: set[tuple[int, ...]] = set()
    bad = other
    for packed in packed_values:
        codes = unpack_packed(packed, depth=depth, base=base)
        if codes is None:
            bad += 1
            continue
        seen.add(codes)
    for key in comma_keys:
        codes = parse_sid_codes(key, base=base)
        if codes is None:
            bad += 1
            continue
        seen.add(codes)
    packed_only = bool(packed_values) and not comma_keys and other == 0
    return sorted(seen), len(mapping), bad, packed_only


def unique_width(sids: Iterable[tuple[int, ...]]) -> int:
    widths = {len(codes) for codes in sids}
    if not widths:
        raise ValueError("no valid SID keys to infer --levels from")
    if len(widths) != 1:
        raise ValueError(
            f"mixed SID depths {sorted(widths)}; pass --catalog-levels to pick one"
        )
    return next(iter(widths))


def infer_levels(sids: list[tuple[int, ...]], *, packed_only: bool) -> int:
    native = unique_width(sids)
    # Classic RecIF pack is 3 codebook layers; FlashRec catalogs append sid_end.
    if packed_only and native == RECIF_PACKED_DEPTH:
        return native + 1
    return native


def normalize_sid(codes: tuple[int, ...], *, levels: int) -> tuple[int, ...]:
    n = len(codes)
    if n == levels:
        return codes
    if n + 1 == levels:
        return codes + (SID_END_CODE,)
    if n - 1 == levels and codes[-1] == SID_END_CODE:
        return codes[:-1]
    raise ValueError(
        f"cannot reshape SID {','.join(map(str, codes))} ({n} levels) to {levels}"
    )


def format_sid_key(codes: tuple[int, ...], *, levels: Optional[int] = None) -> str:
    if levels is not None:
        codes = normalize_sid(codes, levels=levels)
    return ",".join(str(c) for c in codes)


def write_sid2vid(
    sids: Iterable[tuple[int, ...]],
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


def load_unique_sids(
    src: Path, *, base: int = DEFAULT_BASE
) -> tuple[list[tuple[int, ...]], int, int, bool]:
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return iter_unique_sids(data, base=base)


def convert_file(
    src: Path,
    out: Path,
    *,
    levels: Optional[int] = None,
    base: int = DEFAULT_BASE,
) -> tuple[int, int, int, int]:
    sids, n_in, bad, packed_only = load_unique_sids(src, base=base)
    resolved = infer_levels(sids, packed_only=packed_only) if levels is None else levels
    if resolved < 1:
        raise ValueError(f"catalog levels must be >= 1, got {resolved}")
    unique = sorted({normalize_sid(codes, levels=resolved) for codes in sids})
    write_sid2vid(unique, out, levels=resolved)
    return n_in, len(unique), bad, resolved


def _task_jobs(task: str) -> Iterator[tuple[str, str, str]]:
    names = ("video", "product") if task == "both" else (task,)
    for name in names:
        src_name, out_tmpl = TASK_FILES[name]
        yield name, src_name, out_tmpl


def _print_result(
    src: Path, out: Path, n_in: int, n_out: int, bad: int, levels: int
) -> None:
    print(
        f"{src}: {n_in} keys in, {n_out} unique SIDs out, "
        f"{bad} bad keys, levels={levels} -> {out}"
    )


def convert_jobs(
    jobs: Sequence[tuple[Path, Optional[Path], Optional[str]]],
    *,
    levels: Optional[int] = None,
    base: int = DEFAULT_BASE,
) -> List[ConvertResult]:
    results: List[ConvertResult] = []
    for src, out, out_tmpl in jobs:
        if not src.is_file():
            raise FileNotFoundError(f"missing RecIF mapping: {src}")
        sids, n_in, bad, packed_only = load_unique_sids(src, base=base)
        resolved = infer_levels(sids, packed_only=packed_only) if levels is None else levels
        if resolved < 1:
            raise ValueError(f"catalog levels must be >= 1, got {resolved}")
        unique = sorted({normalize_sid(codes, levels=resolved) for codes in sids})
        if out is None:
            if out_tmpl is None:
                raise ValueError(f"{src}: output path required")
            out = Path(out_tmpl.format(levels=resolved))
        write_sid2vid(unique, out, levels=resolved)
        results.append((out, n_in, len(unique), bad, resolved))
        _print_result(src, out, n_in, len(unique), bad, resolved)
    return results


def run_catalog(
    path: Path,
    *,
    out: Optional[Path] = None,
    task: str = "video",
    levels: Optional[int] = None,
    base: int = DEFAULT_BASE,
) -> int:
    """Convert RecIF mappings at ``path`` (directory or JSON file)."""
    path = Path(path)
    try:
        if path.is_dir():
            out_dir = out or DEFAULT_OUT_DIR
            if out is not None and out.suffix.lower() == ".json":
                raise ValueError(
                    f"--catalog-out {out} is a file; pass a directory when "
                    "--catalog is a RecIF data dir"
                )
            jobs: list[tuple[Path, Optional[Path], Optional[str]]] = []
            for _name, src_name, out_tmpl in _task_jobs(task):
                jobs.append((path / src_name, None, str(out_dir / out_tmpl)))
            convert_jobs(jobs, levels=levels, base=base)
            return 0
        if path.is_file():
            if out is None:
                out_tmpl = str(DEFAULT_OUT_DIR / f"{path.stem}_beamrec_l{{levels}}.json")
                convert_jobs([(path, None, out_tmpl)], levels=levels, base=base)
            elif out.suffix.lower() == ".json":
                convert_jobs([(path, out, None)], levels=levels, base=base)
            else:
                tmpl = str(out / f"{path.stem}_beamrec_l{{levels}}.json")
                convert_jobs([(path, None, tmpl)], levels=levels, base=base)
            return 0
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"missing RecIF mapping: {path}", file=sys.stderr)
    return 1


def parse_script_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert RecIF packed SID catalogs to flashrec sid2vid JSON. "
        "Prefer: flashrec --catalog PATH"
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
        default=None,
        help="output SID depth (default: infer from source keys)",
    )
    ap.add_argument(
        "--codebook-base",
        type=int,
        default=DEFAULT_BASE,
        help="RecIF packing base / per-level codebook size (default: 8192)",
    )
    return ap.parse_args(argv)


def script_main(argv: Optional[list[str]] = None) -> int:
    args = parse_script_args(argv)
    if args.src is not None:
        if args.out is None:
            print(
                "convert_recif_catalog.py: out path required with src",
                file=sys.stderr,
            )
            return 2
        return run_catalog(
            Path(args.src),
            out=Path(args.out),
            levels=args.levels,
            base=args.codebook_base,
        )
    if args.data_dir is not None:
        return run_catalog(
            args.data_dir,
            out=args.out_dir,
            task=args.task,
            levels=args.levels,
            base=args.codebook_base,
        )
    print(
        "convert_recif_catalog.py: provide src/out or --data-dir "
        "(or use flashrec --catalog PATH)",
        file=sys.stderr,
    )
    return 2
