"""Infer SID token range / codebook sizes / boundary tokens from a checkpoint.

GenRec tokenizers (OneRec, Qwen3-based SID models) add contiguous codebook
pieces ``<s_a_0>..<s_c_N>`` plus ``<|sid_begin|>`` / ``<|sid_end|>``. Given
those and a catalog, the three serving layout fields are determined:

* 3-level catalog ``"a,b,c"`` → range is the codebooks, boundary wraps after
* 4-level catalog ``"a,b,c,1"`` → last codebook is the two boundary tokens

stdlib only: ``import flashrec.config`` must stay free of transformers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flashrec.config import BeamRecConfig, same_sid_ids

logger = logging.getLogger(__name__)

_CODEBOOK_RE = re.compile(r"^<s_([A-Za-z0-9]+)_(\d+)>$")
_BEGIN_NAMES = ("<|sid_begin|>", "<sid_begin>")
_END_NAMES = ("<|sid_end|>", "<sid_end>")
_PEEK_BYTES = 65536


@dataclass(frozen=True)
class SidLayout:
    token_range: str
    codebook_sizes: str
    boundary_tokens: str


def resolve_sid_layout(config: BeamRecConfig) -> None:
    """Fill missing SID layout fields from the tokenizer + catalog.

    No-ops when the layout is already complete, or when no catalog is set
    (unconstrained smoke test). Raises if a catalog is set but inference fails
    or disagrees with an explicit flag.
    """
    if (
        config.sid_token_range is not None
        and config.sid_codebook_sizes is not None
        and config.sid_boundary_tokens is not None
    ):
        return
    if not config.sid_vocab_file:
        return
    layout = infer_sid_layout(config.model_path, config.sid_vocab_file)
    _fill(config, "sid_token_range", layout.token_range)
    _fill(config, "sid_codebook_sizes", layout.codebook_sizes)
    _fill(config, "sid_boundary_tokens", layout.boundary_tokens)
    if not config.sid:
        config.sid = f"{layout.token_range}/{layout.codebook_sizes}"
    logger.info(
        "Inferred --sid %s/%s (boundary %s) from tokenizer + catalog",
        layout.token_range,
        layout.codebook_sizes,
        layout.boundary_tokens,
    )


def infer_sid_layout(model_path: str, vocab_file: Optional[str] = None) -> SidLayout:
    added = load_added_token_map(model_path)
    layers, begin_id, end_id = parse_codebook_layers(added)
    width = catalog_code_width(vocab_file) if vocab_file else None
    if width == 1:
        raise ValueError(
            f"{vocab_file}: catalog keys look like packed RecIF integers; "
            "convert with scripts/build_catalog.sh first"
        )
    sizes = [n for _, _, n in layers]
    codebook_lo = layers[0][0]
    codebook_hi = layers[-1][1]
    if begin_id is None or end_id is None:
        raise ValueError(
            f"{model_path}: tokenizer has codebook tokens but no "
            "<|sid_begin|> / <|sid_end|>; pass --sid explicitly"
        )
    if begin_id != codebook_hi + 1 or end_id != codebook_hi + 2:
        raise ValueError(
            f"{model_path}: <|sid_begin|>={begin_id}, <|sid_end|>={end_id} "
            f"are not immediately after codebook range {codebook_lo}:{codebook_hi}; "
            "pass --sid explicitly"
        )
    if width == 4:
        token_range = f"{codebook_lo}:{end_id}"
        sizes = sizes + [2]
    else:
        token_range = f"{codebook_lo}:{codebook_hi}"
    boundary = f"{begin_id},{end_id}"
    return SidLayout(
        token_range=token_range,
        codebook_sizes=",".join(str(s) for s in sizes),
        boundary_tokens=boundary,
    )


def load_added_token_map(model_path: str) -> Dict[str, int]:
    root = Path(model_path)
    added = root / "added_tokens.json"
    if added.is_file():
        data = json.loads(added.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data:
            mapping = {str(k): int(v) for k, v in data.items()}
            if any(_CODEBOOK_RE.match(k) for k in mapping):
                return mapping
    cfg = root / "tokenizer_config.json"
    if cfg.is_file():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        dec = data.get("added_tokens_decoder") or {}
        out: Dict[str, int] = {}
        for tid, meta in dec.items():
            if isinstance(meta, dict) and "content" in meta:
                out[str(meta["content"])] = int(tid)
        if out:
            return out
    tok = root / "tokenizer.json"
    if tok.is_file():
        data = json.loads(tok.read_text(encoding="utf-8"))
        pieces = data.get("added_tokens") or []
        out = {
            str(item["content"]): int(item["id"])
            for item in pieces
            if isinstance(item, dict) and "content" in item and "id" in item
        }
        if out:
            return out
    raise ValueError(
        f"{model_path}: no added_tokens.json / tokenizer_config.json SID tokens; "
        "pass --sid START:END/SIZE,..."
    )


def parse_codebook_layers(
    added: Dict[str, int],
) -> Tuple[List[Tuple[int, int, int]], Optional[int], Optional[int]]:
    """Return ``(layers, begin_id, end_id)``.

    Each layer is ``(first_id, last_id, size)`` in token-id order.
    """
    grouped: Dict[str, List[Tuple[int, int]]] = {}
    begin_id = end_id = None
    for tok, tid in added.items():
        m = _CODEBOOK_RE.match(tok)
        if m:
            grouped.setdefault(m.group(1), []).append((int(tid), int(m.group(2))))
            continue
        if tok in _BEGIN_NAMES:
            begin_id = int(tid)
        elif tok in _END_NAMES:
            end_id = int(tid)
    if not grouped:
        raise ValueError(
            "tokenizer has no <s_a_0>-style codebook tokens; pass --sid explicitly"
        )
    layers: List[Tuple[int, int, int]] = []
    for name, pairs in grouped.items():
        layers.append(_layer_span(name, pairs))
    layers.sort(key=lambda x: x[0])
    for i in range(1, len(layers)):
        if layers[i][0] != layers[i - 1][1] + 1:
            raise ValueError(
                "codebook token ids are not packed contiguously; pass --sid"
            )
    return layers, begin_id, end_id


def catalog_code_width(path: Optional[str]) -> Optional[int]:
    """Number of comma-separated codes in the first sid2vid JSON key."""
    if not path:
        return None
    key = _first_json_object_key(Path(path))
    if key is None:
        return None
    parts = [p.strip() for p in key.split(",") if p.strip()]
    if not parts:
        return None
    if not all(_is_int(p) for p in parts):
        return None
    return len(parts)


def _layer_span(name: str, pairs: List[Tuple[int, int]]) -> Tuple[int, int, int]:
    pairs = sorted(pairs, key=lambda x: x[1])
    n = len(pairs)
    if n == 0:
        raise ValueError(f"empty codebook layer {name}")
    codes = [c for _, c in pairs]
    ids = [i for i, _ in pairs]
    if codes[0] != 0 or codes[-1] != n - 1 or len(set(codes)) != n:
        raise ValueError(
            f"codebook layer {name} codes are not a complete 0..{n - 1} range"
        )
    if ids[-1] != ids[0] + n - 1:
        raise ValueError(f"codebook layer {name} token ids are not contiguous")
    return ids[0], ids[-1], n


def _first_json_object_key(path: Path) -> Optional[str]:
    if not path.is_file():
        raise FileNotFoundError(f"sid-vocab-file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        chunk = f.read(_PEEK_BYTES)
    text = chunk.lstrip()
    if not text.startswith("{"):
        return None
    m = re.search(r'\{\s*"((?:\\.|[^"\\])*)"', text)
    if not m:
        return None
    return json.loads(f'"{m.group(1)}"')


def _is_int(text: str) -> bool:
    if text[:1] == "-":
        text = text[1:]
    return bool(text) and text.isdigit()


def _fill(config: BeamRecConfig, name: str, inferred: str) -> None:
    current = getattr(config, name)
    if current is None:
        setattr(config, name, inferred)
        return
    if not same_sid_ids(current, inferred):
        raise ValueError(
            f"{name}={current!r} conflicts with inferred {inferred} "
            f"from tokenizer + catalog; omit the flag or pass --sid"
        )
