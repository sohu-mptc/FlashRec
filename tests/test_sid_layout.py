"""Infer SID layout from tokenizer added tokens + catalog keys."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from flashrec.config import BeamRecConfig
from flashrec.sid_layout import (
    infer_sid_layout,
    resolve_sid_layout,
)


def _write_tokenizer(root: Path, *, start: int, size: int, n_layers: int = 3) -> None:
    added = {}
    cursor = start
    names = "abcdefghijklmnopqrstuvwxyz"
    for layer in range(n_layers):
        for code in range(size):
            added[f"<s_{names[layer]}_{code}>"] = cursor
            cursor += 1
    added["<|sid_begin|>"] = cursor
    added["<|sid_end|>"] = cursor + 1
    (root / "added_tokens.json").write_text(json.dumps(added), encoding="utf-8")


def _write_catalog(path: Path, key: str) -> None:
    path.write_text(json.dumps({key: 1}), encoding="utf-8")


class TestInferSidLayout:
    def test_four_level_catalog_appends_boundary_codebook(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tokenizer(root, start=10, size=4)
            cat = root / "cat.json"
            _write_catalog(cat, "0,0,0,1")
            layout = infer_sid_layout(str(root), str(cat))
            assert layout.token_range == "10:23"
            assert layout.codebook_sizes == "4,4,4,2"
            assert layout.boundary_tokens == "22,23"

    def test_three_level_catalog_keeps_boundary_outside_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tokenizer(root, start=10, size=4)
            cat = root / "cat.json"
            _write_catalog(cat, "0,0,0")
            layout = infer_sid_layout(str(root), str(cat))
            assert layout.token_range == "10:21"
            assert layout.codebook_sizes == "4,4,4"
            assert layout.boundary_tokens == "22,23"

    def test_packed_catalog_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tokenizer(root, start=10, size=4)
            cat = root / "cat.json"
            _write_catalog(cat, "12345")
            with pytest.raises(ValueError) as ctx:
                infer_sid_layout(str(root), str(cat))
            assert "packed" in str(ctx.value)

    def test_resolve_fills_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tokenizer(root, start=151669, size=8)
            cat = root / "sid2vid.json"
            _write_catalog(cat, "0,1,2,1")
            cfg = BeamRecConfig(
                model_path=str(root),
                sid_vocab_file=str(cat),
            )
            resolve_sid_layout(cfg)
            assert cfg.sid_codebook_sizes == "8,8,8,2"
            assert cfg.parsed_boundary_token_ids() == [151693, 151694]
            assert cfg.sid == "151669:151694/8,8,8,2"

    def test_resolve_skips_when_layout_complete(self):
        cfg = BeamRecConfig(
            model_path="/missing/model",
            sid_vocab_file="/missing/cat.json",
            sid="10:21/4,4,4",
        )
        resolve_sid_layout(cfg)
        assert cfg.sid_token_range == "10:21"

    def test_resolve_noop_without_catalog(self):
        cfg = BeamRecConfig(model_path="/missing/model")
        resolve_sid_layout(cfg)
        assert cfg.sid_token_range is None

    def test_explicit_flag_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tokenizer(root, start=10, size=4)
            cat = root / "cat.json"
            _write_catalog(cat, "0,0,0,1")
            cfg = BeamRecConfig(
                model_path=str(root),
                sid_vocab_file=str(cat),
                sid_boundary_tokens="1,2",
            )
            with pytest.raises(ValueError):
                resolve_sid_layout(cfg)
