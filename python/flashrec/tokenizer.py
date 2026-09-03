"""HF tokenizer wrapper. transformers is used only for tokenize/detokenize."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer


class TokenizerAdapter:
    def __init__(self, model_path: str):
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Added/special tokens decode as their literal piece, so a dict lookup
        # reproduces HF decode exactly. GenRec beams are all SID special
        # tokens, which keeps the per-wave 400-sequence decode off the HF
        # python wrapper (to_py_obj etc.) in the response-gating tail.
        self._piece: Dict[int, str] = {}
        try:
            for tok_str, idx in self.tok.get_added_vocab().items():
                self._piece[int(idx)] = str(tok_str)
            for tok_str in self.tok.all_special_tokens:
                idx = self.tok.convert_tokens_to_ids(tok_str)
                if idx is not None and int(idx) >= 0:
                    self._piece[int(idx)] = str(tok_str)
        except Exception:
            self._piece = {}
        # Vocab-indexed piece table: decode a whole (beams, len) id matrix with
        # one fancy-index instead of a per-token dict lookup. Ids without a
        # piece hold None so "".join raises TypeError -> caller falls back.
        self._piece_arr: Optional[np.ndarray] = None
        if self._piece:
            size = max(self._piece) + 1
            arr = np.empty(size, dtype=object)
            for idx, s in self._piece.items():
                arr[idx] = s
            self._piece_arr = arr

    def decode_matrix(self, ids: np.ndarray) -> Optional[List[str]]:
        """Decode an int id matrix (n, L) of special/added tokens; None if any
        id lacks a literal piece (caller must use the real decoder)."""
        arr = self._piece_arr
        if arr is None:
            return None
        if ids.size == 0:
            return ["" for _ in range(ids.shape[0])]
        if int(ids.min()) < 0 or int(ids.max()) >= arr.shape[0]:
            return None
        rows = arr[ids].tolist()
        try:
            return ["".join(r) for r in rows]
        except TypeError:
            return None

    def encode(self, text: str) -> List[int]:
        return [int(x) for x in self.tok.encode(text, add_special_tokens=False)]

    def decode(self, tokens: List[int]) -> str:
        try:
            return self.tok.decode(tokens, skip_special_tokens=False)
        except Exception:
            return ""

    def batch_decode(self, sequences: List[List[int]]) -> List[str]:
        if not sequences:
            return []
        piece = self._piece
        if piece:
            try:
                # map+__getitem__ beats a genexpr with int() casts ~2x; token
                # ids from .tolist() are already ints (np ints hash the same).
                lookup = piece.__getitem__
                return ["".join(map(lookup, seq)) for seq in sequences]
            except (KeyError, TypeError):
                pass  # a non-special id: byte-level merges need the real decoder
        try:
            return [
                str(x)
                for x in self.tok.batch_decode(sequences, skip_special_tokens=False)
            ]
        except Exception:
            return [self.decode(t) for t in sequences]

    def apply_chat(
        self,
        messages: List[Dict[str, Any]],
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        kwargs = dict(chat_template_kwargs or {})
        kwargs.setdefault("enable_thinking", False)
        try:
            encoded = self.tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **kwargs,
            )
        except TypeError:
            encoded = self.tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        return [int(x) for x in encoded]
