"""HF tokenizer wrapper. transformers is used only for tokenize/detokenize."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer


def _compressed_sohu_window(codebook_sizes: Optional[List[int]]) -> bool:
    """SoHu-0.6B serves a 512×3 codebook packed into ``<s_a_0>..<s_a_1535>``.

    OneRec uses the same tokenizer names with real 8192-wide ``<s_a/b/c_*>``
    layers. Remapping that layout prints legal ``<s_a_512+>`` codes as
    ``<s_b_*>`` / ``<s_c_*>`` and the catalog lookup misses them.
    """
    if not codebook_sizes or len(codebook_sizes) < 3:
        return False
    return (
        int(codebook_sizes[0]) == 512
        and int(codebook_sizes[1]) == 512
        and int(codebook_sizes[2]) == 512
    )


class TokenizerAdapter:
    def __init__(
        self,
        model_path: str,
        codebook_sizes: Optional[List[int]] = None,
    ):
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
        # SoHu-0.6B ships contiguous <s_a_0>..<s_a_8191> then separate
        # <s_b_*> / <s_c_*> namespaces, but production GenRec serves the
        # compressed 512×3 window at the start of <s_a_*> (catalog token_ids
        # [base, base+512, base+1024]). Remap that window to s_a/s_b/s_c so
        # decoded beams match GT / catalog naming.
        if _compressed_sohu_window(codebook_sizes):
            self._remap_sohu_compressed_codebook_pieces()
        self._piece_arr: Optional[np.ndarray] = None
        if self._piece:
            size = max(self._piece) + 1
            arr = np.empty(size, dtype=object)
            for idx, s in self._piece.items():
                arr[idx] = s
            self._piece_arr = arr

    def _remap_sohu_compressed_codebook_pieces(self) -> None:
        piece = self._piece
        if not piece:
            return
        a0 = next((i for i, s in piece.items() if s == "<s_a_0>"), None)
        b0 = next((i for i, s in piece.items() if s == "<s_b_0>"), None)
        if a0 is None or b0 is None or b0 <= a0 + 512:
            return
        # Compressed production window: three 512-wide levels packed into s_a_*.
        width = 512
        n_levels = 3
        letters = "abc"
        for level in range(n_levels):
            for code in range(width):
                tid = a0 + level * width + code
                if tid in piece:
                    piece[tid] = f"<s_{letters[level]}_{code}>"

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
        piece = self._piece
        if piece:
            try:
                return "".join(piece[int(t)] for t in tokens)
            except (KeyError, TypeError):
                pass
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
