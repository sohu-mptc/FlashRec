"""Request / batch / result types. A GPU row is a beam slot, not a user request."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from flashrec.search.state import BeamSearchList


@dataclass
class FinishReason:
    type: str
    length: Optional[int] = None
    matched: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": self.type}
        if self.length is not None:
            out["length"] = self.length
        if self.matched is not None:
            out["matched"] = self.matched
        return out


@dataclass(slots=True)
class BeamSequence:
    tokens: List[int]
    cum_logprob: float = 0.0
    beam_score: float = 0.0
    text: str = ""
    finish_reason: Optional[Dict[str, Any]] = None


@dataclass
class BeamResult:
    text: str
    output_ids: List[int]
    sequences: List[BeamSequence]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class BeamRequest:
    input_ids: List[int]
    beam_width: int
    max_new_tokens: int
    rid: str = field(default_factory=lambda: uuid.uuid4().hex)
    ignore_eos: bool = True
    stop_token_ids: Optional[Sequence[int]] = None
    length_penalty: float = 1.0
    # 0 = deterministic top-k beam search; > 0 = Gumbel top-k stochastic beam
    # search over softmax(logits / temperature) (sampling without replacement).
    temperature: float = 0.0
    messages: Optional[List[Dict[str, Any]]] = None
    beam_list: BeamSearchList = field(default_factory=BeamSearchList)
    prefill_pool_idx: int = -1
    beam_pool_indices: Optional[torch.Tensor] = None
    beam_pool_indices_cpu: Optional[List[int]] = None
    seq_len: int = 0
    prompt_len: int = 0
    finished: bool = False
    finish_reason: Optional[FinishReason] = None
    cached_tokens: int = 0
    radix_lock_tokens: Optional[List[int]] = None
    # CPU copy of the prefill KV page ids (radix off), so freeing never reads
    # req_to_token back from GPU.
    prefill_kv_cpu_all: Optional[List[int]] = None
    expanded: bool = False

    @property
    def beam_candidates(self) -> int:
        return max(int(self.beam_width) * 2, int(self.beam_width))


@dataclass
class ForwardBatch:
    """One GPU forward. ``n_rows`` is Σ beam slots (decode) or #prompts (prefill)."""

    input_ids: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    is_prefill: bool
    extend_prefix_lens: Optional[List[int]] = None
    extend_seq_lens: Optional[List[int]] = None
    kv_indices: Optional[torch.Tensor] = None  # ragged page indices, page_size=1
    kv_indptr: Optional[torch.Tensor] = None
    buffers_ready: bool = False  # decode graph static buffers already filled
    want_expand: bool = False

    @property
    def n_rows(self) -> int:
        return int(self.seq_lens.numel())
