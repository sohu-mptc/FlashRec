"""Dense beam-search state (simplified BeamSearchList)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import torch

from flashrec.search.score import beam_score as default_score_fn


@dataclass
class BeamSearchSequence:
    tokens: List[int]
    cum_logprob: float = 0.0
    finish_reason: Optional[object] = None
    text: Optional[str] = None
    beam_score: Optional[float] = None

    def finished(self) -> bool:
        return self.finish_reason is not None


@dataclass
class BeamSearchList:
    """Per-request beam frontier: dense GPU tensors plus optional object stubs."""

    completed: List[BeamSearchSequence] = field(default_factory=list)
    incomplete: List[BeamSearchSequence] = field(default_factory=list)
    cum_logprobs: Optional[torch.Tensor] = None
    last_tokens: Optional[torch.Tensor] = None
    token_ids: Optional[torch.Tensor] = None
    cur_len: int = 0
    dense_authoritative: bool = False
    prompt_len: int = 0
    node_ids: Optional[torch.Tensor] = None
    owned_decode_kv_cpu: List[int] = field(default_factory=list)

    def generated_len(self) -> int:
        if self.dense_authoritative and self.token_ids is not None and self.cur_len > 0:
            return int(self.cur_len)
        if self.incomplete:
            return len(self.incomplete[0].tokens)
        if self.token_ids is not None and self.cur_len > 0:
            return int(self.cur_len)
        if self.completed:
            return len(self.completed[0].tokens)
        return 0

    def init_token_ids(
        self, max_new_tokens: int, device: Optional[torch.device] = None
    ) -> None:
        beam_width = (
            len(self.incomplete)
            if self.incomplete
            else (int(self.last_tokens.shape[0]) if self.last_tokens is not None else 0)
        )
        max_new_tokens = max(int(max_new_tokens), 1)
        if device is None:
            device = (
                self.last_tokens.device
                if self.last_tokens is not None
                else torch.device("cpu")
            )
        self.token_ids = torch.zeros(
            (max(beam_width, 1), max_new_tokens), dtype=torch.int64, device=device
        )
        if beam_width == 0:
            self.token_ids = self.token_ids[:0]
        self.cur_len = 0
        self.dense_authoritative = True
        for i, beam in enumerate(self.incomplete):
            n = min(len(beam.tokens), max_new_tokens)
            if n <= 0:
                continue
            self.token_ids[i, :n] = torch.as_tensor(
                beam.tokens[:n], dtype=torch.int64, device=device
            )
            self.cur_len = max(self.cur_len, n)

    def expand_token_ids(
        self,
        parent_indices: Union[Sequence[int], torch.Tensor],
        new_tokens: Union[Sequence[int], torch.Tensor],
    ) -> None:
        if self.token_ids is None:
            return
        device = self.token_ids.device
        parents = torch.as_tensor(
            parent_indices, dtype=torch.int64, device=device
        ).view(-1)
        toks = torch.as_tensor(
            new_tokens, dtype=self.token_ids.dtype, device=device
        ).view(-1)
        n = int(parents.numel())
        if n == 0:
            self.token_ids = self.token_ids[:0]
            self.dense_authoritative = True
            return
        if toks.numel() != n:
            raise ValueError("parent_indices and new_tokens must have the same length")
        max_parent = max(self.token_ids.shape[0] - 1, 0)
        parents = parents.clamp(min=0, max=max_parent)
        col = int(self.cur_len)
        if (
            self.token_ids.is_cuda
            and self.token_ids.dtype == torch.int64
            and col < int(self.token_ids.shape[1])
        ):
            try:
                from flashrec.kernel.beam_trie import beam_expand_token_ids

                self.token_ids = beam_expand_token_ids(
                    self.token_ids, parents, toks, col
                )
                self.cur_len = col + 1
                self.dense_authoritative = True
                return
            except Exception:
                pass
        gathered = self.token_ids[parents]
        if col >= gathered.shape[1]:
            gathered = torch.nn.functional.pad(gathered, (0, 1))
        gathered[:, col] = toks
        self.token_ids = gathered
        self.cur_len = col + 1
        self.dense_authoritative = True

    def sequences_from_token_ids(
        self,
        cum_logprobs: Optional[Union[Sequence[float], torch.Tensor]] = None,
        length_penalty: float = 1.0,
        finish_reason: Optional[object] = None,
    ) -> List[BeamSearchSequence]:
        if self.token_ids is None or self.cur_len <= 0:
            return []
        tok = self.token_ids[:, : self.cur_len].detach()
        if tok.device.type != "cpu":
            tok = tok.cpu()
        tokens_cpu = tok.numpy().tolist()
        if cum_logprobs is None:
            if self.cum_logprobs is not None:
                vals_t = self.cum_logprobs.detach()
                if vals_t.device.type != "cpu":
                    vals_t = vals_t.cpu()
                vals = vals_t.numpy().tolist()
            else:
                vals = [0.0] * len(tokens_cpu)
        elif isinstance(cum_logprobs, torch.Tensor):
            vals_t = cum_logprobs.detach()
            if vals_t.device.type != "cpu":
                vals_t = vals_t.cpu()
            vals = vals_t.numpy().tolist()
        else:
            vals = list(cum_logprobs)
        out = []
        for toks, val in zip(tokens_cpu, vals):
            out.append(
                BeamSearchSequence(
                    tokens=list(toks),
                    cum_logprob=float(val),
                    finish_reason=finish_reason,
                    beam_score=default_score_fn(float(val), len(toks), length_penalty),
                )
            )
        return out

    def ensure_empty_stubs(self, n: int) -> List[BeamSearchSequence]:
        if (
            self.incomplete
            and len(self.incomplete) == n
            and all(not beam.tokens for beam in self.incomplete)
        ):
            return self.incomplete
        return [BeamSearchSequence(tokens=[], cum_logprob=0.0) for _ in range(n)]

    def owned_decode_kv_count(self) -> int:
        return len(self.owned_decode_kv_cpu)

    def record_owned_decode_kv_cpu(self, ids) -> None:
        """CPU-side bookkeeping of owned decode KV pages (no GPU tensor, no sync)."""
        if ids:
            self.owned_decode_kv_cpu.extend(ids)

    def take_owned_decode_kv_cpu(self) -> List[int]:
        ids = self.owned_decode_kv_cpu
        self.owned_decode_kv_cpu = []
        return ids
