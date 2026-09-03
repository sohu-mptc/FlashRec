# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Beam search precision check: FlashRec vs HuggingFace transformers.

Adapted from SGLang's beam-search diff test for GenRec codebook-constrained beam.

Some GenRec checkpoints store FP8 weights without channel scales. HuggingFace
loads a bf16 copy via bare FP8→bf16 cast (same scale=1 assumption FlashRec uses when
``weight_scale`` is absent). Overlap is measured on decoded SID strings.

Usage::

    FLASHREC_DIFF_MODEL=/path/to/genrec \\
      PYTHONPATH=python pytest tests/test_beam_search_diff.py -v

Optional env:
  FLASHREC_DIFF_MODEL    model dir (required; test skips when unset)
  FLASHREC_DIFF_PROMPT   user text for chat template (default: short GenRec-like prompt)
  FLASHREC_DIFF_CATALOG  sid-vocab JSON; omit for codebook-only (not trie) constraint
  FLASHREC_DIFF_QUANT    ``fp8`` (default, production path) or ``bf16`` (tighter HF match)
  FLASHREC_DIFF_BEAMS    comma-separated beam widths (default ``4,8``)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Set

import pytest
import torch
import torch.nn.functional as F

from flashrec.config import BeamRecConfig, parse_int_list
from flashrec.core import ForwardBatch
from flashrec.engine.engine import ModelEngine
from flashrec.scheduler.scheduler import BeamRecEngine
from flashrec.sid_layout import infer_sid_layout

_DEFAULT_PROMPT = (
    "predict next: <|sid_begin|><s_a_0><s_b_0><s_c_0><|sid_end|>"
)


def _resolve_model_path() -> Optional[str]:
    raw = os.environ.get("FLASHREC_DIFF_MODEL", "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_dir():
        return None
    if not (path / "config.json").is_file():
        return None
    return str(path)


def _sid_layout_for(model_path: str):
    catalog = os.environ.get("FLASHREC_DIFF_CATALOG", "").strip() or None
    return infer_sid_layout(model_path, catalog)


def _quantization() -> str:
    raw = os.environ.get("FLASHREC_DIFF_QUANT", "fp8").strip().lower()
    return raw or "fp8"


def _beam_widths() -> List[int]:
    raw = os.environ.get("FLASHREC_DIFF_BEAMS", "").strip()
    if not raw:
        return [4, 8]
    widths = [int(x) for x in raw.split(",") if x.strip()]
    if not widths or any(n < 1 for n in widths):
        raise ValueError(f"invalid FLASHREC_DIFF_BEAMS={raw!r}")
    return widths


def _codebooks(special_ids: Sequence[int], sizes: Sequence[int]) -> List[List[int]]:
    ids = [int(x) for x in special_ids]
    out: List[List[int]] = []
    offset = 0
    for size in sizes:
        size_i = int(size)
        chunk = ids[offset : offset + size_i]
        if len(chunk) != size_i:
            raise ValueError(
                f"special token pool too small for codebook sizes={list(sizes)}"
            )
        out.append(chunk)
        offset += size_i
    return out


def _iter_safetensors(model_path: str):
    from safetensors import safe_open

    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        import json

        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        single = root / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError(f"no safetensors under {model_path}")
        files = [single.name]
    for f in files:
        with safe_open(str(root / f), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                yield key, handle.get_tensor(key)


def _load_hf_causal_lm_bf16(model_path: str, device: torch.device):
    """Build a transformers Qwen3ForCausalLM and load weights as bf16.

    FP8 tensors are cast with implicit scale=1 (matches FlashRec when scales missing).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    state = {}
    for key, tensor in _iter_safetensors(model_path):
        if key.endswith(".weight_scale") or key.endswith(".weight_scale_inv"):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            state[key] = tensor.to(dtype=torch.bfloat16)
        else:
            state[key] = tensor
    missing, unexpected = model.load_state_dict(state, strict=False)
    # tied lm_head / embed is fine; ignore non-critical leftovers
    del missing, unexpected
    model.to(device)
    model.eval()
    return model


def _flashrec_prefill_logprobs(
    runner: ModelEngine, input_ids: List[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restricted log-probs for the last prompt position, via one extend forward.

    Mirrors ``BeamRecEngine._prefill_batch`` for a single request with no radix
    prefix hit, so the compared numbers come from the production kernels
    (fused qk-norm/rope, FP8 KV pages, restricted LM head).
    """
    n = len(input_ids)
    kv, _ = runner.alloc_tokens_cpu(n)
    req_idx = runner.alloc_req_one()
    try:
        runner.write_prefill(req_idx, kv)
        table = runner.req_pool.req_to_token
        device = runner.device
        batch = ForwardBatch(
            input_ids=torch.tensor(input_ids, dtype=torch.int64, device=device),
            req_pool_indices=torch.tensor([req_idx], dtype=torch.int64, device=device),
            seq_lens=torch.tensor([n], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([n], dtype=torch.int64),
            positions=torch.arange(n, dtype=torch.int64, device=device),
            out_cache_loc=kv,
            is_prefill=True,
            extend_prefix_lens=[0],
            extend_seq_lens=[n],
            kv_indices=table[req_idx, :n].to(dtype=torch.int32),
        )
        logprobs, cand_ids, _ = runner.forward(batch)
        if logprobs is None:
            raise AssertionError("prefill forward returned no logprobs")
        return logprobs[-1].detach().float().cpu(), cand_ids
    finally:
        runner.free_tokens(kv)
        runner.free_req_slots([req_idx])


def _hf_prefill_logprobs(
    model, input_ids: List[int], cand_ids: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Same quantity from HF: gather restricted columns, then log_softmax."""
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    last = logits[0, -1].float()
    gathered = last.index_select(0, cand_ids.to(device=device, dtype=torch.long))
    return F.log_softmax(gathered, dim=-1).cpu()


# ---------------------------------------------------------------------------
# Beam-search-diff fixture: shared state formerly held in TestBeamSearchDiff.
# ---------------------------------------------------------------------------
class _BeamDiffState:
    def __init__(self):
        self.model_path = _resolve_model_path()
        assert self.model_path is not None
        self.prompt = os.environ.get("FLASHREC_DIFF_PROMPT", _DEFAULT_PROMPT)
        layout = _sid_layout_for(self.model_path)
        self.sid_token_range = layout.token_range
        self.codebook_sizes_s = layout.codebook_sizes
        self.boundary_s = layout.boundary_tokens
        self.special_ids = parse_int_list(layout.token_range) or []
        self.codebook_sizes = parse_int_list(layout.codebook_sizes) or []
        self.boundary_ids = parse_int_list(layout.boundary_tokens) or []
        self.depth = len(self.codebook_sizes)
        self.codebooks = _codebooks(self.special_ids, self.codebook_sizes)
        self.begin_id = int(self.boundary_ids[0])
        self.end_id = int(self.boundary_ids[1])
        self.quantization = _quantization()
        # Candidate pool is 2*n in FlashRec vs full codebook in HF → allow some miss.
        # FP8 vs a bf16 HF ref is looser at small n (8192-wide OneRec codebook).
        overlap_default = "0.5" if self.quantization == "fp8" else "0.6"
        self.overlap_threshold = float(
            os.environ.get("FLASHREC_DIFF_OVERLAP", overlap_default)
        )
        self.beam_widths = _beam_widths()
        self.device = torch.device("cuda:0")
        self._cached_hf = None
        self._cached_flashrec = None
        self._cached_tok = None
        self._cached_masks = None
        print(
            f"\nSID layout {layout.token_range}/{layout.codebook_sizes} "
            f"boundary {layout.boundary_tokens} quant={self.quantization} "
            f"beams={self.beam_widths}"
        )

    def release(self):
        self._cached_hf = None
        self._cached_flashrec = None
        self._cached_tok = None
        self._cached_masks = None
        torch.cuda.empty_cache()


def _tokenizer(state: _BeamDiffState):
    if state._cached_tok is None:
        from transformers import AutoTokenizer

        state._cached_tok = AutoTokenizer.from_pretrained(
            state.model_path, trust_remote_code=True
        )
    return state._cached_tok


def _prompt_input_ids(state: _BeamDiffState) -> List[int]:
    tok = _tokenizer(state)
    messages = [{"role": "user", "content": state.prompt}]
    try:
        encoded = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        encoded = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    ids = [int(x) for x in encoded]
    if not ids or ids[-1] != state.begin_id:
        ids.append(state.begin_id)
    return ids


def _wrap_sid(state: _BeamDiffState, body: Sequence[int]) -> List[int]:
    return [state.begin_id, *[int(x) for x in body], state.end_id]


def _decode(state: _BeamDiffState, token_lists: Sequence[Sequence[int]]) -> List[str]:
    return [
        str(x)
        for x in _tokenizer(state).batch_decode(
            [list(t) for t in token_lists], skip_special_tokens=False
        )
    ]


def _hf_model(state: _BeamDiffState):
    if state._cached_hf is None:
        state._cached_hf = _load_hf_causal_lm_bf16(state.model_path, state.device)
    return state._cached_hf


def _flashrec_engine(state: _BeamDiffState) -> BeamRecEngine:
    if state._cached_flashrec is None:
        max_n = max(state.beam_widths)
        state._cached_flashrec = BeamRecEngine(
            BeamRecConfig(
                model_path=state.model_path,
                quantization=state.quantization,
                sid_token_range=state.sid_token_range,
                sid_codebook_sizes=state.codebook_sizes_s,
                sid_boundary_tokens=state.boundary_s,
                sid_vocab_file=os.environ.get("FLASHREC_DIFF_CATALOG", "").strip()
                or None,
                beam_width=max_n,
                max_tokens=state.depth + 2,
                length_penalty=1.0,
                enable_cuda_graph=False,
                enable_graph_expand=False,
                enable_radix=False,
                enable_warmup=False,
                enable_decode_pack=False,
                mem_fraction_static=0.55 if max_n >= 128 else 0.35,
                max_running_requests=8,
                batch_slots=max(max_n * 4, 64),
                cuda_graph_max_bs=max(max_n, 8),
                host_worker_threads=0,
            )
        )
    return state._cached_flashrec


def _codebook_masks(state: _BeamDiffState, vocab_size: int) -> List[torch.Tensor]:
    cached = state._cached_masks
    if cached is not None:
        return cached
    neg = torch.finfo(torch.float32).min
    masks: List[torch.Tensor] = []
    for cb in state.codebooks:
        mask = torch.full((vocab_size,), neg, dtype=torch.float32, device=state.device)
        mask[torch.tensor(cb, dtype=torch.long, device=state.device)] = 0.0
        masks.append(mask)
    state._cached_masks = masks
    return masks


def _get_transformers_beam_sequences(
    state: _BeamDiffState, input_ids: List[int], beam_width: int
) -> List[str]:
    from transformers import LogitsProcessor, LogitsProcessorList

    model = _hf_model(state)
    prompt_len = len(input_ids)
    depth = state.depth
    pad_id = int(getattr(model.config, "eos_token_id", 0) or 0)
    masks = _codebook_masks(state, int(model.config.vocab_size))

    class _StepMask(LogitsProcessor):
        def __call__(self, input_ids_t: torch.Tensor, scores: torch.Tensor):
            step = int(input_ids_t.shape[-1]) - prompt_len
            if 0 <= step < depth:
                return scores + masks[step]
            scores = scores.clone()
            scores.fill_(torch.finfo(scores.dtype).min)
            scores[:, pad_id] = 0.0
            return scores

    inputs = torch.tensor([input_ids], dtype=torch.long, device=state.device)
    attn = torch.ones_like(inputs)
    with torch.no_grad():
        generated = model.generate(
            inputs,
            attention_mask=attn,
            max_new_tokens=depth,
            min_new_tokens=depth,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            do_sample=False,
            length_penalty=1.0,
            early_stopping=False,
            renormalize_logits=True,
            pad_token_id=pad_id,
            logits_processor=LogitsProcessorList([_StepMask()]),
        )

    bodies: List[List[int]] = []
    for row in generated:
        body = row[prompt_len : prompt_len + depth].detach().cpu().tolist()
        bodies.append(_wrap_sid(state, body))
    return _decode(state, bodies)


def _get_flashrec_beam_sequences(
    state: _BeamDiffState, input_ids: List[int], beam_width: int
) -> List[str]:
    result = _flashrec_engine(state).generate(
        input_ids=input_ids,
        n=beam_width,
        max_tokens=state.depth + 2,
    )
    texts = [seq.text for seq in result.sequences]
    if len(texts) < beam_width:
        raise AssertionError(
            f"FlashRec returned {len(texts)} sequences, expected {beam_width}"
        )
    return texts[:beam_width]


def _sequence_overlap(sequences1: List[str], sequences2: List[str]) -> float:
    set1: Set[str] = set(sequences1)
    set2: Set[str] = set(sequences2)
    beam_width = len(sequences1)
    if beam_width == 0:
        return 0.0
    return len(set1 & set2) / float(beam_width)


@pytest.fixture(scope="module")
def beam_diff_state():
    state = _BeamDiffState()
    yield state
    state.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not _resolve_model_path(),
    reason="set FLASHREC_DIFF_MODEL to a GenRec model dir",
)
def test_beam_search_different_widths(beam_diff_state: _BeamDiffState):
    state = beam_diff_state
    input_ids = _prompt_input_ids(state)
    assert input_ids[-1] == state.begin_id
    preview = 3

    hf_by_n = {}
    for beam_width in state.beam_widths:
        print(f"\n[HF] generating n={beam_width}", flush=True)
        hf_by_n[beam_width] = _get_transformers_beam_sequences(input_ids, beam_width)
    state._cached_hf = None
    state._cached_masks = None
    torch.cuda.empty_cache()

    mb_by_n = {}
    for beam_width in state.beam_widths:
        print(f"\n[flashrec] generating n={beam_width}", flush=True)
        mb_by_n[beam_width] = _get_flashrec_beam_sequences(input_ids, beam_width)

    print(f"\n{'n':>5}  {'overlap':>8}  {'top-1':>5}  {'|intersect|':>12}")
    for beam_width in state.beam_widths:
        hf_sequences = hf_by_n[beam_width]
        mb_sequences = mb_by_n[beam_width]
        overlap = _sequence_overlap(hf_sequences, mb_sequences)
        top1 = hf_sequences[0] == mb_sequences[0]
        inter = len(set(hf_sequences) & set(mb_sequences))
        print(
            f"{beam_width:5d}  {overlap:8.2%}  {str(top1):>5}  "
            f"{inter:5d}/{beam_width}",
            flush=True,
        )
        print(f"\n{'=' * 60}")
        print(f"n={beam_width} overlap={overlap:.2%} top-1 match={top1}")
        print(f"{'=' * 60}")
        print(f"[transformers] {len(hf_sequences)} (head {preview}):")
        for i, seq in enumerate(hf_sequences[:preview]):
            print(f"  [{i}] {seq!r}")
        print(f"[flashrec] {len(mb_sequences)} (head {preview}):")
        for i, seq in enumerate(mb_sequences[:preview]):
            print(f"  [{i}] {seq!r}")

    for beam_width in state.beam_widths:
        overlap = _sequence_overlap(hf_by_n[beam_width], mb_by_n[beam_width])
        assert overlap >= state.overlap_threshold, (
            f"Beam search overlap {overlap:.2%} is below threshold "
            f"{state.overlap_threshold:.2%} for beam width {beam_width}"
        )


# ---------------------------------------------------------------------------
# Prefill-logits-diff fixture: shared state formerly held in TestPrefillLogitsDiff.
# ---------------------------------------------------------------------------
class _PrefillState:
    def __init__(self):
        self.model_path = _resolve_model_path()
        assert self.model_path is not None
        self.prompt = os.environ.get("FLASHREC_DIFF_PROMPT", _DEFAULT_PROMPT)
        layout = _sid_layout_for(self.model_path)
        self.sid_token_range = layout.token_range
        self.codebook_sizes_s = layout.codebook_sizes
        self.boundary_s = layout.boundary_tokens
        self.boundary_ids = parse_int_list(layout.boundary_tokens) or []
        self.begin_id = int(self.boundary_ids[0])
        self.quantization = _quantization()
        self.device = torch.device("cuda:0")
        # FP8 W8A8 + FP8 KV pages vs a bf16 HF reference: absolute log-prob gap
        # is the loose bound, rank correlation the tight one. Near-tie top-1
        # swaps under FP8 are allowed if each winner sits in the other's top-2.
        default_atol = "0.65" if self.quantization == "fp8" else "0.25"
        self.max_abs_diff = float(
            os.environ.get("FLASHREC_DIFF_LOGIT_ATOL", default_atol)
        )
        top1_default = "0.0" if self.quantization == "fp8" else "1.0"
        self.min_top1_agree = float(
            os.environ.get("FLASHREC_DIFF_TOP1", top1_default)
        )
        self.topk = int(os.environ.get("FLASHREC_DIFF_TOPK", "10"))
        print(
            f"\nSID layout {layout.token_range}/{layout.codebook_sizes} "
            f"boundary {layout.boundary_tokens} quant={self.quantization} "
            f"atol={self.max_abs_diff}"
        )


def _prefill_prompt_input_ids(state: _PrefillState) -> List[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(state.model_path, trust_remote_code=True)
    messages = [{"role": "user", "content": state.prompt}]
    try:
        encoded = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        encoded = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    ids = [int(x) for x in encoded]
    if not ids or ids[-1] != state.begin_id:
        ids.append(state.begin_id)
    return ids


def _build_runner(state: _PrefillState) -> ModelEngine:
    config = BeamRecConfig(
        model_path=state.model_path,
        quantization=state.quantization,
        sid_token_range=state.sid_token_range,
        sid_codebook_sizes=state.codebook_sizes_s,
        sid_boundary_tokens=state.boundary_s,
        sid_vocab_file=os.environ.get("FLASHREC_DIFF_CATALOG", "").strip() or None,
        enable_cuda_graph=False,
        enable_graph_expand=False,
        enable_radix=False,
        enable_warmup=False,
        mem_fraction_static=0.35,
        max_running_requests=8,
        host_worker_threads=0,
    )
    return ModelEngine(config)


@pytest.fixture(scope="module")
def prefill_state():
    return _PrefillState()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not _resolve_model_path(),
    reason="set FLASHREC_DIFF_MODEL to a GenRec model dir",
)
def test_prefill_logprobs_match_hf(prefill_state: _PrefillState):
    state = prefill_state
    input_ids = _prefill_prompt_input_ids(state)
    assert input_ids[-1] == state.begin_id

    runner = _build_runner(state)
    try:
        mb_lp, cand_ids = _flashrec_prefill_logprobs(runner, input_ids)
    finally:
        del runner
        torch.cuda.empty_cache()
    assert cand_ids is not None, "restricted lm_head produced no candidate ids"

    hf_model = _load_hf_causal_lm_bf16(state.model_path, state.device)
    try:
        hf_lp = _hf_prefill_logprobs(hf_model, input_ids, cand_ids, state.device)
    finally:
        del hf_model
        torch.cuda.empty_cache()

    assert tuple(mb_lp.shape) == tuple(hf_lp.shape)

    diff = (mb_lp - hf_lp).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    k = min(state.topk, int(mb_lp.numel()))
    mb_top = torch.topk(mb_lp, k).indices
    hf_top = torch.topk(hf_lp, k).indices
    top1_agree = float(mb_top[0] == hf_top[0])
    topk_overlap = len(set(mb_top.tolist()) & set(hf_top.tolist())) / float(k)
    # Spearman over the restricted vocab: ties are not expected in log-probs.
    mb_rank = torch.argsort(torch.argsort(mb_lp)).float()
    hf_rank = torch.argsort(torch.argsort(hf_lp)).float()
    corr = float(torch.corrcoef(torch.stack([mb_rank, hf_rank]))[0, 1])

    print(f"\n{'=' * 60}")
    print(f"Prefill logits diff over {mb_lp.numel()} restricted tokens")
    print(f"{'=' * 60}")
    hf_margin = (
        float(hf_lp[int(hf_top[0])] - hf_lp[int(hf_top[1])]) if k >= 2 else float("inf")
    )
    print(f"  max |Δlogprob|   : {max_abs:.6f}")
    print(f"  mean |Δlogprob|  : {mean_abs:.6f}")
    print(f"  top-1 agreement  : {top1_agree:.0%}")
    print(f"  hf top-1 margin  : {hf_margin:.6f}")
    print(f"  top-{k} overlap    : {topk_overlap:.2%}")
    print(f"  rank corr        : {corr:.6f}")
    print(f"  flashrec top-{k}   : {mb_top.tolist()}")
    print(f"  hf top-{k}         : {hf_top.tolist()}")

    if top1_agree < 1.0 and state.quantization == "fp8":
        mb_top2 = set(int(x) for x in mb_top[:2].tolist())
        hf_top2 = set(int(x) for x in hf_top[:2].tolist())
        assert (
            int(hf_top[0]) in mb_top2
        ), f"HF top-1 {int(hf_top[0])} not in FlashRec top-2 {sorted(mb_top2)}"
        assert (
            int(mb_top[0]) in hf_top2
        ), f"FlashRec top-1 {int(mb_top[0])} not in HF top-2 {sorted(hf_top2)}"
    else:
        assert top1_agree >= state.min_top1_agree, (
            f"prefill top-1 token disagrees: flashrec={int(mb_top[0])} "
            f"hf={int(hf_top[0])}"
        )
    assert corr >= 0.99, f"prefill rank correlation {corr:.6f} below 0.99"
    assert (
        max_abs <= state.max_abs_diff
    ), f"max |Δlogprob| {max_abs:.6f} exceeds {state.max_abs_diff:.6f}"
