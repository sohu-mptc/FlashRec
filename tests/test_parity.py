"""Parity: local formulas + optional live HTTP comparison against an SGLang beam server."""

from __future__ import annotations

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
import torch

from flashrec.search.expand import joint_select as be_joint_select
from flashrec.search.score import beam_score as be_score
from flashrec.search.trie import build_beam_valid_path as be_build


class TestAlgorithmParity:
    def test_score_formula(self):
        assert round(abs(be_score(-3.5, 3, 1.0) - -3.5 / 3.0), 7) == 0
        assert round(abs(be_score(-1.0, 1, 0.6) - -1.0), 7) == 0
        assert round(abs(be_score(-9.0, 5, 1.2) - -9.0 / (5**1.2)), 7) == 0

    def test_joint_select_deterministic(self):
        torch.manual_seed(0)
        cum = torch.randn(4)
        lp = torch.randn(4, 8)
        toks = torch.randint(0, 50, (4, 8))
        stop = torch.tensor([3, 7, 11], dtype=torch.int64)
        a = be_joint_select(cum, lp, toks, stop, 4)
        b = be_joint_select(cum, lp, toks, stop, 4)
        assert int(a.num_survivors) == int(b.num_survivors)
        torch.testing.assert_close(a.next_tokens, b.next_tokens)

    def test_trie_factory(self):
        ids = list(range(200, 212))
        a = be_build(codebook_sizes=[4, 4, 4], special_token_ids=ids)
        assert a.mode == "codebook"
        assert a.max_depth == 3
        assert a.allowed_next([], 1) == set(ids[4:8])


@pytest.mark.skipif(
    not os.environ.get("SGLANG_BEAM_URL"),
    reason="set SGLANG_BEAM_URL to compare against a running SGLang beam server",
)
class TestLiveServerParity:
    def test_chat_choices_and_scores(self):
        url = os.environ["SGLANG_BEAM_URL"].rstrip("/") + "/v1/chat/completions"
        engine_url = os.environ.get(
            "FLASHREC_URL", os.environ.get("BEAM_ENGINE_URL", "")
        ).rstrip("/")
        payload = {
            "model": "x",
            "messages": [
                {
                    "role": "user",
                    "content": "predict next: <|sid_begin|><s_a_0><s_b_0><s_c_0><|sid_end|>",
                }
            ],
            "n": int(os.environ.get("BEAM_PARITY_N", "4")),
            "max_tokens": 5,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=120) as resp:
                sgl = json.loads(resp.read().decode("utf-8"))
        except URLError as exc:
            pytest.skip(f"sglang server unreachable: {exc}")

        assert "choices" in sgl
        assert len(sgl["choices"]) >= 1
        assert "sglext" in sgl["choices"][0]

        if not engine_url:
            return
        req2 = Request(
            engine_url + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req2, timeout=180) as resp:
            eng = json.loads(resp.read().decode("utf-8"))
        sgl_texts = [c["message"]["content"] for c in sgl["choices"]]
        eng_texts = [c["message"]["content"] for c in eng["choices"]]
        assert sgl_texts == eng_texts
        sgl_scores = [c["sglext"]["sequence_score"] for c in sgl["choices"]]
        eng_scores = [c["sglext"]["sequence_score"] for c in eng["choices"]]
        for a, b in zip(sgl_scores, eng_scores):
            assert round(abs(float(a) - float(b)), 4) == 0
