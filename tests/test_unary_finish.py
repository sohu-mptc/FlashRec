import torch

from flashrec.config import BeamRecConfig
from flashrec.core import BeamRequest
from flashrec.scheduler.scheduler import BeamRecEngine
from flashrec.search.state import BeamSearchList
from flashrec.search.trie import BeamValidPathTrie
from flashrec.search.unary import (
    apply_unary_finish,
    can_unary_finish,
    try_unary_finish_request,
)


def _sid_end_trie() -> BeamValidPathTrie:
    """Two first-level codes, then unary second and unary sid_end."""
    trie = BeamValidPathTrie(mode="trie")
    trie.children[()] = {10, 11}
    trie.children[(10,)] = {20}
    trie.children[(11,)] = {20}
    trie.children[(10, 20)] = {30}
    trie.children[(11, 20)] = {30}
    trie.children[(10, 20, 30)] = set()
    trie.children[(11, 20, 30)] = set()
    trie.max_depth = 3
    trie.all_tokens = {10, 11, 20, 30}
    trie.finalize_index(token_base=10, vocab_size=21)
    return trie


def _bushy_last_trie() -> BeamValidPathTrie:
    trie = BeamValidPathTrie(mode="trie")
    trie.children[()] = {10}
    trie.children[(10,)] = {20, 21}
    trie.children[(10, 20)] = set()
    trie.children[(10, 21)] = set()
    trie.max_depth = 2
    trie.all_tokens = {10, 20, 21}
    trie.finalize_index(token_base=10, vocab_size=12)
    return trie


def _beam_at(trie: BeamValidPathTrie, prefixes, max_new: int) -> BeamSearchList:
    bw = len(prefixes)
    cur = len(prefixes[0])
    bl = BeamSearchList()
    bl.incomplete = bl.ensure_empty_stubs(bw)
    bl.init_token_ids(max_new, device=torch.device("cpu"))
    for i, prefix in enumerate(prefixes):
        bl.token_ids[i, : len(prefix)] = torch.tensor(prefix, dtype=torch.int64)
    bl.cur_len = cur
    bl.dense_authoritative = True
    bl.cum_logprobs = torch.tensor(
        [-0.1 * (i + 1) for i in range(bw)], dtype=torch.float32
    )
    bl.last_tokens = bl.token_ids[:, cur - 1].clone()
    bl.node_ids = torch.tensor(
        [trie._prefix_to_node[tuple(p)] for p in prefixes], dtype=torch.int64
    )
    return bl


def _request(trie, prefixes, max_new, temperature=0.0) -> BeamRequest:
    req = BeamRequest(
        input_ids=[1, 2, 3],
        beam_width=len(prefixes),
        max_new_tokens=max_new,
        temperature=temperature,
    )
    req.beam_list = _beam_at(trie, prefixes, max_new)
    return req


class TestUnaryFinish:
    def test_last_sid_end_appends_without_changing_scores(self):
        trie = _sid_end_trie()
        bl = _beam_at(trie, [[10, 20], [11, 20]], max_new=3)
        scores = bl.cum_logprobs.clone()
        assert can_unary_finish(bl.node_ids, trie._outdeg, trie._unary_child, 1)
        apply_unary_finish(bl, 1, trie._unary_token, trie._unary_child)
        assert bl.generated_len() == 3
        assert bl.token_ids[:, 2].tolist() == [30, 30]
        assert bl.last_tokens.tolist() == [30, 30]
        assert torch.equal(bl.cum_logprobs, scores)

    def test_bushy_last_is_not_unary(self):
        trie = _bushy_last_trie()
        bl = _beam_at(trie, [[10]], max_new=2)
        cur = bl.cur_len
        toks = bl.token_ids.clone()
        assert not can_unary_finish(bl.node_ids, trie._outdeg, trie._unary_child, 1)
        assert bl.cur_len == cur
        assert torch.equal(bl.token_ids, toks)

    def test_remain_two_walks_both_unary_columns(self):
        trie = _sid_end_trie()
        bl = _beam_at(trie, [[10], [11]], max_new=3)
        assert can_unary_finish(bl.node_ids, trie._outdeg, trie._unary_child, 2)
        apply_unary_finish(bl, 2, trie._unary_token, trie._unary_child)
        assert bl.generated_len() == 3
        assert bl.token_ids[0, 1:].tolist() == [20, 30]
        assert bl.token_ids[1, 1:].tolist() == [20, 30]

    def test_try_finish_request_marks_length(self):
        trie = _sid_end_trie()
        req = _request(trie, [[10, 20], [11, 20]], max_new=3)
        assert try_unary_finish_request(req, trie, 3)
        assert req.finished
        assert req.finish_reason.type == "length"
        assert req.finish_reason.length == 3
        assert req.beam_list.generated_len() == 3

    def test_temperature_blocks_unary_finish(self):
        trie = _sid_end_trie()
        req = _request(trie, [[10, 20]], max_new=3, temperature=0.7)
        assert not try_unary_finish_request(req, trie, 3)
        assert not req.finished
        assert req.beam_list.generated_len() == 2

    def test_scheduler_skips_forward(self):
        trie = _sid_end_trie()
        req = _request(trie, [[10, 20], [11, 20]], max_new=3)

        class _Eng:
            config = BeamRecConfig(model_path="x", enable_unary_finish=True)
            valid_path = trie
            forwarded = False

            def generation_token_limit(self, n):
                return int(n)

            def _unary_finish_live(self, reqs):
                return BeamRecEngine._unary_finish_live(self, reqs)

        eng = _Eng()
        leftover = BeamRecEngine._unary_finish_live(eng, [req])
        assert leftover == []
        assert req.finished
        BeamRecEngine._decode_tick(eng, [req])
        assert not eng.forwarded


class TestUnaryFinishDisabled:
    def test_flag_off_leaves_request_live(self):
        trie = _sid_end_trie()
        req = _request(trie, [[10, 20]], max_new=3)

        class _Eng:
            config = BeamRecConfig(model_path="x", enable_unary_finish=False)
            valid_path = trie

            def generation_token_limit(self, n):
                return int(n)

        leftover = BeamRecEngine._unary_finish_live(_Eng(), [req])
        assert leftover == [req]
        assert not req.finished


def test_cli_disable_unary_finish():
    from flashrec.cli import _build_parser, config_from_args

    args = _build_parser().parse_args(["--model-path", "m", "--disable-unary-finish"])
    cfg = config_from_args(args)
    assert cfg.enable_unary_finish is False
    args_on = _build_parser().parse_args(["--model-path", "m"])
    assert config_from_args(args_on).enable_unary_finish is True
