"""SID compact spec and BeamRecConfig expansion."""

from __future__ import annotations

import pytest

from flashrec.config import BeamRecConfig, parse_sid_spec


class TestParseSidSpec:
    def test_recif_four_level_boundary_is_last_codebook(self):
        token_range, sizes, boundary = parse_sid_spec("151669:176246/8192,8192,8192,2")
        assert token_range == "151669:176246"
        assert sizes == "8192,8192,8192,2"
        assert boundary == "176245,176246"

    def test_depth3_boundary_follows_range(self):
        token_range, sizes, boundary = parse_sid_spec("10:21/4,4,4")
        assert token_range == "10:21"
        assert sizes == "4,4,4"
        assert boundary == "22,23"

    def test_whitespace_is_stripped(self):
        _, sizes, boundary = parse_sid_spec(" 10:23 / 4, 4, 4, 2 ")
        assert sizes == "4,4,4,2"
        assert boundary == "22,23"

    def test_missing_slash(self):
        with pytest.raises(ValueError):
            parse_sid_spec("151669:176246")

    def test_codebook_sum_exceeds_range(self):
        with pytest.raises(ValueError):
            parse_sid_spec("0:3/2,2,2")

    def test_boundary_override_after_at(self):
        token_range, sizes, boundary = parse_sid_spec(
            "151669:153204/512,512,512@176245,176246"
        )
        assert token_range == "151669:153204"
        assert sizes == "512,512,512"
        assert boundary == "176245,176246"

    def test_depth3_without_override_still_follows_range(self):
        _, _, boundary = parse_sid_spec("151669:153204/512,512,512")
        assert boundary == "153205,153206"

    def test_bad_boundary_override(self):
        with pytest.raises(ValueError):
            parse_sid_spec("151669:153204/512,512,512@176245")


class TestBeamRecConfigSid:
    def test_sid_fills_the_three_fields(self):
        cfg = BeamRecConfig(
            model_path="/tmp/model",
            sid="151669:176246/8192,8192,8192,2",
        )
        assert cfg.sid_token_range == "151669:176246"
        assert cfg.sid_codebook_sizes == "8192,8192,8192,2"
        assert cfg.sid_boundary_tokens == "176245,176246"
        assert cfg.parsed_boundary_token_ids() == [176245, 176246]
        assert cfg.parsed_codebook_sizes() == [8192, 8192, 8192, 2]

    def test_matching_explicit_flags_are_ok(self):
        cfg = BeamRecConfig(
            model_path="/tmp/model",
            sid="151669:176246/8192,8192,8192,2",
            sid_token_range="151669:176246",
            sid_codebook_sizes="8192,8192,8192,2",
            sid_boundary_tokens="176245,176246",
        )
        assert cfg.sid_boundary_tokens == "176245,176246"

    def test_matching_list_boundary_is_ok(self):
        cfg = BeamRecConfig(
            model_path="/tmp/model",
            sid="151669:176246/8192,8192,8192,2",
            sid_boundary_tokens=[176245, 176246],
        )
        assert cfg.parsed_boundary_token_ids() == [176245, 176246]

    def test_conflicting_flag_raises(self):
        with pytest.raises(ValueError):
            BeamRecConfig(
                model_path="/tmp/model",
                sid="151669:176246/8192,8192,8192,2",
                sid_boundary_tokens="0,1",
            )

    def test_sohu06_override_fills_boundary(self):
        cfg = BeamRecConfig(
            model_path="/tmp/model",
            sid="151669:153204/512,512,512@176245,176246",
        )
        assert cfg.sid_token_range == "151669:153204"
        assert cfg.sid_codebook_sizes == "512,512,512"
        assert cfg.parsed_boundary_token_ids() == [176245, 176246]

    def test_unset_sid_leaves_fields_none(self):
        cfg = BeamRecConfig(model_path="/tmp/model")
        assert cfg.parsed_sid_token_ids() is None
        assert cfg.parsed_codebook_sizes() is None
        assert cfg.parsed_boundary_token_ids() is None

    def test_unary_finish_defaults_on(self):
        cfg = BeamRecConfig(model_path="/tmp/model")
        assert cfg.enable_unary_finish is True
