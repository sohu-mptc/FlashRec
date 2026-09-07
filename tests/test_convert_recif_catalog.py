import json
import tempfile
from pathlib import Path

from flashrec.catalog import (
    convert_file,
    format_sid_key,
    packed_depth,
    parse_sid_codes,
    script_main,
)
from flashrec.cli import _build_parser, main


class TestParseSidCodes:
    def test_packed_integer(self):
        packed = 12 * 8192 * 8192 + 34 * 8192 + 56
        assert parse_sid_codes(str(packed)) == (12, 34, 56)
        assert parse_sid_codes(packed) == (12, 34, 56)

    def test_comma_key_keeps_all_levels(self):
        assert parse_sid_codes("12,34,56") == (12, 34, 56)
        assert parse_sid_codes("12,34,56,1") == (12, 34, 56, 1)
        assert parse_sid_codes("12,34,56,7") == (12, 34, 56, 7)

    def test_rejects_out_of_range(self):
        assert parse_sid_codes(str(8192**3)) is None
        assert parse_sid_codes("12,34,8192") is None
        assert parse_sid_codes("not-a-sid") is None

    def test_packed_depth_grows_past_recif(self):
        assert packed_depth(8192**3 - 1) == 3
        assert packed_depth(8192**3) == 4


class TestConvertFile:
    def test_packed_auto_appends_sid_end(self):
        packed = 1 * 8192 * 8192 + 2 * 8192 + 3
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sid2pid.json"
            out = Path(tmp) / "out.json"
            src.write_text(json.dumps({str(packed): [{"pid": 7}]}), encoding="utf-8")
            n_in, n_out, bad, levels = convert_file(src, out)
            assert (n_in, n_out, bad, levels) == (1, 1, 0, 4)
            data = json.loads(out.read_text(encoding="utf-8"))
            assert data == {"1,2,3,1": 1}

    def test_levels_3_override(self):
        assert format_sid_key((1, 2, 3), levels=3) == "1,2,3"
        packed = 0 * 8192 * 8192 + 0 * 8192 + 1
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sid2pid.json"
            out = Path(tmp) / "out.json"
            src.write_text(json.dumps({str(packed): 1, "0,0,1,1": 1}), encoding="utf-8")
            _, n_out, bad, levels = convert_file(src, out, levels=3)
            assert (n_out, bad, levels) == (1, 0, 3)
            assert json.loads(out.read_text(encoding="utf-8")) == {"0,0,1": 1}

    def test_comma_source_keeps_native_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sid2pid.json"
            out = Path(tmp) / "out.json"
            src.write_text(json.dumps({"1,2,3,7": 1, "4,5,6,8": 1}), encoding="utf-8")
            _, n_out, bad, levels = convert_file(src, out)
            assert (n_out, bad, levels) == (2, 0, 4)
            assert json.loads(out.read_text(encoding="utf-8")) == {
                "1,2,3,7": 1,
                "4,5,6,8": 1,
            }

    def test_packed_four_layer_does_not_append_end(self):
        packed = 1 * (8192**3) + 2 * (8192**2) + 3 * 8192 + 4
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sid2pid.json"
            out = Path(tmp) / "out.json"
            src.write_text(json.dumps({str(packed): 1}), encoding="utf-8")
            _, n_out, bad, levels = convert_file(src, out)
            assert (n_out, bad, levels) == (1, 0, 4)
            assert json.loads(out.read_text(encoding="utf-8")) == {"1,2,3,4": 1}


class TestFlashrecCatalogCli:
    def test_help_lists_catalog(self):
        text = _build_parser().format_help()
        assert "--catalog PATH" in text
        assert "flashrec --catalog" in text
        assert "Does not need --model-path" in text

    def test_catalog_data_dir(self):
        packed = 4 * 8192 * 8192 + 5 * 8192 + 6
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "benchmark_data"
            out_dir = Path(tmp) / "catalogs"
            data_dir.mkdir()
            (data_dir / "sid2pid.json").write_text(
                json.dumps({str(packed): []}), encoding="utf-8"
            )
            rc = main(
                [
                    "--catalog",
                    str(data_dir),
                    "--catalog-out",
                    str(out_dir),
                    "--catalog-task",
                    "video",
                ]
            )
            assert rc == 0
            out = out_dir / "sid2pid_beamrec_l4.json"
            assert out.is_file()
            assert json.loads(out.read_text(encoding="utf-8")) == {"4,5,6,1": 1}

    def test_script_wrapper_data_dir(self):
        packed = 4 * 8192 * 8192 + 5 * 8192 + 6
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "benchmark_data"
            out_dir = Path(tmp) / "catalogs"
            data_dir.mkdir()
            (data_dir / "sid2pid.json").write_text(
                json.dumps({str(packed): []}), encoding="utf-8"
            )
            rc = script_main(
                [
                    "--data-dir",
                    str(data_dir),
                    "--out-dir",
                    str(out_dir),
                    "--task",
                    "video",
                ]
            )
            assert rc == 0
            assert json.loads(
                (out_dir / "sid2pid_beamrec_l4.json").read_text(encoding="utf-8")
            ) == {"4,5,6,1": 1}
