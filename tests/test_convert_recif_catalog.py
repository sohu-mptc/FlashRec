import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from convert_recif_catalog import (  # noqa: E402
    convert_file,
    format_sid_key,
    main,
    parse_sid_codes,
)


class TestParseSidCodes:
    def test_packed_integer(self):
        packed = 12 * 8192 * 8192 + 34 * 8192 + 56
        assert parse_sid_codes(str(packed)) == (12, 34, 56)
        assert parse_sid_codes(packed) == (12, 34, 56)

    def test_comma_key_strips_end_level(self):
        assert parse_sid_codes("12,34,56") == (12, 34, 56)
        assert parse_sid_codes("12,34,56,1") == (12, 34, 56)

    def test_rejects_out_of_range(self):
        assert parse_sid_codes(str(8192**3)) is None
        assert parse_sid_codes("12,34,8192") is None
        assert parse_sid_codes("not-a-sid") is None


class TestConvertFile:
    def test_packed_to_l4(self):
        packed = 1 * 8192 * 8192 + 2 * 8192 + 3
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sid2pid.json"
            out = Path(tmp) / "out.json"
            src.write_text(json.dumps({str(packed): [{"pid": 7}]}), encoding="utf-8")
            n_in, n_out, bad = convert_file(src, out, levels=4)
            assert (n_in, n_out, bad) == (1, 1, 0)
            data = json.loads(out.read_text(encoding="utf-8"))
            assert data == {"1,2,3,1": 1}

    def test_levels_3(self):
        assert format_sid_key((1, 2, 3), levels=3) == "1,2,3"
        packed = 0 * 8192 * 8192 + 0 * 8192 + 1
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sid2pid.json"
            out = Path(tmp) / "out.json"
            src.write_text(json.dumps({str(packed): 1, "0,0,1,1": 1}), encoding="utf-8")
            _, n_out, bad = convert_file(src, out, levels=3)
            assert n_out == 1
            assert bad == 0
            assert json.loads(out.read_text(encoding="utf-8")) == {"0,0,1": 1}

    def test_data_dir_cli(self):
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
                    "--data-dir",
                    str(data_dir),
                    "--out-dir",
                    str(out_dir),
                    "--task",
                    "video",
                    "--levels",
                    "4",
                ]
            )
            assert rc == 0
            out = out_dir / "sid2pid_beamrec_l4.json"
            assert out.is_file()
            assert json.loads(out.read_text(encoding="utf-8")) == {"4,5,6,1": 1}
