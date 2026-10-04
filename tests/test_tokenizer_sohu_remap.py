"""SoHu-0.6B packed window vs OneRec real codebook names."""

from flashrec.tokenizer import TokenizerAdapter, _compressed_sohu_window


def test_compressed_window_is_512_only():
    assert _compressed_sohu_window([512, 512, 512])
    assert not _compressed_sohu_window([8192, 8192, 8192, 2])
    assert not _compressed_sohu_window(None)


def test_remap_rewrites_only_the_packed_sa_window():
    tok = TokenizerAdapter.__new__(TokenizerAdapter)
    base = 151669
    tok._piece = {base + i: f"<s_a_{i}>" for i in range(1600)}
    tok._piece[159861] = "<s_b_0>"
    tok._remap_sohu_compressed_codebook_pieces()
    assert tok._piece[base + 565] == "<s_b_53>"
    assert tok._piece[base + 1024 + 499] == "<s_c_499>"
    assert tok._piece[base + 1536] == "<s_a_1536>"
    assert tok._piece[159861] == "<s_b_0>"
