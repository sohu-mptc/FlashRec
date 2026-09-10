"""Cascade decode attention == paged decode == manual SDPA on mixed batches.

Layout mirrors a beam-decode wave: every beam row of a request shares the
request's prompt pages and owns its decode-history pages.
"""

import math

import pytest
import torch

from flashrec.core import CascadeMeta, ForwardBatch
from flashrec.engine.graph import DecodeGraphRunner

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _backend(device):
    from flashrec.attention.flashinfer import AttentionBackend
    from flashrec.kvcache.pool import TokenToKVPool

    pool = TokenToKVPool(
        num_layers=1,
        num_tokens=8192,
        num_kv_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
        device=device,
    )
    torch.manual_seed(0)
    pool.k[0].normal_()
    pool.v[0].normal_()
    attn = AttentionBackend(pool, num_qo_heads=32, num_kv_heads=8, head_dim=128)
    if attn._fi_decode is None or not attn.supports_cascade:
        pytest.skip("FlashInfer cascade unavailable")
    return attn, pool


def _scenario(device):
    """(widths, prompt_lens, dec_len, prompt_pages, dec_pages, rows_meta)."""
    widths = [8, 4, 5]
    prompt_lens = [37, 21, 64]
    dec_len = 3
    nxt = 1  # page 0 is the dummy slot
    prompt_pages = []
    dec_pages = []
    for w, pl in zip(widths, prompt_lens):
        prompt_pages.append(torch.arange(nxt, nxt + pl, dtype=torch.int32, device=device))
        nxt += pl
        dec_pages.append(
            torch.arange(nxt, nxt + w * dec_len, dtype=torch.int32, device=device).view(
                w, dec_len
            )
        )
        nxt += w * dec_len
    return widths, prompt_lens, dec_len, prompt_pages, dec_pages


def _ref(pool, q, widths, prompt_pages, dec_pages):
    k_all = pool.k[0].float()
    v_all = pool.v[0].float()
    scale = 1.0 / math.sqrt(pool.head_dim)
    rep = q.shape[1] // pool.num_kv_heads
    outs = []
    row = 0
    for r, w in enumerate(widths):
        for i in range(w):
            pages = torch.cat([prompt_pages[r], dec_pages[r][i]]).long()
            k = k_all[pages].repeat_interleave(rep, dim=1)
            v = v_all[pages].repeat_interleave(rep, dim=1)
            qi = q[row].float()
            att = torch.einsum("hd,lhd->hl", qi, k) * scale
            outs.append(torch.einsum("hl,lhd->hd", att.softmax(-1), v))
            row += 1
    return torch.stack(outs)


def _decode_batch(device, widths, prompt_lens, dec_len, prompt_pages, dec_pages):
    n_rows = sum(widths)
    kv_rows = []
    seq_lens = []
    for r, w in enumerate(widths):
        for i in range(w):
            kv_rows.append(torch.cat([prompt_pages[r], dec_pages[r][i]]))
            seq_lens.append(prompt_lens[r] + dec_len)
    seq = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    dummy = torch.zeros(n_rows, dtype=torch.int64, device=device)
    return ForwardBatch(
        input_ids=dummy,
        req_pool_indices=dummy,
        seq_lens=seq,
        seq_lens_cpu=seq.cpu(),
        positions=dummy,
        out_cache_loc=dummy,
        is_prefill=False,
        extend_seq_lens=[1] * n_rows,
        kv_indices=torch.cat(kv_rows),
    )


def _cascade_meta(widths, prompt_lens, dec_len, prompt_pages, dec_pages):
    n_rows = sum(widths)
    qo0 = torch.tensor(
        [0] + list(torch.tensor(widths).cumsum(0)), dtype=torch.int32
    )
    kv0 = torch.tensor(
        [0] + list(torch.tensor(prompt_lens).cumsum(0)), dtype=torch.int32
    )
    return CascadeMeta(
        qo_indptr=[qo0, torch.arange(n_rows + 1, dtype=torch.int32)],
        kv_indptr=[
            kv0,
            torch.arange(n_rows + 1, dtype=torch.int32) * dec_len,
        ],
        kv_indices=[
            torch.cat(prompt_pages),
            torch.cat([p.reshape(-1) for p in dec_pages]),
        ],
        last_page_len=[
            torch.ones(len(widths), dtype=torch.int32),
            torch.ones(n_rows, dtype=torch.int32),
        ],
    )


@cuda
def test_cascade_matches_paged_and_reference():
    device = torch.device("cuda")
    attn, pool = _backend(device)
    widths, prompt_lens, dec_len, prompt_pages, dec_pages = _scenario(device)
    n_rows = sum(widths)
    torch.manual_seed(1)
    q = torch.randn(n_rows, 32, 128, dtype=torch.bfloat16, device=device)
    dummy_kv = torch.zeros(n_rows, 8, 128, dtype=torch.bfloat16, device=device)

    batch = _decode_batch(device, widths, prompt_lens, dec_len, prompt_pages, dec_pages)
    attn.prepare(batch)
    assert not attn._cascade_active
    o_paged = attn.forward(q, dummy_kv, dummy_kv, 0, batch, skip_store=True).float()

    batch.cascade = _cascade_meta(widths, prompt_lens, dec_len, prompt_pages, dec_pages)
    attn.prepare(batch)
    assert attn._cascade_active, "cascade plan fell back to paged decode"
    o_casc = attn.forward(q, dummy_kv, dummy_kv, 0, batch, skip_store=True).float()

    o_ref = _ref(pool, q, widths, prompt_pages, dec_pages)
    assert (o_paged - o_ref).abs().max().item() < 2e-2
    assert (o_casc - o_ref).abs().max().item() < 2e-2
    assert (o_casc - o_paged).abs().max().item() < 2e-2


@cuda
def test_cascade_l0_plan_reused_across_steps():
    device = torch.device("cuda")
    attn, pool = _backend(device)
    widths, prompt_lens, dec_len, prompt_pages, dec_pages = _scenario(device)
    n_rows = sum(widths)
    torch.manual_seed(2)
    q = torch.randn(n_rows, 32, 128, dtype=torch.bfloat16, device=device)
    dummy_kv = torch.zeros(n_rows, 8, 128, dtype=torch.bfloat16, device=device)
    batch = _decode_batch(device, widths, prompt_lens, dec_len, prompt_pages, dec_pages)
    meta1 = _cascade_meta(widths, prompt_lens, dec_len, prompt_pages, dec_pages)
    batch.cascade = meta1
    attn.prepare(batch)
    assert attn._cascade_active
    o1 = attn.forward(q, dummy_kv, dummy_kv, 0, batch, skip_store=True).float()
    l0_sig = attn._cascade_l0_sig
    assert l0_sig is not None
    meta2 = _cascade_meta(widths, prompt_lens, dec_len, prompt_pages, dec_pages)
    meta2.qo_indptr[0] = meta1.qo_indptr[0]
    meta2.kv_indptr[0] = meta1.kv_indptr[0]
    meta2.kv_indices[0] = meta1.kv_indices[0]
    meta2.last_page_len[0] = meta1.last_page_len[0]
    batch.cascade = meta2
    attn.prepare(batch)
    assert attn._cascade_l0_sig == l0_sig
    o2 = attn.forward(q, dummy_kv, dummy_kv, 0, batch, skip_store=True).float()
    assert (o1 - o2).abs().max().item() < 2e-2


@cuda
def test_cascade_plan_failure_falls_back():
    device = torch.device("cuda")
    attn, _ = _backend(device)
    widths, prompt_lens, dec_len, prompt_pages, dec_pages = _scenario(device)
    batch = _decode_batch(device, widths, prompt_lens, dec_len, prompt_pages, dec_pages)
    meta = _cascade_meta(widths, prompt_lens, dec_len, prompt_pages, dec_pages)
    meta.kv_indptr = [meta.kv_indptr[0][:1], meta.kv_indptr[1]]  # malformed
    batch.cascade = meta
    attn.prepare(batch)
    assert not attn._cascade_active
    assert not attn.supports_cascade  # disabled after failure, paged path planned


@cuda
def test_cascade_cuda_graph_matches_eager():
    device = torch.device("cuda")
    attn, pool = _backend(device)
    widths, prompt_lens, dec_len, prompt_pages, dec_pages = _scenario(device)
    n_rows = sum(widths)
    torch.manual_seed(3)
    q = torch.randn(n_rows, 32, 128, dtype=torch.bfloat16, device=device)
    dummy_kv = torch.zeros(n_rows, 8, 128, dtype=torch.bfloat16, device=device)
    batch = _decode_batch(device, widths, prompt_lens, dec_len, prompt_pages, dec_pages)
    batch.cascade = _cascade_meta(widths, prompt_lens, dec_len, prompt_pages, dec_pages)
    attn.prepare(batch)
    assert attn._cascade_active
    o_eager = attn.forward(q, dummy_kv, dummy_kv, 0, batch, skip_store=True).float()

    attn.init_graph_wrappers(
        [n_rows], max_seq_len=256, max_cascade_reqs=len(widths) + 1
    )
    assert n_rows in attn._cascade_graph_wrappers
    attn.begin_graph_cascade_decode(n_rows)
    try:
        attn.prepare(batch)
        assert attn._cascade_active
        o_graph = attn.forward(q, dummy_kv, dummy_kv, 0, batch, skip_store=True).float()
    finally:
        attn.end_graph_cascade_decode()
    assert (o_graph - o_eager).abs().max().item() < 2e-2
    o_ref = _ref(pool, q, widths, prompt_pages, dec_pages)
    assert (o_graph - o_ref).abs().max().item() < 2e-2


@cuda
def test_cascade_cuda_graph_padding_matches_eager():
    device = torch.device("cuda")
    attn, _ = _backend(device)
    widths, prompt_lens, dec_len, prompt_pages, dec_pages = _scenario(device)
    n_rows = sum(widths)
    capture_bs = 32
    torch.manual_seed(4)
    q = torch.randn(n_rows, 32, 128, dtype=torch.bfloat16, device=device)
    dummy_kv = torch.zeros(n_rows, 8, 128, dtype=torch.bfloat16, device=device)
    batch = _decode_batch(device, widths, prompt_lens, dec_len, prompt_pages, dec_pages)
    batch.cascade = _cascade_meta(widths, prompt_lens, dec_len, prompt_pages, dec_pages)
    attn.prepare(batch)
    o_eager = attn.forward(q, dummy_kv, dummy_kv, 0, batch, skip_store=True).float()

    attn.init_graph_wrappers(
        [capture_bs], max_seq_len=256, max_cascade_reqs=len(widths) + 1
    )
    q_pad = torch.zeros(capture_bs, 32, 128, dtype=torch.bfloat16, device=device)
    q_pad[:n_rows].copy_(q)
    dummy_kv_pad = torch.zeros(
        capture_bs, 8, 128, dtype=torch.bfloat16, device=device
    )
    attn.begin_graph_cascade_decode(capture_bs)
    try:
        attn.prepare(batch)
        assert attn._cascade_active
        o_graph = attn.forward(
            q_pad, dummy_kv_pad, dummy_kv_pad, 0, batch, skip_store=True
        ).float()
    finally:
        attn.end_graph_cascade_decode()
    assert (o_graph[:n_rows] - o_eager).abs().max().item() < 2e-2


@cuda
def test_cascade_graph_runner_replay_matches_eager():
    device = torch.device("cuda")
    attn, _ = _backend(device)
    widths, prompt_lens, dec_len, prompt_pages, dec_pages = _scenario(device)
    n_rows = sum(widths)
    torch.manual_seed(5)
    q = torch.randn(n_rows, 32, 128, dtype=torch.bfloat16, device=device)
    dummy_kv = torch.zeros(n_rows, 8, 128, dtype=torch.bfloat16, device=device)
    batch = _decode_batch(device, widths, prompt_lens, dec_len, prompt_pages, dec_pages)
    batch.cascade = _cascade_meta(widths, prompt_lens, dec_len, prompt_pages, dec_pages)
    attn.prepare(batch)
    o_eager = attn.forward(q, dummy_kv, dummy_kv, 0, batch, skip_store=True).float()

    attn.init_graph_wrappers(
        [n_rows], max_seq_len=256, max_cascade_reqs=len(widths) + 1
    )
    q_static = q.clone()
    kv_static = dummy_kv.clone()

    def model_fn(fb):
        return attn.forward(
            q_static, kv_static, kv_static, 0, fb, skip_store=True
        ).reshape(fb.n_rows, -1)

    runner = DecodeGraphRunner(
        device=device,
        max_bs=n_rows,
        max_seq_len=256,
        capture_bs=[n_rows],
        attn=attn,
    )
    runner.capture(model_fn, attn.prepare)
    assert n_rows in runner.graphs
    assert n_rows in runner.graphs_cascade, "cascade CUDA graph was not captured"
    out, _ = runner.replay(batch, attn.prepare, cascade=True)
    o_graph = out.view(n_rows, 32, 128).float()
    assert (o_graph - o_eager).abs().max().item() < 2e-2


def test_can_replay_cascade_reserves_dummy_slot():
    class _Attn:
        _max_cascade_reqs = 4

    runner = DecodeGraphRunner.__new__(DecodeGraphRunner)
    runner.attn = _Attn()
    runner.enabled = True
    runner.capture_bs = [32]
    runner.graphs = {32: object()}
    runner.graphs_cascade = {32: object()}
    assert runner.can_replay_cascade(17, 3)
    assert not runner.can_replay_cascade(17, 4)
    assert runner.can_replay_cascade(32, 4)
    assert not runner.can_replay_cascade(33, 1)
