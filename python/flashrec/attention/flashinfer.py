"""FlashInfer paged attention (page_size=1) with SDPA fallback.

FP8 KV: plan with kv_data_type=float8_e4m3fn and tensor-core decode (SGLang
``should_use_tensor_core``). ``prepare()`` / ``plan()`` run *outside* CUDA Graph.

CUDA Graph decode (SGLang-style):
  * one ``BatchDecodeWithPagedKVCacheWrapper`` per captured batch size
  * ``use_cuda_graph=True`` with persistent indptr / indices / last_page_len
  * batch size is fixed for the wrapper lifetime; pad dummy rows (seq_len=1, page 0)
  * plan writes the persistent buffers; captured ``run()`` reads them on replay

Cascade CUDA Graph (wide-beam decode):
  * one ``MultiLevelCascadeAttentionWrapper(use_cuda_graph=True)`` per captured bs
  * pad to ``bs`` query rows and ``max_cascade_reqs`` L0 groups (one dummy request
    for leftover rows, remainder empty); both dummy paths attend page 0
  * cascade ``plan()`` stays outside the graph; ``run()`` + ``merge_state`` inside
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from flashrec.core import CascadeMeta, ForwardBatch
from flashrec.kvcache.pool import TokenToKVPool
from flashrec.profiler import trace_range

logger = logging.getLogger(__name__)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _use_tensor_core(kv_dtype: torch.dtype, n_rep: int) -> bool:
    if kv_dtype in (
        torch.float8_e4m3fn,
        getattr(torch, "float8_e5m2", torch.float8_e4m3fn),
    ):
        return True
    return n_rep >= 4


class _PinnedPlanBuf:
    """Reusable pinned CPU buffers so FlashInfer ``plan()`` never D2H-syncs."""

    def __init__(self):
        self.indptr: Optional[torch.Tensor] = None
        self.last: Optional[torch.Tensor] = None
        self.seq: Optional[torch.Tensor] = None

    def grow(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = max(int(n), 1)
        if self.indptr is None or int(self.indptr.numel()) < n + 1:
            self.indptr = torch.zeros(n + 1, dtype=torch.int32, pin_memory=True)
            self.last = torch.ones(n, dtype=torch.int32, pin_memory=True)
            self.seq = torch.zeros(n, dtype=torch.int32, pin_memory=True)
        return self.indptr[: n + 1], self.last[:n], self.seq[:n]


def _seq_lens_cpu32(
    seq_lens_cpu: torch.Tensor, n: int, out: torch.Tensor
) -> torch.Tensor:
    sl = seq_lens_cpu.view(-1)[:n]
    if sl.device.type != "cpu":
        sl = sl.to("cpu")
    if sl.dtype != torch.int32:
        sl = sl.to(dtype=torch.int32)
    out.copy_(sl)
    return out


class AttentionBackend:
    def __init__(
        self,
        kv_pool: TokenToKVPool,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        backend: str = "flashinfer",
        fi_backend: str = "fa2",
    ):
        self.kv_pool = kv_pool
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = max(num_qo_heads // max(num_kv_heads, 1), 1)
        self.device = kv_pool.device
        self.backend = backend
        self.fi_backend = str(fi_backend or "fa2")
        self.q_dtype = torch.bfloat16
        self._fi_prefill = None
        self._fi_decode = None
        self._workspace = None
        self._workspace_prefill = None
        self._workspace_graph = None
        self.prefill_overlap_ok = False
        self._use_tc = _use_tensor_core(kv_pool.dtype, self.n_rep)
        self._pin = _PinnedPlanBuf()
        self._pin_qo = _PinnedPlanBuf()
        self._graph_wrappers: Dict[int, object] = {}
        self._graph_bufs: Dict[int, dict] = {}
        self._graph_bs: Optional[int] = None
        self._active_decode = None
        # Two-level cascade decode (wide beams): created lazily on first use.
        self._fi_cascade = None
        self._cascade_active = False
        self.supports_cascade = False
        self._cascade_l0_sig: Optional[tuple] = None
        self._workspace_cascade = None
        self._cascade_graph_wrappers: Dict[int, object] = {}
        self._cascade_graph_bufs: Dict[int, dict] = {}
        self._cascade_graph_bs: Optional[int] = None
        self._active_cascade = None
        self._max_cascade_reqs: int = 0
        self._cascade_graph_l0_sig: Optional[tuple] = None
        if backend == "flashinfer" and kv_pool.device.type == "cuda":
            try:
                from flashinfer import (
                    BatchDecodeWithPagedKVCacheWrapper,
                    BatchPrefillWithPagedKVCacheWrapper,
                )

                self._workspace = torch.zeros(
                    128 * 1024 * 1024, dtype=torch.uint8, device=self.device
                )
                try:
                    self._workspace_prefill = torch.zeros(
                        128 * 1024 * 1024, dtype=torch.uint8, device=self.device
                    )
                except Exception:
                    self._workspace_prefill = self._workspace
                self._fi_prefill = BatchPrefillWithPagedKVCacheWrapper(
                    self._workspace_prefill, kv_layout="NHD", backend=self.fi_backend
                )
                self._fi_decode = BatchDecodeWithPagedKVCacheWrapper(
                    self._workspace,
                    use_tensor_cores=self._use_tc,
                    kv_layout="NHD",
                    backend=self.fi_backend,
                )
                self._active_decode = self._fi_decode
                self.prefill_overlap_ok = self._workspace_prefill is not self._workspace
                try:
                    from flashinfer.cascade import (  # noqa: F401
                        MultiLevelCascadeAttentionWrapper,
                    )

                    self.supports_cascade = True
                except Exception:
                    self.supports_cascade = False
                logger.info(
                    "FlashInfer wrappers ready: fi_backend=%s tensor_cores=%s",
                    self.fi_backend,
                    self._use_tc,
                )
            except Exception:
                self._fi_prefill = None
                self._fi_decode = None
                self._active_decode = None
                self.prefill_overlap_ok = False

    def init_graph_wrappers(
        self,
        capture_bs: List[int],
        max_seq_len: int,
        max_cascade_reqs: int = 0,
    ) -> None:
        """Create one CUDA-graph FlashInfer decode wrapper per captured batch size.

        When ``max_cascade_reqs > 0`` and cascade is available, also create a
        ``MultiLevelCascadeAttentionWrapper(use_cuda_graph=True)`` per ``bs``.
        Level-0 buffers are sized for ``max_cascade_reqs`` groups (real
        requests plus one dummy slot for pad rows); level-1 buffers are sized
        for ``bs`` rows.
        """
        if self._workspace is None or self.device.type != "cuda":
            raise RuntimeError("FlashInfer decode wrapper unavailable")
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        max_seq_len = max(int(max_seq_len), 1)
        # fa2 tensor-core decode runs on the prefill kernel whose tmp_v scales
        # ~147KiB per row; the shared 128MiB workspace overflows past bs~850.
        per_row = 192 * 1024
        need = max(int(b) for b in capture_bs) * per_row + (32 << 20)
        if need > self._workspace.numel():
            self._workspace_graph = torch.zeros(
                need, dtype=torch.uint8, device=self.device
            )
        else:
            self._workspace_graph = self._workspace
        for bs in capture_bs:
            bs = int(bs)
            if bs <= 0 or bs in self._graph_wrappers:
                continue
            indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
            indices = torch.zeros(
                bs * max_seq_len, dtype=torch.int32, device=self.device
            )
            last = torch.ones(bs, dtype=torch.int32, device=self.device)
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._workspace_graph,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=self._use_tc,
                paged_kv_indptr_buffer=indptr,
                paged_kv_indices_buffer=indices,
                paged_kv_last_page_len_buffer=last,
                backend=self.fi_backend,
            )
            self._graph_bufs[bs] = {
                "indptr": indptr,
                "indices": indices,
                "last": last,
            }
            self._graph_wrappers[bs] = wrapper
        logger.info("FlashInfer CUDA-graph wrappers: %s", sorted(self._graph_wrappers))
        self._init_cascade_graph_wrappers(
            capture_bs, max_seq_len, int(max_cascade_reqs)
        )

    def graph_bufs(self, bs: int) -> Optional[dict]:
        return self._graph_bufs.get(int(bs))

    def begin_graph_decode(self, bs: int) -> None:
        wrapper = self._graph_wrappers.get(int(bs))
        if wrapper is None:
            raise RuntimeError(f"no FlashInfer CUDA-graph wrapper for bs={bs}")
        self._graph_bs = int(bs)
        self._active_decode = wrapper

    def end_graph_decode(self) -> None:
        self._graph_bs = None
        self._active_decode = self._fi_decode

    def begin_graph_cascade_decode(self, bs: int) -> None:
        wrapper = self._cascade_graph_wrappers.get(int(bs))
        if wrapper is None:
            raise RuntimeError(f"no FlashInfer cascade CUDA-graph wrapper for bs={bs}")
        self._cascade_graph_bs = int(bs)
        self._active_cascade = wrapper

    def end_graph_cascade_decode(self) -> None:
        self._cascade_graph_bs = None
        self._active_cascade = self._fi_cascade

    def dummy_cascade_meta(self, bs: int) -> CascadeMeta:
        """All-page-0 cascade tables used while capturing a cascade CUDA graph."""
        bs = int(bs)
        device = self.device
        return CascadeMeta(
            qo_indptr=[
                torch.tensor([0, bs], dtype=torch.int32),
                torch.arange(bs + 1, dtype=torch.int32),
            ],
            kv_indptr=[
                torch.tensor([0, 1], dtype=torch.int32),
                torch.arange(bs + 1, dtype=torch.int32),
            ],
            kv_indices=[
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.zeros(bs, dtype=torch.int32, device=device),
            ],
            last_page_len=[
                torch.ones(1, dtype=torch.int32),
                torch.ones(bs, dtype=torch.int32),
            ],
        )

    def _ensure_cascade_workspace(self, need: int) -> None:
        need = max(int(need), 1)
        cur = 0 if self._workspace_cascade is None else int(self._workspace_cascade.numel())
        if cur >= need:
            return
        self._workspace_cascade = torch.zeros(
            need, dtype=torch.uint8, device=self.device
        )
        # Eager wrapper (if any) still holds the old buffer; rebuild next use.
        self._fi_cascade = None
        self._active_cascade = None
        self._cascade_l0_sig = None

    def _init_cascade_graph_wrappers(
        self, capture_bs: List[int], max_seq_len: int, max_cascade_reqs: int
    ) -> None:
        if not self.supports_cascade or max_cascade_reqs <= 0:
            return
        from flashinfer.cascade import MultiLevelCascadeAttentionWrapper

        max_seq_len = max(int(max_seq_len), 1)
        max_reqs = max(int(max_cascade_reqs), 1)
        sizes = [int(b) for b in capture_bs if int(b) > 0]
        if not sizes:
            return
        max_bs = max(sizes)
        # Two internal prefill wrappers share this buffer; tmp_v scales ~per row.
        per_row = 192 * 1024
        need = max(512 * 1024 * 1024, 2 * max_bs * per_row + (32 << 20))
        try:
            self._ensure_cascade_workspace(need)
        except Exception as exc:
            logger.warning("cascade CUDA-graph workspace alloc failed: %s", exc)
            return
        self._max_cascade_reqs = max_reqs
        created = []
        for bs in sizes:
            if bs in self._cascade_graph_wrappers:
                continue
            try:
                l0_qo = torch.zeros(max_reqs + 1, dtype=torch.int32, device=self.device)
                l0_kv = torch.zeros(max_reqs + 1, dtype=torch.int32, device=self.device)
                l0_idx = torch.zeros(
                    max_reqs * max_seq_len, dtype=torch.int32, device=self.device
                )
                l0_last = torch.ones(max_reqs, dtype=torch.int32, device=self.device)
                l1_qo = torch.arange(bs + 1, dtype=torch.int32, device=self.device)
                l1_kv = torch.arange(bs + 1, dtype=torch.int32, device=self.device)
                l1_idx = torch.zeros(
                    bs * max_seq_len, dtype=torch.int32, device=self.device
                )
                l1_last = torch.ones(bs, dtype=torch.int32, device=self.device)
                # Dummy layout: one request covering all ``bs`` rows on page 0.
                l0_qo[1:] = bs
                l0_kv[1:] = 1
                wrapper = MultiLevelCascadeAttentionWrapper(
                    2,
                    self._workspace_cascade,
                    kv_layout="NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf_arr=[l0_qo, l1_qo],
                    paged_kv_indptr_buf_arr=[l0_kv, l1_kv],
                    paged_kv_indices_buf_arr=[l0_idx, l1_idx],
                    paged_kv_last_page_len_buf_arr=[l0_last, l1_last],
                )
            except Exception as exc:
                logger.warning(
                    "cascade CUDA-graph wrapper failed for bs=%s: %s", bs, exc
                )
                continue
            self._cascade_graph_bufs[bs] = {
                "l0_qo": l0_qo,
                "l0_kv_indptr": l0_kv,
                "l0_indices": l0_idx,
                "l0_last": l0_last,
                "l1_qo": l1_qo,
                "l1_kv_indptr": l1_kv,
                "l1_indices": l1_idx,
                "l1_last": l1_last,
            }
            self._cascade_graph_wrappers[bs] = wrapper
            created.append(bs)
        if created:
            logger.info(
                "FlashInfer cascade CUDA-graph wrappers: %s max_reqs=%d",
                sorted(self._cascade_graph_wrappers),
                max_reqs,
            )

    def prepare(self, batch: ForwardBatch) -> None:
        """Plan FlashInfer wrappers once per batch (all layers share the plan)."""
        self._cascade_active = False
        if batch.kv_indices is None:
            return
        if not batch.is_prefill:
            if batch.cascade is not None and self.supports_cascade:
                with trace_range("flashrec.fi_cascade_plan"):
                    if self._plan_cascade(batch):
                        self._cascade_active = True
                        return
            if self._active_decode is not None:
                with trace_range("flashrec.fi_plan"):
                    self._plan_decode(batch)
        elif self._fi_prefill is not None:
            self._plan_prefill(batch)

    @torch.compiler.disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: ForwardBatch,
        skip_store: bool = False,
    ) -> torch.Tensor:
        """q/k/v: [T, H, D] (k/v kv-heads). Writes new KV then attends."""
        if not skip_store:
            self.kv_pool.store(layer_id, k, v, batch.out_cache_loc)
        if not batch.is_prefill and self._cascade_active:
            wrapper = self._active_cascade or self._fi_cascade
            if wrapper is not None:
                return wrapper.run(q, self._paged_kv(layer_id))
        if (
            not batch.is_prefill
            and self._active_decode is not None
            and batch.kv_indices is not None
        ):
            o = self._active_decode.run(q=q, paged_kv_cache=self._paged_kv(layer_id))
            return o
        if (
            batch.is_prefill
            and self._fi_prefill is not None
            and batch.kv_indices is not None
        ):
            o = self._fi_prefill.run(q=q, paged_kv_cache=self._paged_kv(layer_id))
            return o
        return self._sdpa(q, layer_id, batch)

    def _paged_kv(self, layer_id: int):
        k = self.kv_pool.k_cache(layer_id).view(-1, 1, self.num_kv_heads, self.head_dim)
        v = self.kv_pool.v_cache(layer_id).view(-1, 1, self.num_kv_heads, self.head_dim)
        return (k, v)

    def _plan_kwargs(self, indptr, indices, last_page_len, seq_lens):
        kwargs = dict(
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens=seq_lens,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_pool.dtype,
            non_blocking=True,
        )
        return kwargs

    def _plan_decode(self, batch: ForwardBatch) -> None:
        wrapper = self._active_decode
        assert wrapper is not None
        n = int(batch.n_rows)
        host_indptr, last_cpu, seq = self._pin.grow(n)
        sl_cpu = batch.seq_lens_cpu.view(-1)[:n]
        if sl_cpu.device.type == "cpu" and sl_cpu.dtype == torch.int32:
            seq.copy_(sl_cpu)
        else:
            _seq_lens_cpu32(sl_cpu, n, seq)
        kv = batch.kv_indices.view(-1)
        if kv.dtype != torch.int32:
            kv = kv.to(dtype=torch.int32)
        gbs = self._graph_bs
        bufs = self._graph_bufs.get(int(gbs)) if gbs is not None else None
        if bufs is not None:
            indptr = bufs["indptr"]
            last = bufs["last"]
            indices = bufs["indices"]
            seq_gpu = batch.seq_lens.view(-1)[:n]
            if seq_gpu.dtype != torch.int32:
                seq_gpu = seq_gpu.to(dtype=torch.int32)
            # indptr was torch.zeros(...) and cumsum only writes [1:n+1], so
            # indptr[0] stays 0; writing it here is a scalar pageable H2D that
            # stream-syncs behind the previous replay. Same for last: created
            # as ones and never overwritten, the fill is a dead launch.
            torch.cumsum(seq_gpu, dim=0, out=indptr[1 : n + 1])
            nidx = int(kv.numel())
            if nidx > 0 and kv.data_ptr() != indices.data_ptr():
                indices[:nidx].copy_(kv, non_blocking=True)
            host_indptr[0] = 0
            torch.cumsum(seq, dim=0, out=host_indptr[1:])
            idx_view = indices[: max(nidx, 1)]
            kwargs = self._plan_kwargs(indptr[: n + 1], idx_view, last[:n], seq)
            if self._plan_graph_decode(wrapper, kwargs, host_indptr):
                return
            fb = dict(kwargs)
            fb["indptr"] = host_indptr
            try:
                wrapper.plan(**fb, data_type=self.kv_pool.dtype)
            except TypeError:
                wrapper.plan(**fb)
            return
        host_indptr[0] = 0
        torch.cumsum(seq, dim=0, out=host_indptr[1:])
        last_cpu.fill_(1)
        kwargs = self._plan_kwargs(host_indptr, kv, last_cpu, seq)
        try:
            wrapper.plan(**kwargs, data_type=self.kv_pool.dtype)
        except TypeError:
            wrapper.plan(**kwargs)

    def _plan_graph_decode(
        self, wrapper, kwargs: dict, host_indptr: torch.Tensor
    ) -> bool:
        """Skip FlashInfer's graph-buffer memcpy (SGLang ``fast_decode_plan``).

        The first ``plan()`` on a wrapper must go through FlashInfer's stock
        path so ``_cached_module`` is created; later steps can skip the D2D.
        """
        if getattr(wrapper, "_cached_module", None) is None:
            return False
        try:
            from flashinfer.decode import fast_decode_plan
        except Exception:
            return False
        try:
            fast_decode_plan(
                wrapper,
                kwargs["indptr"],
                kwargs["indices"],
                kwargs["last_page_len"],
                kwargs["num_qo_heads"],
                kwargs["num_kv_heads"],
                kwargs["head_dim"],
                kwargs["page_size"],
                pos_encoding_mode=kwargs["pos_encoding_mode"],
                q_data_type=kwargs["q_data_type"],
                kv_data_type=kwargs["kv_data_type"],
                non_blocking=True,
                global_override_indptr_cpu=host_indptr,
            )
            return True
        except Exception:
            return False

    def _plan_cascade(self, batch: ForwardBatch) -> bool:
        """Plan two-level cascade decode. False = fall back to paged decode.

        Level 0 attends each request's shared prompt pages once for all its
        beam rows; level 1 attends each row's own decode-history pages. Both
        levels read the same paged KV pool, so this is index-table-only.

        Level-0 (shared prompt) is constant for a wave, so its FlashInfer plan
        is reused across decode steps. Level 1 still replans every step.

        When ``begin_graph_cascade_decode`` is active, meta is padded into the
        persistent CUDA-graph buffers (fixed request slots + dummy pad rows)
        and the graph wrapper is planned. A failed graph plan raises so capture
        / replay cannot mix a paged plan with a cascade ``run``.
        """
        meta = batch.cascade
        assert meta is not None
        if self._cascade_graph_bs is not None:
            if self._plan_cascade_graph(batch):
                return True
            raise RuntimeError(
                f"cascade CUDA-graph plan failed for bs={self._cascade_graph_bs}"
            )
        if self._fi_cascade is None:
            try:
                from flashinfer.cascade import MultiLevelCascadeAttentionWrapper

                # The two internal prefill wrappers split this buffer; 128MiB
                # overflows past ~100 rows (batch_prefill_tmp_s scaling).
                self._ensure_cascade_workspace(512 * 1024 * 1024)
                self._fi_cascade = MultiLevelCascadeAttentionWrapper(
                    2, self._workspace_cascade, kv_layout="NHD"
                )
                self._active_cascade = self._fi_cascade
            except Exception as exc:
                logger.warning("cascade attention unavailable: %s", exc)
                self.supports_cascade = False
                return False
        try:
            wrappers = getattr(self._fi_cascade, "_batch_prefill_wrappers", None)
            l0_sig = (
                id(meta.qo_indptr[0]),
                id(meta.kv_indptr[0]),
                id(meta.kv_indices[0]),
                int(meta.kv_indices[0].numel()),
            )
            skip_l0 = (
                wrappers is not None
                and len(wrappers) >= 2
                and self._cascade_l0_sig == l0_sig
                and getattr(wrappers[0], "_cached_module", None) is not None
            )

            def _plan_level(wrapper, level: int) -> None:
                wrapper.plan(
                    meta.qo_indptr[level],
                    meta.kv_indptr[level],
                    meta.kv_indices[level],
                    meta.last_page_len[level],
                    self.num_qo_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    1,
                    causal=False,
                    q_data_type=self.q_dtype,
                    kv_data_type=self.kv_pool.dtype,
                )

            if wrappers is not None and len(wrappers) >= 2:
                if not skip_l0:
                    _plan_level(wrappers[0], 0)
                    self._cascade_l0_sig = l0_sig
                _plan_level(wrappers[1], 1)
                return True
            self._fi_cascade.plan(
                meta.qo_indptr,
                meta.kv_indptr,
                meta.kv_indices,
                meta.last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                page_size=1,
                q_data_type=self.q_dtype,
                kv_data_type=self.kv_pool.dtype,
            )
            return True
        except Exception as exc:
            logger.warning("cascade plan failed, disabling cascade: %s", exc)
            self.supports_cascade = False
            self._cascade_l0_sig = None
            return False

    def _copy_plan_vec(self, dst: torch.Tensor, src: torch.Tensor, n: int) -> None:
        s = src.view(-1)[:n]
        if s.dtype != dst.dtype:
            s = s.to(dtype=dst.dtype)
        dst[:n].copy_(s, non_blocking=s.device == dst.device)

    def _fill_cascade_graph_bufs(
        self,
        meta: CascadeMeta,
        bufs: dict,
        bs: int,
        skip_l0: bool = False,
    ) -> Optional[tuple[int, int]]:
        """Pad ``CascadeMeta`` into persistent cascade-graph buffers.

        Real request groups occupy ``[0, n_reqs)``. Pad rows ``[n_rows, bs)``
        become one extra dummy request that attends page 0 at both levels.
        Remaining L0 slots are empty groups. Returns ``(nidx0, nidx1)`` totals
        including dummy pages, or None if the meta cannot fit.
        """
        max_reqs = int(self._max_cascade_reqs)
        qo0 = meta.qo_indptr[0].view(-1)
        kv0 = meta.kv_indptr[0].view(-1)
        kv1 = meta.kv_indptr[1].view(-1)
        n_reqs = int(qo0.numel()) - 1
        if n_reqs < 1 or qo0.numel() != kv0.numel():
            return None
        n_rows = int(qo0[-1].item())
        pad_rows = int(bs) - n_rows
        if n_rows < 1 or pad_rows < 0:
            return None
        if n_reqs > max_reqs:
            return None
        if pad_rows > 0 and n_reqs >= max_reqs:
            return None
        if int(kv1.numel()) != n_rows + 1:
            return None

        idx0 = meta.kv_indices[0].view(-1)
        idx1 = meta.kv_indices[1].view(-1)
        if idx0.dtype != torch.int32:
            idx0 = idx0.to(dtype=torch.int32)
        if idx1.dtype != torch.int32:
            idx1 = idx1.to(dtype=torch.int32)
        nidx0 = int(idx0.numel())
        nidx1 = int(idx1.numel())
        extra0 = 1 if pad_rows > 0 else 0
        extra1 = pad_rows
        if nidx0 + extra0 > int(bufs["l0_indices"].numel()):
            return None
        if nidx1 + extra1 > int(bufs["l1_indices"].numel()):
            return None

        if not skip_l0:
            self._copy_plan_vec(bufs["l0_qo"], qo0, n_reqs + 1)
            tail_q = bs if pad_rows > 0 else n_rows
            if n_reqs + 1 < int(bufs["l0_qo"].numel()):
                bufs["l0_qo"][n_reqs + 1 :].fill_(tail_q)
            self._copy_plan_vec(bufs["l0_kv_indptr"], kv0, n_reqs + 1)
            tail_kv = nidx0 + extra0
            if n_reqs + 1 < int(bufs["l0_kv_indptr"].numel()):
                bufs["l0_kv_indptr"][n_reqs + 1 :].fill_(tail_kv)
            if nidx0 > 0 and idx0.data_ptr() != bufs["l0_indices"].data_ptr():
                bufs["l0_indices"][:nidx0].copy_(
                    idx0, non_blocking=idx0.device == bufs["l0_indices"].device
                )
            if extra0:
                bufs["l0_indices"][nidx0 : nidx0 + 1].zero_()

        self._copy_plan_vec(bufs["l1_kv_indptr"], kv1, n_rows + 1)
        if pad_rows > 0:
            device = bufs["l1_kv_indptr"].device
            bufs["l1_kv_indptr"][n_rows + 1 : bs + 1].copy_(
                torch.arange(
                    nidx1 + 1,
                    nidx1 + pad_rows + 1,
                    dtype=torch.int32,
                    device=device,
                )
            )
        if nidx1 > 0 and idx1.data_ptr() != bufs["l1_indices"].data_ptr():
            bufs["l1_indices"][:nidx1].copy_(
                idx1, non_blocking=idx1.device == bufs["l1_indices"].device
            )
        if extra1:
            bufs["l1_indices"][nidx1 : nidx1 + extra1].zero_()
        return nidx0 + extra0, nidx1 + extra1

    def _plan_cascade_graph(self, batch: ForwardBatch) -> bool:
        bs = int(self._cascade_graph_bs)
        wrapper = self._cascade_graph_wrappers.get(bs)
        bufs = self._cascade_graph_bufs.get(bs)
        if wrapper is None or bufs is None:
            return False
        meta = batch.cascade
        assert meta is not None
        wrappers = getattr(wrapper, "_batch_prefill_wrappers", None)
        if wrappers is None or len(wrappers) < 2:
            return False
        l0_sig = (
            bs,
            id(meta.qo_indptr[0]),
            id(meta.kv_indptr[0]),
            id(meta.kv_indices[0]),
            int(meta.kv_indices[0].numel()),
        )
        skip_l0 = (
            self._cascade_graph_l0_sig == l0_sig
            and getattr(wrappers[0], "_cached_module", None) is not None
        )
        filled = self._fill_cascade_graph_bufs(meta, bufs, bs, skip_l0=skip_l0)
        if filled is None:
            return False
        nidx0, nidx1 = filled
        try:

            def _plan_level(w, qo, kv_indptr, indices, last) -> None:
                w.plan(
                    qo,
                    kv_indptr,
                    indices,
                    last,
                    self.num_qo_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    1,
                    causal=False,
                    q_data_type=self.q_dtype,
                    kv_data_type=self.kv_pool.dtype,
                )

            if not skip_l0:
                _plan_level(
                    wrappers[0],
                    bufs["l0_qo"],
                    bufs["l0_kv_indptr"],
                    bufs["l0_indices"][: max(nidx0, 1)],
                    bufs["l0_last"],
                )
                self._cascade_graph_l0_sig = l0_sig
            _plan_level(
                wrappers[1],
                bufs["l1_qo"],
                bufs["l1_kv_indptr"],
                bufs["l1_indices"][: max(nidx1, 1)],
                bufs["l1_last"],
            )
            return True
        except Exception as exc:
            logger.warning("cascade CUDA-graph plan failed: %s", exc)
            self._cascade_graph_l0_sig = None
            return False

    def _plan_prefill(self, batch: ForwardBatch) -> None:
        assert self._fi_prefill is not None
        n = batch.n_rows
        ext = batch.extend_seq_lens or [1] * n
        kv_indptr, last, seq = self._pin.grow(n)
        qo, _, _ = self._pin_qo.grow(n)
        _seq_lens_cpu32(batch.seq_lens_cpu, n, seq)
        kv_indptr[0] = 0
        torch.cumsum(seq, dim=0, out=kv_indptr[1:])
        last.fill_(1)
        ext_t = torch.as_tensor(list(ext[:n]), dtype=torch.int32)
        qo[0] = 0
        torch.cumsum(ext_t, dim=0, out=qo[1:])
        self._fi_prefill.plan(
            qo_indptr=qo,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=batch.kv_indices.to(dtype=torch.int32),
            paged_kv_last_page_len=last,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens=seq,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_pool.dtype,
            causal=True,
        )

    def _sdpa(
        self, q: torch.Tensor, layer_id: int, batch: ForwardBatch
    ) -> torch.Tensor:
        """Per-row SDPA using gathered KV. Correct, used when FlashInfer is off."""
        k_all = self.kv_pool.k_cache(layer_id)
        v_all = self.kv_pool.v_cache(layer_id)
        if k_all.dtype != q.dtype:
            k_all = k_all.to(dtype=q.dtype)
            v_all = v_all.to(dtype=q.dtype)
        rows = batch.n_rows
        seq_lens = [int(x) for x in batch.seq_lens_cpu.tolist()]
        prefix = batch.extend_prefix_lens or (
            [int(s) - 1 for s in seq_lens] if not batch.is_prefill else [0] * rows
        )
        ext = batch.extend_seq_lens or [1] * rows
        if batch.kv_indices is None:
            raise RuntimeError("ForwardBatch.kv_indices required for attention")
        outs = []
        offset_q = 0
        offset_kv = 0
        scale = self.head_dim**-0.5
        for i in range(rows):
            sl = seq_lens[i]
            e = int(ext[i])
            idx = batch.kv_indices[offset_kv : offset_kv + sl].to(dtype=torch.int64)
            k = k_all[idx]
            v = v_all[idx]
            qi = q[offset_q : offset_q + e]
            k_t = k.permute(1, 0, 2).unsqueeze(0)
            v_t = v.permute(1, 0, 2).unsqueeze(0)
            q_t = qi.permute(1, 0, 2).unsqueeze(0)
            k_t = _repeat_kv(k_t, self.n_rep)
            v_t = _repeat_kv(v_t, self.n_rep)
            if batch.is_prefill:
                pfx = int(prefix[i])
                if pfx <= 0:
                    attn = F.scaled_dot_product_attention(
                        q_t, k_t, v_t, is_causal=True, scale=scale
                    )
                else:
                    e_idx = torch.arange(e, device=q.device)[:, None]
                    s_idx = torch.arange(sl, device=q.device)[None, :]
                    mask = s_idx <= (pfx + e_idx)
                    attn = F.scaled_dot_product_attention(
                        q_t, k_t, v_t, attn_mask=mask, scale=scale
                    )
            else:
                attn = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
            outs.append(attn.squeeze(0).permute(1, 0, 2))
            offset_q += e
            offset_kv += sl
        return torch.cat(outs, dim=0)
