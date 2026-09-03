from flashrec.config import BeamRecConfig
from flashrec.core import BeamResult, BeamSequence
from flashrec.hostpool import HostPool
from flashrec.scheduler.batching import (
    batch_wait_seconds,
    can_admit_job,
    group_by_beam_depth,
    pack_release_count,
    should_prefer_decode,
)
from flashrec.scheduler.loop import (
    InflightLoop,
    can_inflight_prefill,
    take_prefill_batch,
)
from flashrec.scheduler.pipeline import DecodePipeline
from flashrec.scheduler.prefill import group_by_prompt_len
from flashrec.scheduler.warmup import _lcp, _probe_users, strip_trailing_begin
from flashrec.server.api import _dumps_json, _usage_dict, select_jobs_lpm


class _R:
    def __init__(self, n, plen, gen_len=0, expanded=False):
        self.beam_width = n
        self.prompt_len = plen
        self.finished = False
        self.expanded = expanded
        self._gen = gen_len

    def generated_len(self):
        return self._gen


class TestBatchPrefillPipeline:
    def test_prefill_group_by_len(self):
        reqs = [_R(4, 10), _R(4, 12), _R(4, 10)]
        groups = group_by_prompt_len(reqs, lambda r: r.prompt_len)
        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 1

    def test_pipeline_split_conc8_three_pipes(self):
        pipe = DecodePipeline(stages=3, enabled=True)
        reqs = [_R(50, 10) for _ in range(8)]
        pipes = pipe.split(reqs)
        assert [len(p) for p in pipes] == [3, 3, 2]

    def test_pipeline_split_conc8_two_pipes(self):
        pipe = DecodePipeline(stages=2, enabled=True)
        reqs = [_R(50, 10) for _ in range(8)]
        pipes = pipe.split(reqs)
        assert [len(p) for p in pipes] == [4, 4]

        one = DecodePipeline(stages=3, enabled=True)
        assert len(one.split(reqs[:1])) == 1
        assert len(one.split(reqs[:1])[0]) == 1

    def test_materialize_two_pads_empty_third_pipe(self):
        pipe = DecodePipeline(stages=3, enabled=True)
        reqs = [_R(50, 10) for _ in range(2)]
        assert pipe.try_materialize(reqs, lambda r: r.finished)
        assert len(pipe.pipes) == 3
        assert sorted(len(p) for p in pipe.pipes) == [0, 1, 1]
        extra = [_R(50, 10) for _ in range(6)]
        pipe.merge_prefill(extra)
        assert sorted(len(p) for p in pipe.pipes) == [2, 3, 3]

    def test_pipeline_materialize_needs_two(self):
        pipe = DecodePipeline(stages=3, enabled=True, idle_collapse_ms=8.0)
        is_fin = lambda r: r.finished
        assert not pipe.try_materialize([_R(50, 10)], is_fin)
        assert not pipe.active
        reqs = [_R(50, 10) for _ in range(8)]
        assert pipe.try_materialize(reqs, is_fin)
        assert [len(p) for p in pipe.pipes] == [3, 3, 2]

    def test_pipeline_overlap_launches_before_process(self):
        pipe = DecodePipeline(stages=3, enabled=True)
        reqs = [_R(50, 10) for _ in range(8)]
        is_fin = lambda r: r.finished
        pipe.try_materialize(reqs, is_fin)
        log = []

        def process(launch):
            log.append(("process", len(launch)))
            for r in launch:
                r._gen += 1
                if r._gen >= 3:
                    r.finished = True

        def launch(group):
            log.append(("launch", len(group)))
            return group

        for _ in range(3):
            batch = pipe.get_next(process, is_fin)
            assert batch is not None
            pipe.enqueue(pipe.current_pipe_id, launch(batch))
        assert [x[0] for x in log] == ["launch", "launch", "launch"]
        assert [x[1] for x in log] == [3, 3, 2]

    def test_pipeline_run_drains_without_wait_all_step(self):
        pipe = DecodePipeline(stages=3, enabled=True)
        reqs = [_R(50, 10) for _ in range(8)]

        def launch(group):
            return group

        def process(group):
            for r in group:
                r._gen += 1
                if r._gen >= 3:
                    r.finished = True

        pipe.run(reqs, lambda r: r.finished, launch, process)
        assert all(r.finished for r in reqs)
        assert [r.generated_len() for r in reqs] == [3] * 8

    def test_pipeline_sticky_collapse(self):
        pipe = DecodePipeline(stages=3, enabled=True, idle_collapse_ms=8.0)
        reqs = [_R(50, 10) for _ in range(2)]
        is_fin = lambda r: r.finished
        pipe.try_materialize(reqs, is_fin)
        for r in reqs:
            r.finished = True
        pipe.try_dematerialize(is_fin, now=1.0)
        assert pipe.active
        pipe.try_dematerialize(is_fin, now=1.001)
        assert pipe.active
        pipe.try_dematerialize(is_fin, now=1.010)
        assert not pipe.active

    def test_merge_prefill_skips_inflight(self):
        pipe = DecodePipeline(stages=2, enabled=True)
        reqs = [_R(50, 10) for _ in range(2)]
        is_fin = lambda r: r.finished
        pipe.try_materialize(reqs, is_fin)
        pipe.enqueue(0, reqs[0])
        extra = [_R(50, 10)]
        pipe.merge_prefill(extra)
        assert len(pipe.pipes[0]) == 1
        assert len(pipe.pipes[1]) == 2

    def test_merge_prefill_spreads_eight_across_pipes(self):
        pipe = DecodePipeline(stages=3, enabled=True)
        first = [_R(50, 10) for _ in range(8)]
        is_fin = lambda r: r.finished
        pipe.try_materialize(first, is_fin)
        for r in first:
            r.finished = True
        nxt = [_R(50, 10) for _ in range(8)]
        pipe.sync_pipes_from_running(nxt, is_fin)
        assert sorted(len(p) for p in pipe.pipes) == [2, 3, 3]
        assert all(len(p) <= 3 for p in pipe.pipes)

    def test_held_prefill_not_merged_until_ready(self):
        class _Ev:
            def __init__(self, done):
                self.done = done

            def query(self):
                return self.done

        pipe = DecodePipeline(stages=3, enabled=True)
        first = [_R(50, 10) for _ in range(2)]
        is_fin = lambda r: r.finished
        pipe.try_materialize(first, is_fin)
        nxt = _R(50, 10)
        nxt.prefill_done = _Ev(False)
        pipe.sync_pipes_from_running(first + [nxt], is_fin)
        assert sum(len(p) for p in pipe.pipes) == 2
        assert len(pipe.held_prefills) == 1
        nxt.prefill_done.done = True
        pipe.sync_pipes_from_running(first + [nxt], is_fin)
        assert sum(len(p) for p in pipe.pipes) == 3
        assert pipe.held_prefills == []

    def test_get_next_all_inflight_relaunches_not_none(self):
        pipe = DecodePipeline(stages=3, enabled=True)
        reqs = [_R(50, 10) for _ in range(8)]
        is_fin = lambda r: r.finished
        pipe.try_materialize(reqs, is_fin)
        sizes = []

        def process(launch):
            for r in launch:
                r._gen += 1
                if r._gen >= 3:
                    r.finished = True

        for _ in range(12):
            batch = pipe.get_next(process, is_fin)
            if batch is None:
                break
            sizes.append(len(batch))
            pipe.enqueue(pipe.current_pipe_id, batch)
        assert len(sizes) >= 6
        assert all(s <= 3 for s in sizes)
        assert 8 not in sizes
        assert 3 in sizes
        assert 2 in sizes

    def test_take_prefill_inflight_topup_three(self):
        waiting = [_R(50, 10) for _ in range(3)]
        batch = take_prefill_batch(
            waiting,
            running_empty=False,
            used_slots=400,
            budget=800,
            preferred=[8, 16],
            inflight_min=8,
        )
        assert len(batch) == 3
        assert waiting == []

    def test_take_prefill_inflight_skips_three_without_topup(self):
        waiting = [_R(50, 10) for _ in range(3)]
        batch = take_prefill_batch(
            waiting,
            running_empty=False,
            used_slots=400,
            budget=800,
            preferred=[8, 16],
            inflight_min=8,
            allow_topup=False,
        )
        assert batch == []
        assert len(waiting) == 3


class TestInflightLoop:
    def test_inflight_requires_eight(self):
        waiting = [_R(50, 10) for _ in range(3)]
        assert not can_inflight_prefill(waiting, used_slots=400, budget=800, min_reqs=8)
        waiting8 = [_R(50, 10) for _ in range(8)]
        assert can_inflight_prefill(waiting8, used_slots=400, budget=800, min_reqs=8)
        assert not can_inflight_prefill(
            waiting8, used_slots=800, budget=800, min_reqs=8
        )
        waiting6 = [_R(50, 10) for _ in range(6)]
        assert can_inflight_prefill(waiting6, used_slots=400, budget=800, min_reqs=8)
        waiting5 = [_R(50, 10) for _ in range(5)]
        assert not can_inflight_prefill(
            waiting5, used_slots=400, budget=800, min_reqs=8
        )
        waiting1 = [_R(50, 10)]
        assert not can_inflight_prefill(
            waiting1, used_slots=750, budget=800, min_reqs=8
        )

    def test_take_prefill_idle_small_batch(self):
        waiting = [_R(50, 10) for _ in range(3)]
        batch = take_prefill_batch(
            waiting,
            running_empty=True,
            used_slots=0,
            budget=800,
            preferred=[8, 16],
            inflight_min=8,
        )
        assert len(batch) == 3
        assert waiting == []

    def test_take_prefill_near_pack_six(self):
        waiting = [_R(50, 10) for _ in range(6)]
        batch = take_prefill_batch(
            waiting,
            running_empty=False,
            used_slots=400,
            budget=800,
            preferred=[8, 16],
            inflight_min=8,
        )
        assert len(batch) == 6
        assert waiting == []

    def test_loop_topup_three_while_decode(self):
        prefills = []
        completed = []

        def prefill(batch):
            prefills.append(len(batch))
            for r in batch:
                r.expanded = True

        def decode(group):
            for r in group:
                r._gen += 1
                if r._gen >= 3:
                    r.finished = True

        def complete(done):
            completed.extend(done)

        loop = InflightLoop(
            slots=800,
            preferred=[8, 16],
            inflight_min=8,
            short_genrec=True,
            prefill_fn=prefill,
            decode_step_fn=decode,
            complete_fn=complete,
            generated_len_of=lambda r: r._gen,
        )
        for _ in range(8):
            loop.submit(_R(50, 10))
        loop.step()
        assert prefills == [8]
        assert len(loop.running) == 8
        for _ in range(3):
            loop.submit(_R(50, 10))
        loop.step()
        # short_genrec locks mid-wave top-up after expand.
        assert prefills == [8]
        assert len(loop.waiting) == 3
        assert len(loop.running) == 8
        while loop.has_work():
            loop.step()
        assert len(completed) == 11
        assert 3 in prefills

    def test_eight_finish_callbacks_without_next_wave(self):
        completed = []

        def prefill(batch):
            for r in batch:
                r.expanded = True

        def decode(group):
            for r in group:
                r._gen += 1
                if r._gen >= 1:
                    r.finished = True

        def complete(done):
            completed.append(len(done))

        loop = InflightLoop(
            slots=800,
            preferred=[8, 16],
            inflight_min=8,
            short_genrec=True,
            prefill_fn=prefill,
            decode_step_fn=decode,
            complete_fn=complete,
            generated_len_of=lambda r: r._gen,
        )
        for _ in range(8):
            loop.submit(_R(50, 10))
        loop.step()
        assert completed == [8]
        assert not loop.has_work()

    def test_mixed_depth_one_decode_call(self):
        decode_sizes = []

        def prefill(batch):
            for r in batch:
                r.expanded = True

        def decode(reqs):
            decode_sizes.append(len(reqs))
            for r in reqs:
                r._gen += 1
                if r._gen >= 3:
                    r.finished = True

        loop = InflightLoop(
            slots=800,
            preferred=[8, 16],
            inflight_min=8,
            short_genrec=True,
            prefill_fn=prefill,
            decode_step_fn=decode,
            complete_fn=lambda done: None,
            generated_len_of=lambda r: r._gen,
        )
        first = [_R(50, 10) for _ in range(8)]
        for r in first:
            loop.submit(r)
        loop.step()
        assert decode_sizes == [8]
        for r in first:
            r._gen = 2
        for _ in range(8):
            loop.submit(_R(50, 10))
        # short_genrec: no mid-wave merge; first eight finish, then next eight.
        loop.step()
        assert decode_sizes[-1] == 8
        assert all(r.finished for r in first)
        assert len(loop.waiting) == 8

    def test_burst_skips_three_then_finishes(self):
        prefills = []

        def prefill(batch):
            prefills.append(len(batch))
            for r in batch:
                r.expanded = True

        def decode(reqs):
            for r in reqs:
                r._gen += 1
                if r._gen >= 3:
                    r.finished = True

        loop = InflightLoop(
            slots=800,
            preferred=[8, 16],
            inflight_min=8,
            short_genrec=True,
            prefill_fn=prefill,
            decode_step_fn=decode,
            complete_fn=lambda done: None,
            generated_len_of=lambda r: r._gen,
        )
        for _ in range(8):
            loop.submit(_R(50, 10))
        loop.step()
        assert prefills == [8]
        assert len(loop.running) == 8
        late = [_R(50, 10) for _ in range(3)]
        for r in late:
            loop.submit(r)
        loop.run_burst()
        assert prefills == [8, 3]
        assert not loop.has_work()
        assert all(r.finished for r in late)

    def test_burst_interrupts_for_full_eight(self):
        prefills = []
        decode_sizes = []

        def prefill(batch):
            prefills.append(len(batch))
            for r in batch:
                r.expanded = True

        def decode(reqs):
            decode_sizes.append(len(reqs))
            for r in reqs:
                r._gen += 1
                if r._gen >= 3:
                    r.finished = True

        loop = InflightLoop(
            slots=800,
            preferred=[8, 16],
            inflight_min=8,
            short_genrec=True,
            prefill_fn=prefill,
            decode_step_fn=decode,
            complete_fn=lambda done: None,
            generated_len_of=lambda r: r._gen,
        )
        for _ in range(8):
            loop.submit(_R(50, 10))

        def peek():
            if len(decode_sizes) == 1 and not loop.waiting:
                for _ in range(8):
                    loop.submit(_R(50, 10))

        loop.run_burst(peek=peek)
        # short_genrec locks top-up until a wave finishes, so peek's 8 wait.
        assert prefills[0] == 8
        assert all(s <= 8 for s in decode_sizes)
        assert not loop.has_work()
        assert sum(prefills) >= 16

    def test_burst_near_pack_six_midwave(self):
        prefills = []

        def prefill(batch):
            prefills.append(len(batch))
            for r in batch:
                r.expanded = True

        def decode(reqs):
            for r in reqs:
                r._gen += 1
                if r._gen >= 3:
                    r.finished = True

        loop = InflightLoop(
            slots=800,
            preferred=[8, 16],
            inflight_min=8,
            short_genrec=True,
            prefill_fn=prefill,
            decode_step_fn=decode,
            complete_fn=lambda done: None,
            generated_len_of=lambda r: r._gen,
        )
        for _ in range(8):
            loop.submit(_R(50, 10))
        loop.step()
        assert prefills == [8]
        for _ in range(6):
            loop.submit(_R(50, 10))
        loop.step()
        # short_genrec: six wait until the first eight drain.
        assert prefills == [8]
        assert len(loop.waiting) == 6
        while loop.has_work():
            loop.step()
        assert 6 in prefills


class TestGenRecAdmit:
    def test_conc1_does_not_use_wait_max(self):
        wait = batch_wait_seconds(
            n_jobs=1,
            slots=50,
            budget=800,
            wait_s=0.004,
            wait_max_s=0.010,
            recent_batch=1,
        )
        assert wait == 0.001

    def test_high_qps_single_job_uses_wait_max(self):
        wait = batch_wait_seconds(
            n_jobs=1,
            slots=50,
            budget=800,
            wait_s=0.004,
            wait_max_s=0.010,
            recent_batch=8,
        )
        assert wait == 0.010

    def test_pack_high_qps_keeps_eight_leaves_straggler(self):
        assert pack_release_count(9, recent_batch=8) == 8
        assert pack_release_count(12, recent_batch=8) == 12
        assert pack_release_count(15, recent_batch=8) == 15
        assert pack_release_count(16, recent_batch=8) == 16
        assert pack_release_count(8, recent_batch=8) == 8
        assert pack_release_count(1, recent_batch=1) == 1

    def test_underfill_uses_wait_max(self):
        wait = batch_wait_seconds(
            n_jobs=2,
            slots=100,
            budget=800,
            wait_s=0.004,
            wait_max_s=0.010,
            recent_batch=2,
        )
        assert wait == 0.010

    def test_half_full_keeps_base_wait(self):
        wait = batch_wait_seconds(
            n_jobs=8,
            slots=400,
            budget=800,
            wait_s=0.004,
            wait_max_s=0.010,
            recent_batch=8,
        )
        assert wait == 0.004

    def test_can_admit_soft_and_slots(self):
        assert can_admit_job(400, 8, 50, 800, 16)
        assert not can_admit_job(800, 8, 50, 800, 16)
        assert not can_admit_job(400, 16, 50, 800, 16)

    def test_group_by_depth_does_not_merge(self):
        reqs = [_R(50, 10, 1), _R(50, 10, 1), _R(50, 10, 0)]
        groups = group_by_beam_depth(
            reqs, lambda r: r.beam_width, lambda r: r.generated_len()
        )
        assert len(groups) == 2
        sizes = sorted(len(g) for g in groups)
        assert sizes == [1, 2]

    def test_fill_before_expand_pulls_straggler(self):
        pending = [_R(50, 10, 0)]
        initial = [_R(50, 10, 0), _R(50, 10, 0)]

        def pull_more(used_slots, used_reqs, wait_s=0.0):
            if not pending:
                return []
            if used_slots + pending[0].beam_width > 800:
                return []
            return [pending.pop(0)]

        wave = list(initial)
        extra = pull_more(sum(r.beam_width for r in wave), len(wave), wait_s=0.01)
        wave.extend(extra)
        assert len(wave) == 3
        assert len(pending) == 0
        groups = group_by_beam_depth(
            wave, lambda r: r.beam_width, lambda r: r.generated_len()
        )
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_mid_wave_keeps_depths_split(self):
        running = [_R(50, 10, 2), _R(50, 10, 2)]
        late = [_R(50, 10, 0)]
        running.extend(late)
        groups = group_by_beam_depth(
            running, lambda r: r.beam_width, lambda r: r.generated_len()
        )
        assert len(groups) == 2

    def test_graph_bs_covers_genrec_n50(self):
        cfg = BeamRecConfig(model_path="x")
        for bs in (50, 100, 150, 200, 250, 300, 350, 400, 800):
            assert bs in cfg.cuda_graph_capture_sizes
        assert cfg.preferred_batch_sizes() == [8, 16]
        assert cfg.target_admit_reqs() == 8
        assert cfg.soft_admit_max_reqs() == 16
        assert cfg.resolved_batch_slots() == cfg.cuda_graph_max_bs
        assert cfg.pipeline_stages == 0
        assert not cfg.enable_pipeline
        assert cfg.enable_fused_silu_fp8
        assert cfg.enable_fused_qk_rope_kv

    def test_shared_prefix_lcp(self):
        a = [1, 2, 3, 10, 11]
        b = [1, 2, 3, 20, 21]
        assert _lcp(a, b) == [1, 2, 3]

    def test_strip_trailing_begin(self):
        assert strip_trailing_begin([1, 2, 99], 99) == [1, 2]
        assert strip_trailing_begin([1, 2, 3], 99) == [1, 2, 3]
        assert strip_trailing_begin([], 99) == []

    def test_warmup_probes_default_and_custom(self):
        class _E:
            config = BeamRecConfig(model_path=".")

        a, b = _probe_users(_E())
        assert a != b
        assert a.startswith("warmup-probe")

        class _Custom:
            config = BeamRecConfig(
                model_path=".", warmup_user_a="head-aaa", warmup_user_b="head-bbb"
            )

        assert _probe_users(_Custom()) == ("head-aaa", "head-bbb")

    def test_usage_dict_cached_tokens(self):
        result = BeamResult(
            text="x",
            output_ids=[1],
            sequences=[BeamSequence(tokens=[1], text="x")],
            prompt_tokens=100,
            completion_tokens=20,
            cached_tokens=8,
        )
        usage = _usage_dict(result)
        assert usage["prompt_tokens_details"]["cached_tokens"] == 8
        assert usage["total_tokens"] == 120

    def test_prefer_decode_full_slots_blocks_waiting(self):
        running = [_R(50, 10, 1, expanded=True)]
        waiting = [_R(50, 10, 0)]
        assert should_prefer_decode(running, waiting, used_slots=800, budget=800)

    def test_prefer_decode_underfill_allows_prefill(self):
        running = [_R(50, 10, 1, expanded=True)]
        waiting = [_R(50, 10, 0)]
        assert not should_prefer_decode(running, waiting, used_slots=50, budget=800)

    def test_prefer_decode_short_genrec_locks_after_expand(self):
        running = [_R(50, 10, 1, expanded=True)]
        waiting = [_R(50, 10, 0)]
        assert should_prefer_decode(
            running, waiting, used_slots=50, budget=800, short_genrec=True
        )

    def test_prefer_decode_empty_waiting(self):
        running = [_R(50, 10, 1, expanded=True)]
        assert should_prefer_decode(running, [], used_slots=50, budget=800)

    def test_prefer_decode_not_expanded_prefills(self):
        running = [_R(50, 10, 0, expanded=False)]
        waiting = [_R(50, 10, 0)]
        assert not should_prefer_decode(running, waiting, used_slots=50, budget=800)

    def test_lpm_admits_longest_prefix_first(self):
        class J:
            def __init__(self, n, plen):
                self.n = n
                self.plen = plen

        queue = [J(50, 1), J(50, 40), J(50, 10)]
        taken = select_jobs_lpm(queue, 0, 0, 100, 16, lambda j: j.plen)
        assert [j.plen for j in taken] == [40, 10]
        assert [j.plen for j in queue] == [1]

    def test_openai_payload_is_json_dumpsable(self):
        payload = {
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "length",
                    "sglext": {"sequence_score": float(-1.5)},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        raw = _dumps_json(payload)
        assert isinstance(raw, (bytes, bytearray))
        assert b"sequence_score" in raw


class TestHostPool:
    def test_disabled_runs_inline(self):
        seen = []
        pool = HostPool(0)
        assert not pool.enabled
        pool.submit(seen.append, 1)
        assert seen == [1]

    def test_threads_offload_and_flush(self):
        seen = []
        pool = HostPool(2)
        assert pool.enabled
        pool.submit(seen.append, 7)
        pool.flush()
        assert seen == [7]
        pool.shutdown()
