from typing import List

import torch

from flashrec.kvcache.radix import PrefixCache


class _FakeAllocator:
    def __init__(self):
        self.freed: List[int] = []

    def free(self, idx) -> None:
        if isinstance(idx, torch.Tensor):
            self.freed.extend(int(x) for x in idx.reshape(-1).tolist())
        else:
            self.freed.extend(int(x) for x in idx)


class TestPrefixCache:
    def test_insert_match_shared_prefix(self):
        cache = PrefixCache(device=torch.device("cpu"))
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3, 9, 8]
        kv_a = torch.arange(100, 105)
        canon_a = cache.insert(a, kv_a)
        assert canon_a.tolist() == [100, 101, 102, 103, 104]
        hit, kv = cache.match(b[:-1])
        assert hit == 3
        assert kv.tolist() == [100, 101, 102]
        suffix = torch.tensor([201, 202])
        full_b = torch.cat([kv, suffix])
        canon_b = cache.insert(b, full_b)
        assert canon_b.tolist() == [100, 101, 102, 201, 202]
        dups = full_b[canon_b != full_b]
        assert dups.numel() == 0

    def test_lock_blocks_eviction(self):
        alloc = _FakeAllocator()
        cache = PrefixCache(allocator=alloc, device=torch.device("cpu"))
        tokens = [7, 8, 9]
        cache.insert(tokens, torch.tensor([10, 11, 12]))
        cache.lock(tokens)
        assert cache.evict(10) == 0
        assert alloc.freed == []
        cache.unlock(tokens)
        assert cache.evict(2) == 2
        assert len(alloc.freed) == 2
        hit, _ = cache.match(tokens)
        assert hit == 1

    def test_lru_evicts_older_branch(self):
        alloc = _FakeAllocator()
        cache = PrefixCache(allocator=alloc, device=torch.device("cpu"))
        cache.insert([1, 2], torch.tensor([1, 2]))
        cache.insert([1, 3], torch.tensor([1, 3]))
        cache.match([1, 3])
        freed = cache.evict(1)
        assert freed == 1
        assert 2 in alloc.freed
        hit_old, _ = cache.match([1, 2])
        hit_new, _ = cache.match([1, 3])
        assert hit_old == 1
        assert hit_new == 2

    def test_insert_cpu_lists_share_prefix(self):
        cache = PrefixCache(device=torch.device("cpu"))
        a = list(range(20, 50))
        b = list(range(20, 40)) + [99, 98, 97]
        cache.insert_cpu(a, list(range(1000, 1000 + len(a))))
        hit, kv = cache.match_cpu(b)
        assert hit == 20
        assert kv == list(range(1000, 1020))
        cache.insert_cpu(b, kv + [2001, 2002, 2003])
        hit2, _ = cache.match_cpu(b)
        assert hit2 == len(b)

    def test_match_empty(self):
        cache = PrefixCache(device=torch.device("cpu"))
        hit, kv = cache.match([1, 2, 3])
        assert hit == 0
        assert int(kv.numel()) == 0

    def test_evict_many_leaves_heap(self):
        alloc = _FakeAllocator()
        cache = PrefixCache(allocator=alloc, device=torch.device("cpu"))
        for i in range(200):
            cache.insert([0, i + 1], torch.tensor([0, i + 10]))
        assert cache.num_cached_tokens > 200
        n = cache.evict(50)
        assert n == 50
        assert len(alloc.freed) == 50
        assert (len(cache.evictable_leaves) > 0) == True

    def test_insert_cpu_list_no_tensor_required(self):
        cache = PrefixCache(device=torch.device("cpu"))
        canon = cache.insert_cpu([1, 2, 3], [10, 11, 12])
        assert canon == [10, 11, 12]
        hit, kv = cache.match_cpu([1, 2, 9])
        assert hit == 2
        assert kv == [10, 11]

    def test_prefix_len_does_not_tick(self):
        cache = PrefixCache(device=torch.device("cpu"))
        cache.insert_cpu([1, 2, 3], [10, 11, 12])
        clock = cache._clock
        assert cache.prefix_len([1, 2, 9]) == 2
        assert cache.prefix_len([1, 2, 3]) == 3
        assert cache._clock == clock
        hit, _ = cache.match_cpu([1, 2, 3])
        assert hit == cache.prefix_len([1, 2, 3])

    def test_skip_suffix_keeps_shared_prefix_only(self):
        cache = PrefixCache(device=torch.device("cpu"))
        shared = [1, 2, 3]
        cache.insert_cpu(shared, [10, 11, 12])
        cache.lock(shared)
        user = shared + [100, 101, 102]
        assert cache.prefix_len(user) == 3
        hit, kv = cache.match_cpu(user)
        assert hit == 3
        assert kv == [10, 11, 12]
        other = shared + [200, 201]
        assert cache.prefix_len(other) == 3

    def test_insert_after_matches_full_insert(self):
        full = PrefixCache(device=torch.device("cpu"))
        suf = PrefixCache(device=torch.device("cpu"))
        prefix = [1, 2, 3]
        prefix_kv = [10, 11, 12]
        suffix = [4, 5]
        suffix_kv = [13, 14]
        full.insert_cpu(prefix + suffix, prefix_kv + suffix_kv)
        suf.insert_cpu(prefix, prefix_kv)
        got = suf.insert_after(prefix, suffix, suffix_kv)
        assert got == [13, 14]
        assert suf.match_cpu(prefix + suffix)[1] == full.match_cpu(prefix + suffix)[1]
