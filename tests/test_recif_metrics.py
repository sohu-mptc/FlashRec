"""RecIF ranking metrics must not inflate on duplicate beams."""

from __future__ import annotations

import unittest

from flashrec.benchmark.recif import (
    hit_recall,
    mrr_at_k,
    ndcg_at_k,
    sid_duplicate_rate,
    unique_topk,
)


class RecIFMetricsTest(unittest.TestCase):
    def test_unique_topk_preserves_order(self) -> None:
        self.assertEqual(
            unique_topk(["a", "a", "b", "a", "c", "b"], 3),
            ["a", "b", "c"],
        )

    def test_hit_recall_ignores_duplicate_gt(self) -> None:
        gt = {f"g{i}" for i in range(10)}
        # Historical collapse: one GT SID fills the whole beam.
        beams = ["g0"] * 100
        hit, rec = hit_recall(beams, gt, 32)
        self.assertEqual(hit, 1.0)
        self.assertAlmostEqual(rec, 0.1)  # 1/10, not 32/10=3.2

    def test_ndcg_ignores_duplicate_gt(self) -> None:
        gt = {"g0", "g1"}
        dup = ndcg_at_k(["g0"] * 32, gt, 32)
        once = ndcg_at_k(["g0", "x"] + ["y"] * 30, gt, 32)
        self.assertAlmostEqual(dup, once)

    def test_mrr_uses_first_unique_hit(self) -> None:
        gt = {"g0"}
        self.assertAlmostEqual(mrr_at_k(["x", "x", "g0"], gt, 32), 1.0 / 2)

    def test_sid_duplicate_rate(self) -> None:
        n, u, rate = sid_duplicate_rate(["a", "a", "b", "a"])
        self.assertEqual((n, u), (4, 2))
        self.assertAlmostEqual(rate, 0.5)
        self.assertEqual(sid_duplicate_rate(["a"] * 100)[2], 0.99)
        self.assertEqual(sid_duplicate_rate(list("abcde"))[2], 0.0)
        self.assertEqual(sid_duplicate_rate([])[2], 0.0)


if __name__ == "__main__":
    unittest.main()
