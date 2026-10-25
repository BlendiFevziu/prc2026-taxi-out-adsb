from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_300 import incremental as I  # noqa: E402

NAN = np.nan


class IncrementalTest(unittest.TestCase):
    def test_policy_falls_back_to_the_comparator(self):
        np.testing.assert_array_equal(I.policy([100, 200], [NAN, 150]), [100, 150])

    def test_identical_arms_give_zero(self):
        g = I.gain_rows([500, 600], [450, 650], [NAN, 600], [NAN, 600])
        np.testing.assert_array_equal(g, [0, 0])

    def test_extra_good_estimate_is_a_gain(self):
        # arm B adds an estimate 10 s off where the comparator was 100 s off
        g = I.gain_rows([500], [600], [NAN], [510])
        self.assertEqual(g[0], 100 ** 2 - 10 ** 2)

    def test_extra_bad_estimate_is_a_loss(self):
        g = I.gain_rows([500], [510], [NAN], [2000])
        self.assertLess(g[0], 0)

    def test_without_top_ignores_unchanged_rows_and_drops_the_best(self):
        g = [0, 0, 0] + [1.0] * 19 + [1000.0]
        self.assertEqual(I.without_top(g, 0.05), 1.0)

    def test_block_bootstrap_brackets_a_constant_gain(self):
        lo, hi = I.block_bootstrap([5.0] * 40, ["a"] * 10 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10)
        self.assertAlmostEqual(lo, 5.0)
        self.assertAlmostEqual(hi, 5.0)

    def test_block_bootstrap_is_wide_when_one_block_carries_everything(self):
        lo, hi = I.block_bootstrap([0.0] * 30 + [40.0] * 10, ["a"] * 10 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 10.0)


if __name__ == "__main__":
    unittest.main()
