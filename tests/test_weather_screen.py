from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_300.weather_screen import permute_within  # noqa: E402


class PermuteTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"a": np.arange(12.0), "b": np.arange(12.0) * 10, "keep": np.arange(12.0),
                                   "g": ["x"] * 6 + ["y"] * 6})

    def test_values_stay_within_their_group(self):
        out = permute_within(self.frame, ["a", "b"], self.frame.g, 1)
        self.assertEqual(sorted(out.a[:6]), list(range(6)))
        self.assertEqual(sorted(out.a[6:]), list(range(6, 12)))

    def test_columns_move_together(self):
        out = permute_within(self.frame, ["a", "b"], self.frame.g, 1)
        np.testing.assert_array_equal(out.b.to_numpy(), out.a.to_numpy() * 10)

    def test_other_columns_untouched_and_order_actually_changes(self):
        out = permute_within(self.frame, ["a", "b"], self.frame.g, 1)
        np.testing.assert_array_equal(out.keep, self.frame.keep)
        self.assertFalse(np.array_equal(out.a, self.frame.a))

    def test_seed_is_reproducible_and_seeds_differ(self):
        a = permute_within(self.frame, ["a"], self.frame.g, 1).a.to_numpy()
        b = permute_within(self.frame, ["a"], self.frame.g, 1).a.to_numpy()
        c = permute_within(self.frame, ["a"], self.frame.g, 2).a.to_numpy()
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))


if __name__ == "__main__":
    unittest.main()
