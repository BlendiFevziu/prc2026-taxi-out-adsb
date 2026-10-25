"""Checks for research_240/features_auslast_rate.py. No data files, no model fits."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_240 import features_auslast_rate as AR  # noqa: E402


def stamp(text):
    return pd.Timestamp(text, tz="UTC")


FULL_2025 = AR.month_spans([(2025, m) for m in range(1, 13)])
JAN_JUL_2026 = AR.month_spans([(2026, 1), (2026, 7)])


class AuslastRateTest(unittest.TestCase):
    def test_full_window_is_41_67_days(self):
        self.assertAlmostEqual(AR.FULL_WINDOW_DAYS, 41 + 2/3, places=9)

    def test_interior_row_in_full_year_table_has_full_coverage(self):
        days = AR.available_days(pd.Series([stamp("2025-07-15 12:00")]), FULL_2025)
        self.assertAlmostEqual(days[0], AR.FULL_WINDOW_DAYS, places=9)

    def test_january_row_in_two_month_table_is_truncated(self):
        # window 2025-12-20 16:00 .. 2026-01-31 08:00, data only from 2026-01-01 00:00
        days = AR.available_days(pd.Series([stamp("2026-01-10 12:00")]), JAN_JUL_2026)
        self.assertAlmostEqual(days[0], 30 + 8/24, places=9)

    def test_july_row_in_two_month_table_sees_only_july(self):
        days = AR.available_days(pd.Series([stamp("2026-07-16 12:00")]), JAN_JUL_2026)
        self.assertAlmostEqual(days[0], 31.0, places=9)

    def test_rate_is_scale_free(self):
        t = pd.Series([stamp("2025-07-16 12:00")])
        two_month = AR.month_spans([(2025, 7), (2025, 1)])
        same_volume_in_two_months = 1000.0 * 31.0 / AR.FULL_WINDOW_DAYS
        self.assertAlmostEqual(AR.rate([1000.0], t, FULL_2025)[0], 1000.0, places=6)
        self.assertAlmostEqual(AR.rate([same_volume_in_two_months], t, two_month)[0], 1000.0, places=6)

    def test_adjacent_months_merge_and_order_does_not_matter(self):
        self.assertEqual(AR.month_spans([(2025, 2), (2025, 1)]), AR.month_spans([(2025, 1), (2025, 2)]))
        self.assertEqual(len(AR.month_spans([(2025, 1), (2025, 2)])), 1)
        self.assertEqual(len(JAN_JUL_2026), 2)

    def test_microsecond_storage_gives_identical_values(self):
        s = pd.Series([stamp("2026-01-10 12:00"), stamp("2026-07-03 05:17")])
        np.testing.assert_array_equal(AR.available_days(s.dt.as_unit("us"), JAN_JUL_2026),
                                      AR.available_days(s.dt.as_unit("ns"), JAN_JUL_2026))

    def test_missing_time_gives_nan(self):
        days = AR.available_days(pd.Series([stamp("2026-01-10"), pd.NaT]), JAN_JUL_2026)
        self.assertFalse(np.isnan(days[0]))
        self.assertTrue(np.isnan(days[1]))
        self.assertTrue(np.isnan(AR.rate([10.0, 10.0], pd.Series([stamp("2026-01-10"), pd.NaT]), JAN_JUL_2026)[1]))

    def test_boundary_strays_are_ignored_and_partial_month_raises(self):
        march = pd.Series(pd.date_range("2025-03-01", "2025-03-31 23:00", freq="h", tz="UTC"))
        stray = pd.Series([stamp("2025-02-28 23:59:57")])
        self.assertEqual(AR.spans_from_times(pd.concat([march, stray], ignore_index=True)),
                         AR.month_spans([(2025, 3)]))
        partial = pd.Series(pd.date_range("2025-03-01", "2025-03-10", freq="h", tz="UTC"))
        with self.assertRaises(ValueError):
            AR.spans_from_times(partial)


if __name__ == "__main__":
    unittest.main()
