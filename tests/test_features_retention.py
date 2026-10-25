"""Unit tests for research_720_retention/features_retention.py (synthetic data only)."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_720_retention import features_retention as R  # noqa: E402


def at(minute, day="2025-01-10"):
    return pd.Timestamp(f"{day} 06:00", tz="UTC") + pd.Timedelta(minutes=minute)


def raw(rows, airport="LFPG"):
    """rows: (id, runway, sched_min, aobt_min, mvt_min) -> raw departures."""
    return pd.DataFrame({"MVT_ID_mvt": [r[0] for r in rows], "PHASE_mvt": "DEP", "ADEP_mvt": airport,
                         "RUNWAY_mvt": [r[1] for r in rows], "SCHED_TIME_UTC_mvt": [at(r[2]) for r in rows],
                         "AOBT_3_flt": [at(r[3]) if r[3] is not None else pd.NaT for r in rows],
                         "MVT_TIME_UTC_mvt": [at(r[4]) for r in rows]})


ROWS = [(0, "26L", 0, 10, 25), (1, "26L", 5, 20, 35), (2, "08R", 8, 12, 30), (3, "26L", 30, 40, 55)]


class RetentionTest(unittest.TestCase):
    def setUp(self):
        self.out = R.compute(raw(ROWS))

    def test_hand_computed_row_d1(self):                    # t = minute 20
        r = self.out.loc[1]
        self.assertEqual(r.ret_wait_apt, 0)                  # d0 and d2 have pushed, d3 not yet past schedule
        self.assertEqual(r.ret_roll_apt, 2)                  # d0 (10 <= 20 < 25) and d2 (12 <= 20 < 30)
        self.assertEqual(r.ret_roll_bahn, 1)                 # only d0 is on 26L
        self.assertEqual(r.ret_push15, 2)
        self.assertEqual(r.ret_push60, 2)
        self.assertAlmostEqual(r.ret_delay15_s, (10 + 4) / 2 * 60)

    def test_hand_computed_row_d0(self):                    # t = minute 10
        r = self.out.loc[0]
        self.assertEqual(r.ret_wait_apt, 2)                  # d1 (sched 5, push 20) and d2 (sched 8, push 12) are held
        self.assertEqual(r.ret_wait_bahn, 1)                 # only d1 on 26L
        self.assertEqual(r.ret_roll_apt, 0)
        self.assertEqual(r.ret_push15, 0)
        self.assertTrue(np.isnan(r.ret_delay15_s))

    def test_a_row_never_counts_itself(self):
        solo = R.compute(raw([(0, "26L", 0, 10, 25)]))
        self.assertEqual(list(solo.loc[0, ["ret_wait_apt", "ret_wait_bahn", "ret_roll_apt", "ret_roll_bahn", "ret_push15", "ret_push60"]]), [0] * 6)

    def test_rows_without_aobt_are_dropped_and_do_not_count(self):
        out = R.compute(raw(ROWS + [(9, "26L", 1, None, 20)]))
        self.assertNotIn(9, out.index)
        self.assertEqual(out.loc[0, "ret_wait_apt"], 2)

    def test_airports_and_months_are_separate(self):
        other = raw([(20, "26L", 0, 15, 40)], airport="EGLL")
        both = pd.concat([raw(ROWS), other], ignore_index=True)
        self.assertEqual(R.compute(both).loc[0, "ret_wait_apt"], 2)
        self.assertEqual(R.compute(both).loc[20, "ret_wait_apt"], 0)

    def test_attach_aligns_to_the_given_ids(self):
        out = R.attach([2, 0, 99], raw(ROWS))
        self.assertEqual(len(out), 3)
        self.assertEqual(out.loc[1, "ret_wait_apt"], 2)
        self.assertTrue(out.loc[2].isna().all())
        self.assertEqual(list(out.columns), R.MERKMALE)


if __name__ == "__main__":
    unittest.main()
