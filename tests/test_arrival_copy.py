"""Arrival-copy features: copied flag, inbound match per airport and stand, seven-day shares."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_240 import features_arrival_copy as AC


def raw_arrivals(rows):
    frame = pd.DataFrame(rows, columns=["ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                                        "SCHED_TIME_UTC_mvt", "AIRCRAFT_OPERATOR_flt"])
    frame.insert(0, "PHASE_mvt", "ARR")
    for col in ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def departures(rows):
    frame = pd.DataFrame(rows, columns=["ADEP_mvt", "STAND_mvt", "AIRCRAFT_OPERATOR_flt", "MVT_TIME_UTC_mvt", "ersatz_s"])
    frame["MVT_TIME_UTC_mvt"] = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.as_unit("us")
    return frame


ARRIVALS = raw_arrivals([
    ("LIRF", "S1", "2025-01-10 08:00", "2025-01-10 08:10:00", "2025-01-10 08:10:05", "AZA"),
    ("LIRF", "S1", "2025-01-10 11:00", "2025-01-10 11:12:00", "2025-01-10 11:00:00", "AZA"),
    ("LIRF", "S2", "2025-01-09 07:00", "2025-01-09 07:10:00", "2025-01-09 07:10:00", "AZA"),
    ("EGLL", "S1", "2025-01-10 10:50", "2025-01-10 11:00:00", "2025-01-10 11:00:00", "BAW"),
    ("LIRF", "S9", "2025-01-03 10:00", "2025-01-03 10:10:00", "2025-01-03 10:10:00", "XYZ"),
    ("LIRF", "S9", "2025-01-08 10:00", "2025-01-08 10:20:00", "2025-01-08 10:10:00", "XYZ"),
    ("LIRF", "S9", "2025-01-09 10:00", "2025-01-09 10:10:00", "2025-01-09 10:10:00", "XYZ"),
    ("LIRF", "S9", "2025-01-10 09:00", "2025-01-10 09:10:00", "2025-01-10 09:10:00", "XYZ"),
])


class ArrivalsTable(unittest.TestCase):
    def test_copied_flag_and_departures_excluded(self):
        raw = pd.concat([ARRIVALS, ARRIVALS.head(1).assign(PHASE_mvt="DEP")], ignore_index=True)
        table = AC.arrivals_table(raw)
        self.assertEqual(len(table), len(ARRIVALS))
        self.assertEqual(list(table.copied), [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        self.assertEqual(table.stand_key[0], "LIRF|S1")


class Features(unittest.TestCase):
    def setUp(self):
        self.table = AC.arrivals_table(ARRIVALS)

    def test_inbound_match_window_and_airport_separation(self):
        dep = departures([
            ("LIRF", "S1", "AZA", "2025-01-10 11:20", 900),   # AOBT 11:05: A2 landed 11:00, too late
            ("LIRF", "S1", "AZA", "2025-01-10 12:30", 600),   # AOBT 12:20: last is A2
            ("LIRF", "S3", "AZA", "2025-01-10 12:30", 600),   # no arrivals at S3
            ("LIRF", "S1", "AZA", "2025-01-12 12:00", 600),   # last arrival over 24 h before
            ("EGLL", "S1", "BAW", "2025-01-10 11:30", 600),   # same stand name, other airport
            ("LIRF", None, "AZA", "2025-01-10 12:30", 600),   # missing stand
        ])
        out = AC.features_for(dep, self.table)
        self.assertEqual(out.ac_inbound_copied[0], 1.0)
        self.assertAlmostEqual(out.ac_inbound_gap_min[0], 185.0)
        self.assertEqual(out.ac_inbound_copied[1], 0.0)
        self.assertAlmostEqual(out.ac_inbound_gap_min[1], 80.0)
        self.assertTrue(np.isnan(out.ac_inbound_copied[2]))
        self.assertTrue(np.isnan(out.ac_inbound_copied[3]))
        self.assertEqual(out.ac_inbound_copied[4], 1.0)
        self.assertAlmostEqual(out.ac_inbound_gap_min[4], 30.0)
        self.assertTrue(np.isnan(out.ac_inbound_copied[5]))
        self.assertTrue(np.isnan(out.ac_stand_share_7d[0]))

    def test_seven_day_shares_and_minimum_counts(self):
        dep = departures([("LIRF", "S9", "XYZ", "2025-01-10 10:30", 600)])   # window (Jan 3 10:10, Jan 10 10:10]
        out = AC.features_for(dep, self.table)
        self.assertAlmostEqual(out.ac_stand_share_7d[0], 2 / 3)
        self.assertTrue(np.isnan(out.ac_operator_share_7d[0]))              # three operator arrivals, below five
        self.assertEqual(out.ac_inbound_copied[0], 1.0)

    def test_missing_ersatz_gives_missing_features(self):
        dep = departures([("LIRF", "S1", "AZA", "2025-01-10 12:30", np.nan)])
        out = AC.features_for(dep, self.table)
        self.assertTrue(out.isna().all().all())


if __name__ == "__main__":
    unittest.main()
