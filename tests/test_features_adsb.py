"""Unit tests for research_730_adsb_learner/features_adsb.py (synthetic data only)."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_730_adsb_learner import features_adsb as F  # noqa: E402

T0 = pd.Timestamp("2025-01-10 10:00:00", tz="UTC")
TAKEOFF = F.seconds([T0])[0]


def tracks(points, hexa="aaaaaa", callsign="AFR1", airport="LFPG"):
    """points: (seconds before take-off, gs kt, on_ground)."""
    return pd.DataFrame({"hex": hexa, "callsign": callsign, "ts": [TAKEOFF - s for s, _, _ in points],
                         "lat": 49.0 + 0.0001 * np.arange(len(points)), "lon": 2.5, "on_ground": [g for _, _, g in points],
                         "gs": [v for _, v, _ in points], "airport": airport})


def dep(callsign="AFR1", airport="LFPG"):
    return pd.DataFrame({"MVT_ID_mvt": [1.0], "ADEP_mvt": airport, "CALLSIGN_flt": callsign, "MVT_TIME_UTC_mvt": [T0]})


class AdsbFeatureTest(unittest.TestCase):
    def test_hand_computed_push_and_taxi(self):
        pts = [(900, 0.5, True), (840, 0.5, True), (780, 0.5, True), (600, 8.0, True), (300, 12.0, True), (60, 30.0, False)]
        r = F.compute(dep(), tracks(pts)).loc[1.0]
        self.assertEqual(r.ad_matched, 1)
        self.assertEqual(r.ad_pts, 6)
        self.assertAlmostEqual(r.ad_first_rel_s, 900)
        self.assertAlmostEqual(r.ad_span_s, 840)
        self.assertAlmostEqual(r.ad_last_stop_rel_s, 780)        # last point below 2 kt
        self.assertAlmostEqual(r.ad_move_rel_s, 600)             # first moving point after it
        self.assertEqual(r.ad_stops, 1)                          # 900 to 780 s is one stop of 120 s
        self.assertEqual(r.ad_airborne_first, 0)
        self.assertAlmostEqual(r.ad_gs_max, 30.0)

    def test_unmatched_row_has_matched_zero_and_nans(self):
        r = F.compute(dep("OTHER"), tracks([(600, 5.0, True)])).loc[1.0]
        self.assertEqual(r.ad_matched, 0)
        self.assertTrue(np.isnan(r.ad_pts))

    def test_other_airport_does_not_match(self):
        r = F.compute(dep(airport="EGLL"), tracks([(600, 5.0, True)])).loc[1.0]
        self.assertEqual(r.ad_matched, 0)

    def test_points_outside_the_window_are_ignored(self):
        r = F.compute(dep(), tracks([(6000, 5.0, True)])).loc[1.0]   # 100 min before take-off
        self.assertEqual(r.ad_matched, 0)

    def test_gap_above_20_min_uses_the_last_segment(self):
        pts = [(5000, 0.5, True), (4900, 0.5, True), (1000, 0.5, True), (900, 6.0, True)]
        r = F.compute(dep(), tracks(pts)).loc[1.0]
        self.assertEqual(r.ad_pts, 2)                            # only the segment after the 3900 s gap
        self.assertAlmostEqual(r.ad_first_rel_s, 1000)

    def test_two_aircraft_on_one_callsign_keeps_the_one_with_more_points(self):
        a = tracks([(600, 5.0, True), (500, 5.0, True), (400, 5.0, True)], hexa="aaaaaa")
        b = tracks([(300, 5.0, True)], hexa="bbbbbb")
        r = F.compute(dep(), pd.concat([a, b], ignore_index=True)).loc[1.0]
        self.assertEqual(r.ad_pts, 3)

    def test_distance_counts_only_ground_pairs(self):
        r = F.compute(dep(), tracks([(300, 5.0, True), (200, 5.0, True), (100, 40.0, False)])).loc[1.0]
        self.assertAlmostEqual(r.ad_dist_m, 11.1, delta=0.5)     # one 0.0001 degree step on the ground only


class StandReadoutTest(unittest.TestCase):
    def setUp(self):
        self.SD = F._stand_module()
        key = self.SD.FG.stand_schluessel("A1")
        self.table = {key: [np.array([[49.0, 2.5]])]}               # one stand vertex
        self.base = dict(takeoff=0.0)

    def seg(self, rows):
        """rows: (seconds before take-off, lat offset in degrees, gs kt)."""
        return (np.array([-r[0] for r in rows], float), np.array([49.0 + r[1] for r in rows]),
                np.full(len(rows), 2.5), np.array([r[2] for r in rows], float))

    def test_parked_then_pushed_gives_the_bracket_midpoint(self):
        ts, lat, lon, gs = self.seg([(1500, 0, 0.5), (1440, 0, 0.5), (1380, 0, 0.5), (1320, 0, 0.5), (1260, 0.0003, 6), (1200, 0.002, 8)])
        r = F.stand_features(self.SD, self.table, "A1", ts, lat, lon, gs, 0.0)
        self.assertEqual(r["sd_status"], 1)
        self.assertAlmostEqual(r["sd_onset_rel_s"], 1290)            # midpoint of the 1320 to 1260 bracket
        self.assertAlmostEqual(r["sd_gap_s"], 60)
        self.assertLess(r["sd_first_dist_m"], 1)

    def test_unmapped_stand_gets_code_8(self):
        ts, lat, lon, gs = self.seg([(600, 0, 0.5), (540, 0.002, 6)])
        r = F.stand_features(self.SD, self.table, "ZZ9", ts, lat, lon, gs, 0.0)
        self.assertEqual(r["sd_status"], 8)
        self.assertTrue(np.isnan(r["sd_onset_rel_s"]))

    def test_missing_stand_value_gets_code_8(self):
        ts, lat, lon, gs = self.seg([(600, 0, 0.5), (540, 0.002, 6)])
        for missing in (np.nan, None):
            self.assertEqual(F.stand_features(self.SD, self.table, missing, ts, lat, lon, gs, 0.0)["sd_status"], 8)

    def test_never_at_the_stand_gets_code_6(self):
        ts, lat, lon, gs = self.seg([(600, 0.01, 6), (540, 0.012, 8)])   # about 1 km away the whole time
        r = F.stand_features(self.SD, self.table, "A1", ts, lat, lon, gs, 0.0)
        self.assertEqual(r["sd_status"], 6)

    def test_compute_with_stands_adds_columns_and_zero_for_unmatched(self):
        d = dep()
        d["STAND_mvt"] = "A1"
        out = F.compute(d, tracks([(300, 5.0, True)], callsign="OTHER"), stands=True)
        self.assertEqual(list(out.columns), F.MERKMALE + F.STAND_MERKMALE)
        self.assertEqual(out.loc[1.0, "sd_status"], 0)


if __name__ == "__main__":
    unittest.main()
