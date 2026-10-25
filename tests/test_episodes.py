from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_320 import episodes as E  # noqa: E402

STAND = np.array([[50.03, 8.57]])


def track(rows, ground=True):
    """rows of (t, metres north of the stand, speed)."""
    return pd.DataFrame({"ts": [float(r[0]) for r in rows],
                         "lat": [STAND[0, 0] + r[1] / 110540.0 for r in rows],
                         "lon": [STAND[0, 1]] * len(rows), "gs": [float(r[2]) for r in rows],
                         "on_ground": [ground] * len(rows)})


class EpisodeTest(unittest.TestCase):
    def test_stand_wait_push_back_taxi_and_runway_hold_in_order(self):
        rows = [(t, 5, 0) for t in range(0, 600, 30)]                  # parked 0-570 at the stand
        rows += [(600, 30, 3), (630, 55, 3)] + [(t, 60, 0) for t in range(660, 840, 30)]   # push, stop 630-810
        rows += [(870, 200, 12), (930, 900, 15)] + [(t, 1500, 0) for t in range(960, 1500, 60)]  # hold 960-1440
        rows += [(1500, 1600, 20), (1560, 2500, 120)]
        out = E.episodes(track(rows), STAND, 1560)
        self.assertEqual(out["stop_count"], 3.0)
        self.assertEqual(out["last_stop_s"], 480.0)                    # the runway hold
        self.assertEqual(out["last_stop_end_age_s"], 120.0)
        self.assertAlmostEqual(out["last_stop_stand_m"], 1500.0, delta=1.0)
        self.assertEqual(out["prev_stop_s"], 180.0)                    # 630 to 810, the post-push-back stop
        # the wait at the stand: the first push-back point (600 s, 30 m) is 25 m from the parked point (5 m), inside
        # STOP_RADIUS_M, so by definition the stand stop runs 0 to 600 s
        self.assertEqual(out["longest_stop_s"], 600.0)
        self.assertEqual(out["stand_stop_s"], 180.0)                   # the LAST stop within 80 m
        self.assertEqual(out["stopped_after_stand_s"], 480.0)          # only the hold starts beyond 80 m

    def test_window_and_ground_mask_match_the_summaries(self):
        rows = [(t, 5, 0) for t in range(0, 300, 30)]
        early = track([(-8000, 5, 0), (-7900, 5, 0)])                  # before the two-hour window
        airborne = track([(200, 400, 150)], ground=False)
        pts = pd.concat([early, track(rows), airborne], ignore_index=True)
        out = E.episodes(pts, STAND, 7000)
        self.assertEqual(out["stop_count"], 1.0)
        self.assertEqual(out["last_stop_s"], 270.0)

    def test_creeping_movement_is_not_a_stop(self):
        rows = [(t, 5 + 0.9 * t, 3.5) for t in range(0, 120, 5)]       # 3.5 kt, 108 m in 115 s
        out = E.episodes(track(rows), STAND, 200)
        self.assertEqual(out["stop_count"], 0.0)
        self.assertTrue(np.isnan(out["last_stop_s"]))

    def test_no_ground_points_gives_missing_values(self):
        out = E.episodes(track([(0, 5, 0), (60, 5, 0)], ground=False), STAND, 100)
        self.assertTrue(all(np.isnan(v) for v in out.values()))

    def test_unmapped_stand_leaves_stand_fields_missing(self):
        rows = [(t, 5, 0) for t in range(0, 300, 30)]
        out = E.episodes(track(rows), None, 400)
        self.assertEqual(out["stop_count"], 1.0)
        self.assertTrue(np.isnan(out["last_stop_stand_m"]) and np.isnan(out["stand_stop_s"]))

    def test_string_ground_flags_are_refused(self):
        with self.assertRaises(ValueError):
            E.episodes(track([(0, 5, 0), (60, 5, 0)], ground="yes"), STAND, 100)

    def test_ground_mask_equals_codex_implementation(self):
        v56 = ROOT.parent / "skenderbeu-v56-readonly"
        if not v56.exists():
            self.skipTest("frozen v56 checkout not present")
        sys.path.insert(0, str(v56))
        try:
            from research_310.partial_track_audit import ground_mask as reference
        finally:
            sys.path.remove(str(v56))
        values = pd.Series([True, False, None, 1, 0, np.nan], dtype=object)
        for a, b in zip(E.ground_mask(values), reference(values)):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
