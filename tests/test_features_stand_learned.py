"""Unit tests for research_730_adsb_learner/features_stand_learned.py (synthetic data only)."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_730_adsb_learner import features_adsb as F  # noqa: E402
from research_730_adsb_learner import features_stand_learned as L  # noqa: E402

LAND = F.seconds([pd.Timestamp("2025-01-10 10:00:00", tz="UTC")])[0]


def arrival_track(callsign, lat0, lon0, rest=True):
    """Roll-in then a 5-minute stop at (lat0, lon0)."""
    pts = [(10, 49.0 + 0.002, 2.5, 20.0, True), (60, lat0 + 0.0004, lon0, 8.0, True)]
    if rest:
        pts += [(120 + 60 * k, lat0, lon0, 0.4, True) for k in range(6)]
    return pd.DataFrame({"hex": "aaaaaa", "callsign": callsign, "ts": [LAND + p[0] for p in pts], "lat": [p[1] for p in pts],
                         "lon": [p[2] for p in pts], "gs": [p[3] for p in pts], "on_ground": [p[4] for p in pts], "airport": "LFPG"})


def arrivals(callsigns, stand="A1"):
    return pd.DataFrame({"ADES_mvt": "LFPG", "CALLSIGN_flt": callsigns, "MVT_TIME_UTC_mvt": pd.Timestamp("2025-01-10 10:00:00", tz="UTC"),
                         "STAND_mvt": stand})


class LearnedStandTest(unittest.TestCase):
    def test_parked_position_is_the_median_of_the_longest_rest(self):
        t = arrival_track("AFR1", 49.1, 2.6)
        pos = L.parked_position(t.ts.to_numpy(), t.lat.to_numpy(), t.lon.to_numpy(), t.gs.to_numpy(), t.on_ground.to_numpy(), LAND)
        self.assertAlmostEqual(pos[0], 49.1, places=6)
        self.assertAlmostEqual(pos[1], 2.6, places=6)

    def test_no_rest_gives_no_position(self):
        t = arrival_track("AFR1", 49.1, 2.6, rest=False)
        self.assertIsNone(L.parked_position(t.ts.to_numpy(), t.lat.to_numpy(), t.lon.to_numpy(), t.gs.to_numpy(), t.on_ground.to_numpy(), LAND))

    def test_three_agreeing_arrivals_learn_the_stand(self):
        tracks = pd.concat([arrival_track(c, 49.1, 2.6) for c in ("AFR1", "AFR2", "AFR3")], ignore_index=True)
        pts = L.parked_positions(arrivals(["AFR1", "AFR2", "AFR3"]), tracks)
        self.assertEqual(len(pts), 3)
        table = L.learn_table(pts, lambda s: s.upper())
        self.assertIn("A1", table["LFPG"])
        self.assertAlmostEqual(table["LFPG"]["A1"][0][0, 0], 49.1, places=6)

    def test_two_arrivals_are_not_enough(self):
        tracks = pd.concat([arrival_track(c, 49.1, 2.6) for c in ("AFR1", "AFR2")], ignore_index=True)
        table = L.learn_table(L.parked_positions(arrivals(["AFR1", "AFR2"]), tracks), lambda s: s.upper())
        self.assertNotIn("LFPG", table)

    def test_one_position_far_from_the_rest_drops_the_stand(self):
        tracks = pd.concat([arrival_track("AFR1", 49.1, 2.6), arrival_track("AFR2", 49.1, 2.6), arrival_track("AFR3", 49.1008, 2.6)], ignore_index=True)
        table = L.learn_table(L.parked_positions(arrivals(["AFR1", "AFR2", "AFR3"]), tracks), lambda s: s.upper())   # 88 m apart
        self.assertNotIn("LFPG", table)


if __name__ == "__main__":
    unittest.main()
