from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
V56 = ROOT.parent / "skenderbeu-v56-readonly"


@unittest.skipUnless(V56.exists(), "frozen v56 checkout not present")
class StandExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from research_320 import stand_experiment as S
        cls.S = S

    def test_detector_feature_is_takeoff_minus_the_push_back_onset(self):
        stand = np.array([[51.47, -0.45]])
        rows = [(t, 5, 0.0) for t in range(0, 1200, 20)] + [(1210, 20, 3.0), (1230, 45, 3.0)] + \
               [(t, 60, 0.0) for t in range(1250, 1400, 20)] + [(1420, 300, 12.0), (1600, 2000, 20.0), (1800, 3500, 140.0)]
        pts = pd.DataFrame({"ts": [float(r[0]) for r in rows], "lat": [stand[0, 0] + r[1] / 110540.0 for r in rows],
                            "lon": [stand[0, 1]] * len(rows), "gs": [r[2] for r in rows], "on_ground": [True] * len(rows)})
        taxi = self.S.detector_taxi(pts, stand, 1800.0)
        self.assertAlmostEqual(taxi, 1800.0 - (1180 + 1210) / 2)
        self.assertTrue(np.isnan(self.S.detector_taxi(pts, None, 1800.0)))

    def test_controls_keep_the_width_and_move_only_track_columns_within_airport_and_date(self):
        S = self.S
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({c: rng.normal(size=40) for c in S.COLS})
        frame["airport"] = ["EDDF"] * 20 + ["LSZH"] * 20
        frame["date"] = (["2025-01-04"] * 10 + ["2025-01-13"] * 10) * 2
        x = S.inputs(frame, "shuffle_4101")
        self.assertEqual(list(x.columns), S.COLS)
        context = [c for c in S.COLS if c not in S.TRACK]
        pd.testing.assert_frame_equal(x[context], frame[context])
        for _, g in frame.groupby(["airport", "date"]):
            self.assertEqual(sorted(x.loc[g.index, "detector_taxi_s"]), sorted(g.detector_taxi_s))
        self.assertTrue((S.inputs(frame, "context")[S.TRACK] == 0).all().all())


if __name__ == "__main__":
    unittest.main()
