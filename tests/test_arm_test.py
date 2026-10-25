from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "adsb-lol")
import offblock_from_tracks as OB  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402
from research_300 import arm_test as AT  # noqa: E402


def frames():
    dep = pd.DataFrame({"MVT_ID_mvt": [1, 2, 3], "airport": ["EHAM"] * 3, "mvt_epoch": [1000.0, 2000.0, 3000.0],
                        "callsign": ["KL0981", "KLM702", "LH002"], "CALLSIGN_flt": ["KLM47M", None, "DLH1TN"]})
    visits = pd.DataFrame({"hex": ["a", "b", "c", "d"], "airport": ["EHAM"] * 4,
                           "callsign_adsb": ["KLM47M", "KLM702", "DLH1TN", "KLM981"],
                           "t_first": [0.0, 0.0, 0.0, 0.0], "t_last": [1100.0, 2100.0, 3900.0, 1050.0],
                           "points": [50, 50, 50, 50]})
    return dep, visits


class MatchTest(unittest.TestCase):
    def test_reproduces_the_original_match_with_its_own_candidates(self):
        dep, visits = frames()
        mapping = {"KL": "KLM", "LH": "DLH"}
        mine = AT.match_candidates(dep, visits, OB.candidates(dep.callsign, mapping))
        orig = OB.match(dep, visits, mapping)
        pd.testing.assert_frame_equal(mine.reset_index(drop=True), orig.reset_index(drop=True))

    def test_filed_callsign_adds_matches_only_within_the_time_window(self):
        dep, visits = frames()
        mapping = {"KL": "KLM", "LH": "DLH"}
        base = AT.match_candidates(dep, visits, AT.options_for(dep, "flight_mvt", mapping))
        plus = AT.match_candidates(dep, visits, AT.options_for(dep, "plus_callsign_flt", mapping))
        self.assertEqual(sorted(base.MVT_ID_mvt), [1, 2])        # KL0981 -> KLM981 visit ends 50 s after
        self.assertEqual(sorted(plus.MVT_ID_mvt), [1, 2])        # DLH1TN visit ends 900 s after: still no match
        self.assertEqual(plus.set_index("MVT_ID_mvt").loc[1, "callsign_adsb"], "KLM981")  # closest visit wins

    def test_missing_filed_callsign_changes_nothing(self):
        dep, visits = frames()
        opts = AT.options_for(dep, "plus_callsign_flt", {})
        self.assertEqual(opts[1], {"KLM702"})


class LookupTest(unittest.TestCase):
    def test_xplane_first_falls_back_to_osm(self):
        osm = {"A1": [np.array([[52.0, 4.0]])], "B2": [np.array([[52.1, 4.1]])]}
        xp = {"A1": [np.array([[52.0001, 4.0001]])]}
        v, s = AT.lookup("xplane_first", osm, xp, "A01")
        self.assertEqual(s, "xplane_ok")
        np.testing.assert_allclose(v, [[52.0001, 4.0001]])
        v, s = AT.lookup("xplane_first", osm, xp, "B2")
        self.assertEqual(s, "ok")
        v, s = AT.lookup("osm", osm, xp, "A1")
        np.testing.assert_allclose(v, [[52.0, 4.0]])


STAND = np.array([[48.35, 11.78]])


def track(rows):
    """rows of (t, metres north of the stand, speed)."""
    ts = np.array([r[0] for r in rows], float)
    lat = STAND[0, 0] + np.array([r[1] for r in rows], float) / 110540.0
    return ts, lat, np.full(len(rows), STAND[0, 1]), np.array([r[2] for r in rows], float)


class GuardTest(unittest.TestCase):
    def test_late_docking_step_is_withdrawn(self):
        # the 29 July 2026 pattern: landing at 0, docking step at 995-1005 (beyond the 600 s rule), then parked at
        # 36 m, BEYOND R_PARK, so the push-back at 4010 is no transition and v2 alone would take the docking step
        rows = [(0, 3000, 140.0), (300, 1500, 20.0), (900, 70, 8.0), (995, 20, 1.2), (1005, 15, 2.6),
                (1060, 36, 0.0), (3900, 36, 0.0), (4010, 50, 3.0), (4100, 200, 9.0)]
        self.assertEqual(SD.detect_onset_v2(*track(rows), STAND, 4600, SD.R_PARK_M)["status"], "ok")
        self.assertEqual(AT.guarded(*track(rows), STAND, 4600)["status"], "arrival_stay_guard")

    def test_ordinary_pushback_after_a_turnaround_is_kept(self):
        rows = [(0, 5, 0.0), (2400, 5, 0.0), (2420, 20, 2.5), (2470, 45, 3.0), (2600, 200, 9.0)]
        out = AT.guarded(*track(rows), STAND, 3200)
        self.assertEqual((out["status"], out["t_before"], out["t_after"]), ("ok", 2400.0, 2420.0))


if __name__ == "__main__":
    unittest.main()
