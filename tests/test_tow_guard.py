from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "adsb-lol")
from research_290 import stand_detector as SD  # noqa: E402
from research_300 import arm_test as AT  # noqa: E402
from research_300 import tow_guard as TG  # noqa: E402

STAND = np.array([[52.30, 4.76]])


def track(rows):
    """rows of (t, metres north of the stand, metres east, speed); every point on the ground."""
    ts = np.array([r[0] for r in rows], float)
    lat = STAND[0, 0] + np.array([r[1] for r in rows], float) / 110540.0
    lon = STAND[0, 1] + np.array([r[2] for r in rows], float) / (111320.0 * np.cos(np.radians(STAND[0, 0])))
    return ts, lat, lon, np.array([r[3] for r in rows], float), np.ones(len(rows), bool)


def run(rows, t_takeoff):
    ts, lat, lon, gs, ground = track(rows)
    onset = AT.guarded(ts, lat, lon, gs, STAND, t_takeoff)
    return TG.check(ts, lat, lon, gs, ground, STAND, onset, t_takeoff)


PARKED = [(t, 5, 0, 0.0) for t in range(0, 3000, 60)]
PUSH = [(3000, 12, 0, 3.0), (3030, 40, 0, 3.0), (3060, 55, 0, 0.0), (3180, 55, 0, 0.0)]    # push-back, engine start
TAXI = [(3240, 120, 0, 12.0), (3300, 600, 0, 15.0), (3360, 1100, 0, 15.0)]


class TowGuardTest(unittest.TestCase):
    def test_ordinary_push_back_is_kept(self):
        out = run(PARKED + PUSH + TAXI + [(3500, 1500, 0, 140.0)], 3500)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["rest_at_spot_s"], 0.0)
        self.assertLess(out["longest_stop_s"], TG.LONG_STOP_S)

    def test_speed_blip_followed_by_a_long_rest_at_the_parked_spot_is_withdrawn(self):
        blip = [(t, 5, 0, 0.0) for t in range(0, 1200, 60)] + [(1210, 8, 0, 2.2)] \
            + [(t, 6, 0, 0.0) for t in range(1220, 1560, 30)]      # resting at the spot 1,220 to 1,550 s: 330 s
        out = run(blip + [(1600, 60, 0, 12.0), (1700, 600, 0, 15.0), (1800, 1100, 0, 15.0), (1900, 1500, 0, 140.0)], 1900)
        self.assertEqual(out["status"], "false_onset")
        self.assertGreaterEqual(out["rest_at_spot_s"], TG.REST_AT_SPOT_S)

    def test_short_rest_at_the_spot_is_kept(self):
        blip = [(t, 5, 0, 0.0) for t in range(0, 1200, 60)] + [(1210, 8, 0, 2.2), (1220, 6, 0, 0.0), (1400, 6, 0, 0.0)]
        out = run(blip + [(1450, 60, 0, 12.0), (1550, 600, 0, 15.0), (1650, 1100, 0, 15.0), (1750, 1500, 0, 140.0)], 1750)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["rest_at_spot_s"], 180.0)

    def test_tow_to_another_stand_with_a_long_stop_is_withdrawn(self):
        tow = [(3240, 300, 0, 10.0), (3400, 1500, 200, 15.0)] + [(t, 1500, 205, 0.0) for t in range(3500, 5700, 60)]
        out = run(PARKED + PUSH + tow + [(5760, 1600, 400, 12.0), (6000, 2500, 400, 15.0), (6100, 3000, 400, 140.0)], 6100)
        self.assertEqual(out["status"], "long_stop")
        self.assertGreaterEqual(out["longest_stop_s"], TG.LONG_STOP_S)

    def test_a_twenty_minute_hold_is_kept(self):
        # the stop is positional, so it starts at the last taxi point, which already sits at 1,100 m (t = 3,360):
        # 3,360 to 4,540 s is 1,180 s, below LONG_STOP_S
        hold = [(t, 1100, 0, 0.0) for t in range(3400, 4600, 60)]
        out = run(PARKED + PUSH + TAXI + hold + [(4700, 1500, 0, 140.0)], 4700)
        self.assertEqual(out["status"], "ok")
        self.assertAlmostEqual(out["longest_stop_s"], 1180.0)

    def test_stops_before_leaving_the_stand_radius_do_not_count(self):
        # the post-push-back stop at 55 m is inside R_STAND_M, so however long it lasts it is no "long stop"
        push = [(3000, 12, 0, 3.0), (3030, 40, 0, 3.0)] + [(t, 55, 0, 0.0) for t in range(3060, 5400, 60)]
        out = run(PARKED + push + [(5460, 120, 0, 12.0), (5600, 1100, 0, 15.0), (5700, 1500, 0, 140.0)], 5700)
        self.assertEqual(out["status"], "ok")
        self.assertLess(out["longest_stop_s"], TG.LONG_STOP_S)

    def test_an_abstention_passes_through_unchanged(self):
        out = run([(0, 500, 0, 0.0), (100, 600, 0, 10.0)], 500)
        self.assertNotIn(out["status"], ("ok", "false_onset", "long_stop"))
        self.assertNotIn("rest_at_spot_s", out)


if __name__ == "__main__":
    unittest.main()
