"""Checks for research_290/stand_detector.py on synthetic tracks. No data files, no model fits."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_290 import stand_detector as SD  # noqa: E402

STAND = np.array([[48.3500, 11.7800]])
FAR = (48.3600, 11.7800)          # about 1.1 km north, a taxiway stop


def track(rows):
    """rows of (t, where, speed); where is 'stand' or 'far'."""
    ts = np.array([r[0] for r in rows], float)
    lat = np.array([STAND[0, 0] if r[1] == "stand" else FAR[0] for r in rows])
    lon = np.array([STAND[0, 1] if r[1] == "stand" else FAR[1] for r in rows])
    gs = np.array([r[2] for r in rows], float)
    return ts, lat, lon, gs


class LookupTest(unittest.TestCase):
    def test_unmapped(self):
        self.assertEqual(SD.lookup({}, "A12"), (None, "unmapped"))

    def test_normalised_key_is_found(self):
        vertices, status = SD.lookup({"A12": [STAND]}, "A012")
        self.assertEqual(status, "ok")
        np.testing.assert_array_equal(vertices, STAND)

    def test_candidates_far_apart_are_ambiguous(self):
        far = np.array([[48.3600, 11.7800]])
        self.assertEqual(SD.lookup({"B1": [STAND, far]}, "B1"), (None, "ambiguous"))

    def test_bare_number_maps_to_lettered_subpositions(self):
        a, b = np.array([[48.3500, 11.7800]]), np.array([[48.3502, 11.7800]])   # about 22 m apart
        vertices, status = SD.lookup({"111A": [a], "111B": [b]}, "111")
        self.assertEqual(status, "ok_subdivided")
        self.assertEqual(len(vertices), 2)

    def test_subpositions_far_apart_are_ambiguous(self):
        a, b = np.array([[48.3500, 11.7800]]), np.array([[48.3600, 11.7800]])
        self.assertEqual(SD.lookup({"111A": [a], "111B": [b]}, "111"), (None, "ambiguous"))

    def test_bare_number_never_matches_a_longer_number(self):
        self.assertEqual(SD.lookup({"1111": [STAND], "111AB": [STAND]}, "111"), (None, "unmapped"))

    def test_exact_key_wins_over_subpositions(self):
        vertices, status = SD.lookup({"111": [STAND], "111A": [np.array([[48.36, 11.78]])]}, "111")
        self.assertEqual(status, "ok")
        np.testing.assert_array_equal(vertices, STAND)

    def test_candidates_close_together_are_merged(self):
        near = np.array([[48.35005, 11.78005]])   # a few metres
        vertices, status = SD.lookup({"B1": [STAND, near]}, "B1")
        self.assertEqual(status, "ok")
        self.assertEqual(len(vertices), 2)


class DetectTest(unittest.TestCase):
    def test_clean_pushback(self):
        out = SD.detect(*track([(0, "stand", 0), (30, "stand", 0), (60, "stand", 0), (90, "far", 12)]),
                        STAND, t_takeoff=900)
        self.assertEqual(out["status"], "ok")
        self.assertEqual((out["t_rest"], out["t_leave"], out["gap_s"]), (60.0, 90.0, 30.0))

    def test_later_taxiway_stop_is_never_used(self):
        rows = [(0, "stand", 0), (40, "stand", 0), (70, "far", 10), (300, "far", 0), (360, "far", 0), (420, "far", 15)]
        out = SD.detect(*track(rows), STAND, t_takeoff=900)
        self.assertEqual(out["t_rest"], 40.0)   # not the stationary stretch at 300-360 on the taxiway

    def test_moving_inside_the_radius_counts_as_leaving(self):
        out = SD.detect(*track([(0, "stand", 0), (20, "stand", 0), (40, "stand", 5)]), STAND, t_takeoff=900)
        self.assertEqual((out["t_rest"], out["t_leave"]), (20.0, 40.0))

    def test_last_occupancy_wins_after_an_earlier_one(self):
        rows = [(0, "stand", 0), (50, "stand", 0), (60, "far", 8), (200, "stand", 0), (300, "stand", 0),
                (310, "far", 9)]
        out = SD.detect(*track(rows), STAND, t_takeoff=900)
        self.assertEqual((out["t_rest"], out["t_leave"]), (300.0, 310.0))

    def test_no_stationary_point_at_stand(self):
        out = SD.detect(*track([(0, "stand", 6), (30, "far", 12)]), STAND, t_takeoff=900)
        self.assertEqual(out["status"], "no_occupancy")

    def test_no_departure_observed_before_takeoff(self):
        out = SD.detect(*track([(0, "stand", 0), (30, "stand", 0)]), STAND, t_takeoff=900)
        self.assertEqual(out["status"], "no_exit")

    def test_loose_bracket_abstains(self):
        out = SD.detect(*track([(0, "stand", 0), (500, "far", 12)]), STAND, t_takeoff=900)
        self.assertEqual(out["status"], "gap_too_large")
        self.assertEqual(out["gap_s"], 500.0)

    def test_points_after_takeoff_are_ignored(self):
        rows = [(0, "stand", 0), (30, "far", 12), (2000, "stand", 0), (2100, "far", 12)]
        out = SD.detect(*track(rows), STAND, t_takeoff=900)
        self.assertEqual(out["t_rest"], 0.0)

    def test_input_order_does_not_matter(self):
        rows = [(0, "stand", 0), (30, "stand", 0), (60, "far", 12)]
        a = SD.detect(*track(rows), STAND, t_takeoff=900)
        b = SD.detect(*track(rows[::-1]), STAND, t_takeoff=900)
        self.assertEqual(a, b)

    def test_missing_speed_is_neither_resting_nor_moving(self):
        rows = [(0, "stand", 0), (30, "stand", np.nan), (60, "far", 12)]
        out = SD.detect(*track(rows), STAND, t_takeoff=900)
        self.assertEqual((out["t_rest"], out["t_leave"]), (0.0, 60.0))


def metres_track(rows):
    """rows of (t, metres north of the stand vertex, speed)."""
    ts = np.array([r[0] for r in rows], float)
    lat = STAND[0, 0] + np.array([r[1] for r in rows], float) / 110540.0
    lon = np.full(len(rows), STAND[0, 1])
    gs = np.array([r[2] for r in rows], float)
    return ts, lat, lon, gs


# The development-day pattern of LHX3TR: parked, push-back at 2-3 kt, a stop at 38 m for engine start, taxi.
PUSHBACK = [(-31, 6, 0.0), (0, 6, 0.0), (13, 19, 2.2), (40, 30, 3.2), (67, 38, 0.5), (135, 38, 0.0),
            (200, 38, 0.0), (239, 34, 3.8), (247, 36, 5.8), (264, 84, 7.8), (300, 250, 10.2)]


class OnsetTest(unittest.TestCase):
    def test_pushback_onset_not_taxi_start(self):
        out = SD.detect_onset(*metres_track(PUSHBACK), STAND, t_takeoff=900, r_park_m=30)
        self.assertEqual(out["status"], "ok")
        self.assertEqual((out["t_before"], out["t_after"]), (0.0, 13.0))
        self.assertAlmostEqual(out["parked_distance_m"], 6.0, places=3)

    def test_old_rule_is_late_on_the_same_track(self):
        out = SD.detect(*metres_track(PUSHBACK), STAND, t_takeoff=900)
        self.assertEqual(out["t_rest"], 200.0)      # the post-push-back stop, which is why the rule changed

    def test_unobserved_parking_abstains_instead_of_taking_the_taxi_start(self):
        rows = [(31, 4, 3.2), (60, 38, 2.2), (138, 39, 0.0), (203, 38, 0.0), (222, 36, 3.2), (238, 58, 6.8),
                (274, 246, 13.8)]
        out = SD.detect_onset(*metres_track(rows), STAND, t_takeoff=900, r_park_m=30)
        self.assertEqual(out["status"], "no_parked_onset")

    def test_power_out_departure(self):
        rows = [(0, 5, 0.0), (60, 5, 0.0), (80, 12, 6.0), (110, 150, 9.0)]
        out = SD.detect_onset(*metres_track(rows), STAND, t_takeoff=900, r_park_m=30)
        self.assertEqual((out["t_before"], out["t_after"]), (60.0, 80.0))

    def test_arrival_in_the_same_run_is_skipped(self):
        rows = [(0, 60, 6.0), (20, 25, 3.0), (40, 8, 1.0), (100, 5, 0.0), (500, 5, 0.0), (520, 20, 2.5),
                (600, 38, 0.0), (700, 150, 9.0)]
        out = SD.detect_onset(*metres_track(rows), STAND, t_takeoff=900, r_park_m=30)
        self.assertEqual((out["t_before"], out["t_after"]), (500.0, 520.0))

    def test_parked_distance_bound(self):
        rows = [(0, 35, 0.0), (20, 50, 3.0), (80, 200, 9.0)]
        self.assertEqual(SD.detect_onset(*metres_track(rows), STAND, 900, r_park_m=30)["status"], "no_parked_onset")
        self.assertEqual(SD.detect_onset(*metres_track(rows), STAND, 900, r_park_m=40)["status"], "ok")

    def test_loose_bracket_abstains(self):
        rows = [(0, 5, 0.0), (400, 30, 3.0), (460, 200, 9.0)]
        out = SD.detect_onset(*metres_track(rows), STAND, t_takeoff=900, r_park_m=30)
        self.assertEqual(out["status"], "gap_too_large")

    def test_tow_inside_the_stand_area_abstains(self):
        rows = [(0, 5, 0.0), (20, 22, 2.5), (100, 5, 0.0), (3500, 5, 0.0), (3520, 20, 2.5), (3600, 150, 9.0)]
        out = SD.detect_onset(*metres_track(rows), STAND, t_takeoff=4000, r_park_m=30)
        self.assertEqual(out["status"], "ambiguous_onsets")
        self.assertEqual(out["onsets"], 2)
        self.assertGreater(out["onset_spread_s"], SD.MAX_ONSET_SPREAD_S)

    def test_post_pushback_stop_inside_the_bound_still_gives_the_pushback(self):
        rows = [(0, 5, 0.0), (15, 15, 2.8), (70, 25, 0.0), (200, 25, 0.0), (240, 27, 3.8), (260, 120, 8.0)]
        out = SD.detect_onset(*metres_track(rows), STAND, t_takeoff=900, r_park_m=30)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["onsets"], 2)
        self.assertEqual((out["t_before"], out["t_after"]), (0.0, 15.0))   # the first, not the taxi start at 240

    def test_never_at_stand(self):
        rows = [(0, 500, 0.0), (60, 600, 9.0)]
        self.assertEqual(SD.detect_onset(*metres_track(rows), STAND, 900, r_park_m=30)["status"], "no_occupancy")

    def test_input_order_does_not_matter(self):
        a = SD.detect_onset(*metres_track(PUSHBACK), STAND, 900, r_park_m=30)
        b = SD.detect_onset(*metres_track(PUSHBACK[::-1]), STAND, 900, r_park_m=30)
        self.assertEqual(a, b)



# The 3 December 2025 failure at stand 215: landing roll, taxi-in, a stop-and-creep step while docking inside the
# parked bound, a turnaround parked 36 m out (beyond R_PARK_M), then the push-back. v1 took the docking step.
ARRIVAL_DOCKING = [(0, 2500, 140.0), (60, 1500, 25.0), (200, 300, 15.0), (240, 70, 6.0), (250, 24, 1.4),
                   (255, 18, 2.8), (270, 36, 0.1), (3400, 36, 0.0), (3430, 33, 3.2), (3500, 150, 9.0)]


class OnsetV2Test(unittest.TestCase):
    def test_v1_takes_the_docking_step(self):
        out = SD.detect_onset(*metres_track(ARRIVAL_DOCKING), STAND, t_takeoff=4000, r_park_m=30)
        self.assertEqual((out["status"], out["t_before"], out["t_after"]), ("ok", 250.0, 255.0))

    def test_v2_rejects_a_transition_right_after_fast_motion(self):
        out = SD.detect_onset_v2(*metres_track(ARRIVAL_DOCKING), STAND, t_takeoff=4000, r_park_m=30)
        self.assertEqual(out["status"], "arrival_transition")

    def test_v2_equals_v1_on_an_ordinary_pushback(self):
        a = SD.detect_onset(*metres_track(PUSHBACK), STAND, 900, r_park_m=30)
        b = SD.detect_onset_v2(*metres_track(PUSHBACK), STAND, 900, r_park_m=30)
        self.assertEqual(a, b)

    def test_fast_motion_outside_the_window_is_ignored(self):
        rows = [(0, 2000, 140.0), (100, 60, 6.0), (120, 8, 1.0), (180, 5, 0.0), (1500, 5, 0.0), (1520, 20, 2.5),
                (1600, 150, 9.0)]
        out = SD.detect_onset_v2(*metres_track(rows), STAND, t_takeoff=2000, r_park_m=30)
        self.assertEqual((out["status"], out["t_before"], out["t_after"]), ("ok", 1500.0, 1520.0))

    def test_window_starts_exactly_at_t_before_minus_the_window(self):
        rows = [(900, 2000, 140.0), (1400, 5, 0.0), (1500, 5, 0.0), (1520, 20, 2.5), (1600, 150, 9.0)]
        out = SD.detect_onset_v2(*metres_track(rows), STAND, t_takeoff=2000, r_park_m=30)
        self.assertEqual(out["status"], "arrival_transition")       # 1500 - 600 = 900 is inside
        rows[0] = (899, 2000, 140.0)
        out = SD.detect_onset_v2(*metres_track(rows), STAND, t_takeoff=2000, r_park_m=30)
        self.assertEqual(out["status"], "ok")

    def test_missing_speed_in_the_window_is_ignored(self):
        rows = [(1000, 2000, np.nan), (1400, 5, 0.0), (1500, 5, 0.0), (1520, 20, 2.5), (1600, 150, 9.0)]
        out = SD.detect_onset_v2(*metres_track(rows), STAND, t_takeoff=2000, r_park_m=30)
        self.assertEqual(out["status"], "ok")

    def test_abstentions_of_v1_pass_through(self):
        rows = [(0, 5, 0.0), (400, 30, 3.0), (460, 200, 9.0)]
        a = SD.detect_onset(*metres_track(rows), STAND, t_takeoff=900, r_park_m=30)
        b = SD.detect_onset_v2(*metres_track(rows), STAND, t_takeoff=900, r_park_m=30)
        self.assertEqual((a, b["status"]), (b, "gap_too_large"))

    def test_input_order_does_not_matter(self):
        a = SD.detect_onset_v2(*metres_track(ARRIVAL_DOCKING), STAND, 4000, r_park_m=30)
        b = SD.detect_onset_v2(*metres_track(ARRIVAL_DOCKING[::-1]), STAND, 4000, r_park_m=30)
        self.assertEqual(a, b)

if __name__ == "__main__":
    unittest.main()
