"""Unit tests for research_370_konfig/features_konfig.py (synthetic data only)."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_370_konfig import features_konfig as K  # noqa: E402


def at(text):
    return pd.Timestamp(text, tz="UTC")


def arrivals(airport, rows):
    """rows: (time, runway) pairs -> raw movements with arrival phase."""
    return pd.DataFrame({"PHASE_mvt": "ARR", "ADES_mvt": airport, "ADEP_mvt": "XXXX",
                         "RUNWAY_mvt": [r for _, r in rows], "MVT_TIME_UTC_mvt": [at(t) for t, _ in rows]})


def lfpg_day():
    rows = [(f"2025-01-10 06:{m:02d}", r) for m, r in zip((5, 15, 25, 35, 45, 55), ("26L",) * 4 + ("27R",) * 2)]
    rows += [(f"2025-01-10 07:{m:02d}", r) for m, r in zip((5, 15, 25, 35, 45, 55), ("26L",) * 5 + ("27R",))]
    rows += [(f"2025-01-10 08:{m:02d}", r) for m, r in zip((10, 20, 30, 40, 50), ("08R",) * 3 + ("09L",) * 2)]
    rows += [(f"2025-01-10 09:{m:02d}", "26L") for m in (10, 20, 30)]                  # three arrivals: '(wenig)'
    rows += [(f"2025-01-10 10:{m:02d}", "26L") for m in (5, 15, 25, 35, 45)]
    return arrivals("LFPG", rows)


def departures(times, airport="LFPG", ades="EGLL"):
    return pd.DataFrame({"ADEP_mvt": airport, "ADES_mvt": ades, "MVT_TIME_UTC_mvt": [at(t) for t in times]},
                        index=pd.Index([f"d{i}" for i in range(len(times))], name="MVT_ID_mvt"))


class RunwayTest(unittest.TestCase):
    def test_designators_are_normalised_and_non_runways_dropped(self):
        name, hdg = K.runway(["7", "25l", " 16R ", "H", "HEL", "NA", "36", "0", "37"])
        self.assertEqual(name.tolist(), ["07", "25L", "16R", None, None, None, "36", None, None])
        np.testing.assert_array_equal(hdg.to_numpy(), [70, 250, 160, np.nan, np.nan, np.nan, 0, np.nan, np.nan])


class ConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.raw = lfpg_day()
        self.dep = departures(["2025-01-10 06:10", "2025-01-10 07:30", "2025-01-10 08:40", "2025-01-10 09:30",
                               "2025-01-10 11:30"])
        self.out = K.attach(self.dep, self.raw, {})

    def test_runway_set_uses_the_twenty_percent_rule_and_few_arrivals(self):
        self.assertEqual(self.out.kf_konfig.tolist(), ["26L+27R", "26L", "08R+09L", "(wenig)", "(wenig)"])

    def test_reversals_are_timed_at_the_first_arrival_on_the_new_side(self):
        # reversals at 08:10 (to side B) and 10:05 (back to A); the '(wenig)' hour at 09:00 is skipped
        np.testing.assert_allclose(self.out.kf_seit_umkehr_min, [720, 720, 30, 80, 85])
        np.testing.assert_allclose(self.out.kf_bis_umkehr_min, [120, 40, 85, 35, 720])

    def test_index_and_column_order_are_kept(self):
        self.assertTrue(self.out.index.equals(self.dep.index))
        self.assertEqual(list(self.out.columns), K.MERKMALE)
        self.assertEqual(self.out.kf_seit_umkehr_min.dtype, float)

    def test_no_runway_above_twenty_percent_falls_back_to_all_runways(self):
        raw = arrivals("EHAM", [(f"2025-01-10 12:{m:02d}", r)
                                for m, r in zip(range(0, 60, 10), ("18R", "18C", "06", "36R", "27", "22"))])
        out = K.attach(departures(["2025-01-10 12:30"], airport="EHAM"), raw, {})
        self.assertEqual(out.kf_konfig.iloc[0], "06+18C+18R+22+27+36R")

    def test_months_are_computed_separately(self):
        rows = [(f"2025-01-31 22:{m:02d}", "26L") for m in (5, 15, 25, 35, 45)]
        rows += [(f"2025-01-31 23:{m:02d}", "08R") for m in (10, 20, 30, 40, 50)]
        out = K.attach(departures(["2025-01-31 23:50", "2025-02-01 00:30"]), arrivals("LFPG", rows), {})
        np.testing.assert_allclose(out.kf_seit_umkehr_min, [40, 720])

    def test_airport_without_principal_heading_keeps_the_set_but_no_flow(self):
        raw = lfpg_day().assign(ADES_mvt="ZZZZ")
        out = K.attach(departures(["2025-01-10 08:40"], airport="ZZZZ"), raw, {})
        self.assertEqual(out.kf_konfig.iloc[0], "08R+09L")
        self.assertTrue(out[["kf_seit_umkehr_min", "kf_bis_umkehr_min"]].isna().all(axis=None))

    def test_labels_departures_and_time_units_do_not_matter(self):
        noisy = pd.concat([self.raw.assign(TAXITIME_SEC_mvt=-1234, BLOCK_TIME_UTC_mvt=at("2000-01-01")),
                           arrivals("LFPG", [("2025-01-10 08:15", "26R")]).assign(PHASE_mvt="DEP")], ignore_index=True)
        pd.testing.assert_frame_equal(self.out, K.attach(self.dep.assign(y=-999), noisy, {}))
        raw_us, dep_us = self.raw.copy(), self.dep.copy()
        raw_us["MVT_TIME_UTC_mvt"] = raw_us.MVT_TIME_UTC_mvt.dt.as_unit("us")
        dep_us["MVT_TIME_UTC_mvt"] = dep_us.MVT_TIME_UTC_mvt.dt.as_unit("us")
        pd.testing.assert_frame_equal(self.out, K.attach(dep_us, raw_us, {}))


class BearingTest(unittest.TestCase):
    COORDS = {"AAAA": (0.0, 0.0), "EAST": (0.0, 10.0), "NRTH": (10.0, 0.0)}

    def test_cardinal_bearings_unknown_and_identical_codes(self):
        s, c = K.bearing(["AAAA", "AAAA", "AAAA", "AAAA"], ["EAST", "NRTH", "????", "AAAA"], self.COORDS)
        np.testing.assert_allclose(s[:2], [1, 0], atol=1e-12)
        np.testing.assert_allclose(c[:2], [0, 1], atol=1e-12)
        self.assertTrue(np.isnan(s[2:]).all() and np.isnan(c[2:]).all())

    def test_load_coords_prefers_icao_code_and_drops_closed(self):
        text = ("id,ident,type,name,latitude_deg,longitude_deg,icao_code,gps_code\n"
                "1,EDDF,heliport,a helipad named like an ICAO code,1,1,,\n"
                "2,DE-0001,large_airport,Frankfurt,50.03,8.57,EDDF,EDDF\n"
                "3,OLD1,closed,gone,5,5,GONE,\n"
                "4,K00A,heliport,gps only,40,-74,,ABCD\n")
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write(text)
        coords = K.load_coords(fh.name)
        self.assertEqual(coords["EDDF"], (50.03, 8.57))
        self.assertEqual(coords["ABCD"], (40.0, -74.0))
        self.assertNotIn("GONE", coords)
        Path(fh.name).unlink()


class ScreenPermuteTest(unittest.TestCase):
    def setUp(self):
        from research_370_konfig.konfig_screen import permute_within
        self.permute = permute_within
        self.frame = pd.DataFrame({"k": [f"s{i}" for i in range(12)], "a": np.arange(12.0), "keep": np.arange(12.0),
                                   "g": ["x"] * 6 + ["y"] * 6})

    def test_dtypes_kept_rows_move_together_within_groups(self):
        out = self.permute(self.frame, ["k", "a"], self.frame.g, 1)
        self.assertEqual(out.a.dtype, float)
        self.assertEqual(out.k.tolist(), [f"s{int(v)}" for v in out.a])
        self.assertEqual(sorted(out.a[:6]), list(range(6)))
        np.testing.assert_array_equal(out.keep, self.frame.keep)
        self.assertFalse(np.array_equal(out.a, self.frame.a))

    def test_same_permutation_as_the_weather_screen_on_numeric_columns(self):
        from research_300.weather_screen import permute_within as weather
        np.testing.assert_array_equal(self.permute(self.frame, ["a"], self.frame.g, 7).a,
                                      weather(self.frame, ["a"], self.frame.g, 7).a)


if __name__ == "__main__":
    unittest.main()
