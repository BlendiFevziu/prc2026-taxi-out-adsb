"""Weather and operating-direction features: parsing, units, time joins, direction changes."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_240 import features_weather as W


def reports(rows):
    metar = pd.DataFrame(rows, columns=["station", "valid", "tmpf", "dwpf", "drct", "sknt", "gust", "vsby", "wxcodes"])
    return W.report_features(metar)


class Phenomena(unittest.TestCase):
    def test_groups_prefixes_and_missing(self):
        self.assertEqual(W.phenomena("SHSGRASN"), {"SH", "SG", "RA", "SN"})
        self.assertEqual(W.phenomena("-FZDZ BR"), {"FZ", "DZ", "BR"})
        self.assertEqual(W.phenomena("VCSH +TSRA"), {"SH", "TS", "RA"})
        self.assertEqual(W.phenomena("M"), set())
        self.assertEqual(W.phenomena(float("nan")), set())


class ReportFeatures(unittest.TestCase):
    def test_units_and_flags(self):
        r = reports([["EHAM", "2026-01-03 00:25", 32.0, 28.4, 280, 16, 25, 1.0, "SHSGRASN"],
                     ["EHAM", "2026-01-03 00:55", 50.0, 50.0, 90, 5, np.nan, 6.0, "FZFG"]])
        self.assertAlmostEqual(r.wx_temp_c[0], 0.0)
        self.assertAlmostEqual(r.wx_dewpoint_spread_c[0], 2.0)
        self.assertAlmostEqual(r.wx_gust_excess_kt[0], 9.0)
        self.assertAlmostEqual(r.wx_gust_excess_kt[1], 0.0)
        self.assertAlmostEqual(r.wx_visibility_km[0], 1.609344)
        self.assertEqual((r.wx_frozen[0], r.wx_freezing[0], r.wx_icing_risk[0]), (1.0, 0.0, 1.0))
        self.assertEqual((r.wx_frozen[1], r.wx_freezing[1], r.wx_icing_risk[1]), (0.0, 1.0, 0.0))


class WeatherJoin(unittest.TestCase):
    def test_latest_report_at_or_before_within_three_hours(self):
        r = reports([["EHAM", "2026-01-03 10:00", 32, 30, 270, 10, np.nan, 5, ""],
                     ["EHAM", "2026-01-03 10:30", 41, 30, 270, 12, np.nan, 5, ""],
                     ["LFPG", "2026-01-03 09:00", 59, 50, 90, 3, np.nan, 10, ""]])
        times = pd.Series(pd.to_datetime(["2026-01-03 10:29", "2026-01-03 09:59", "2026-01-03 14:31",
                                          "2026-01-03 10:30", "2026-01-03 11:00"], utc=True)).dt.as_unit("us")
        out = W.weather_for(["EHAM", "EHAM", "EHAM", "EHAM", "LFPG"], times, r)
        self.assertAlmostEqual(out.wx_wind_kt[0], 10)
        self.assertTrue(np.isnan(out.wx_wind_kt[1]))
        self.assertTrue(np.isnan(out.wx_wind_kt[2]))
        self.assertAlmostEqual(out.wx_wind_kt[3], 12)
        self.assertAlmostEqual(out.wx_wind_kt[4], 3)
        self.assertAlmostEqual(out.wx_age_min[0], 29)

    def test_order_and_index_do_not_matter(self):
        r = reports([["EHAM", "2026-01-03 10:00", 32, 30, 270, 10, np.nan, 5, ""],
                     ["EHAM", "2026-01-03 12:00", 41, 30, 270, 20, np.nan, 5, ""]])
        times = pd.Series(pd.to_datetime(["2026-01-03 12:10", "2026-01-03 10:10"], utc=True), index=[7, 3])
        out = W.weather_for(np.array(["EHAM", "EHAM"]), times, r)
        self.assertEqual(list(out.wx_wind_kt), [20.0, 10.0])


class OperatingDirection(unittest.TestCase):
    def test_hourly_mode_purity_change_and_headwind(self):
        rows = [("AAAA", "2026-01-03 10:%02d" % m, "27L") for m in range(6)] + [("AAAA", "2026-01-03 10:40", "09R")]
        rows += [("AAAA", "2026-01-03 11:%02d" % m, "09L") for m in range(3)]
        rows += [("AAAA", "2026-01-03 12:%02d" % m, "09L") for m in range(5)]
        arrivals = pd.DataFrame(rows, columns=["ADES_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"])
        directions = W.operating_direction(arrivals)
        times = pd.Series(pd.to_datetime(["2026-01-03 10:15", "2026-01-03 11:30", "2026-01-03 12:40"], utc=True))
        out = W.direction_for(["AAAA"] * 3, times, directions, wind_dir_deg=[270, 270, 270], wind_kt=[10, 10, 10])
        self.assertEqual(out.od_dir[0], 27)
        self.assertAlmostEqual(out.od_purity[0], 6 / 7)
        self.assertTrue(np.isnan(out.od_dir[1]))
        self.assertEqual(out.od_dir[2], 9)
        self.assertTrue(np.isnan(out.od_minutes_since_change[0]))
        self.assertAlmostEqual(out.od_minutes_since_change[2], 40)
        self.assertAlmostEqual(out.od_headwind_kt[0], 10)
        self.assertAlmostEqual(out.od_headwind_kt[2], -10)


if __name__ == "__main__":
    unittest.main()
