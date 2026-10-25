"""Research feature contracts; v44 production definitions remain untouched."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from research_250 import features_record_time as R
from research_250 import features_reporting_context as C
from research_250 import features_time_units as U
from research_250 import features_correct_queue as Q
from research_250 import candidate_features as B


class TimeUnitResearchTest(unittest.TestCase):
    def test_corrected_clock_windows_and_units(self):
        U._self_check()

    def test_record_time_contracts(self):
        R._self_check()

    def test_reporting_context_full_contracts(self):
        C._self_check()

    def test_correct_clock_queue_contracts(self):
        Q._self_check()

    def test_candidate_schema_keeps_fallback_and_legacy_lists(self):
        import main_model as H
        original_main, original_fallback = list(H.MERKMALE_HAUPT), list(H.MERKMALE_RUECK)
        spec = B.schema(original_main, original_fallback)
        self.assertEqual(len(spec["merkmale_haupt"]), 91)
        self.assertEqual(len(set(spec["merkmale_haupt"])), 91)
        self.assertEqual(spec["merkmale_rueck"], original_fallback)
        self.assertEqual(H.MERKMALE_HAUPT, original_main)
        self.assertEqual(H.MERKMALE_RUECK, original_fallback)
        self.assertTrue(set(U.MERKMALE+Q.MERKMALE) <= set(spec["merkmale_haupt"]))
        self.assertTrue(set(U.CORRECTED_COLUMNS).isdisjoint(spec["merkmale_haupt"]))
        self.assertFalse(spec["production_ready"])
        with self.assertRaises(ValueError):
            B.main_features(spec["merkmale_haupt"])

    def test_candidate_attach_preserves_legacy_values_and_ignores_labels(self):
        dep = pd.DataFrame({"ADEP_mvt": ["TEST"]*5, "STAND_mvt": ["A"]*5,
                            "RUNWAY_mvt": ["01"]*5,
                            "MVT_TIME_UTC_mvt": pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC"),
                            "FLIGHT_ID_mvt": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)})
        dep["ADEP_mvt"] = dep.ADEP_mvt.astype("category")
        for col in U.CORRECTED_COLUMNS:
            dep[col] = -123.
        raw = dep.assign(PHASE_mvt="DEP", ADES_mvt="OTHER")[B.RAW_COLUMNS]
        original = dep.copy(deep=True)
        built = B.attach(raw, dep)
        assert_frame_equal(dep, original)
        assert_frame_equal(built[dep.columns], original)
        poison = dep.assign(y=-99999, TAXITIME_SEC_mvt=-99999,
                            BLOCK_TIME_UTC_mvt=pd.NaT)
        assert_frame_equal(built[B.NEW_COLUMNS],
                           B.attach(raw.assign(TAXITIME_SEC_mvt=-99999), poison)[B.NEW_COLUMNS])
        with self.assertRaises(ValueError):
            B.attach(raw, built)

    def test_queue_arrival_airport_and_departure_airport_are_isolated(self):
        t = pd.Timestamp("2025-01-01", tz="UTC")
        dep = pd.DataFrame({"ADEP_mvt": ["TEST", "OTHER"], "RUNWAY_mvt": ["01", "01"],
                            "MVT_TIME_UTC_mvt": [t, t]})
        raw = dep.assign(PHASE_mvt="DEP", ADES_mvt="DEST")[Q.RAW_COLUMNS]
        arrival = pd.DataFrame({"ADEP_mvt": ["ORIGIN"], "ADES_mvt": ["TEST"],
                                "RUNWAY_mvt": ["01"], "PHASE_mvt": ["ARR"],
                                "MVT_TIME_UTC_mvt": [t-pd.Timedelta(minutes=2)]})
        built = Q.baue(pd.concat([raw, arrival], ignore_index=True), dep)
        self.assertEqual(built.qt_bahn_ank_vor_10.tolist(), [1, 0])
        self.assertEqual(built.qt_flh_vor_10.tolist(), [1, 1])
        self.assertEqual(built.qt_bahn_luecke_vor.tolist(), [-1, -1])

    def test_pooled_main_record_is_weighted_by_rows(self):
        path = Path(__file__).resolve().parents[1]/"research_250/results/time_units_six_pair.json"
        report = json.loads(path.read_text())["main_single_lightgbm"]
        folds = report["folds"]
        n = sum(f["rows"] for f in folds)
        self.assertEqual(n, report["rows"])
        for name in ("baseline_rmse", "candidate_rmse"):
            expected = np.sqrt(sum(f["rows"]*f[name]**2 for f in folds)/n)
            self.assertAlmostEqual(expected, report[name], places=9)
        self.assertEqual(sum(f["mse_gain"] > 0 for f in folds), report["positive_folds"])

    def test_pooled_fallback_record_is_weighted_by_rows(self):
        path = Path(__file__).resolve().parents[1]/"research_250/results/fallback_six_pair.json"
        report = json.loads(path.read_text())["fallback"]
        folds = report["folds"]
        n = sum(f["rows"] for f in folds)
        self.assertEqual(n, report["rows"])
        for name in ("baseline_rmse", "candidate_rmse"):
            expected = np.sqrt(sum(f["rows"]*f[name]**2 for f in folds)/n)
            self.assertAlmostEqual(expected, report[name], places=9)
        self.assertEqual(sum(f["mse_gain"] > 0 for f in folds), report["positive_folds"])

    def test_queue_results_keep_screen_confirmation_and_weights_separate(self):
        path = Path(__file__).resolve().parents[1]/"research_250/results/queue_six_pair.json"
        record = json.loads(path.read_text())
        self.assertTrue(record["official_score_unchanged"])
        self.assertEqual(record["official_best_rmse"], 272.334)
        self.assertEqual(len(record["fold_reports"]), 6)
        for key, expected_folds in (("all_six_vs_v44_component", 6),
                                    ("all_six_incremental_over_corrected_units", 6),
                                    ("five_confirmation_pairs_incremental", 5)):
            report = record[key]
            folds = report["folds"]
            self.assertEqual(len(folds), expected_folds)
            n = sum(f["rows"] for f in folds)
            self.assertEqual(n, report["rows"])
            for name in ("baseline_rmse", "candidate_rmse"):
                expected = np.sqrt(sum(f["rows"]*f[name]**2 for f in folds)/n)
                self.assertAlmostEqual(expected, report[name], places=9)
        self.assertNotIn("1_7", [f["fold"] for f in record["five_confirmation_pairs_incremental"]["folds"]])

    def test_reference_airport_for_arrivals_and_nan_time(self):
        base = pd.Timestamp("2025-01-01", tz="UTC")
        dep = pd.DataFrame({"ADEP_mvt": ["TEST", "OTHER", "TEST"],
                            "STAND_mvt": ["A", "A", "A"],
                            "MVT_TIME_UTC_mvt": [base, base, pd.NaT],
                            "FLIGHT_ID_mvt": [1., 2., 3.]})
        raw = dep.assign(PHASE_mvt="DEP", ADES_mvt="DEST")[U.RAW_COLUMNS]
        arrival = pd.DataFrame({"PHASE_mvt": ["ARR"], "ADEP_mvt": ["ORIGIN"],
                                "ADES_mvt": ["TEST"], "STAND_mvt": ["A"],
                                "MVT_TIME_UTC_mvt": [base+pd.Timedelta(minutes=30)]})
        got = U.corrected_values(pd.concat([raw, arrival], ignore_index=True), dep)
        self.assertEqual(got.iloc[0].um_dichte_stand_3h, 2)
        self.assertEqual(got.iloc[1].um_dichte_stand_3h, 1)
        self.assertTrue(pd.isna(got.iloc[2].auslast))

    def test_reporting_context_ns_us_and_no_labels(self):
        n = 250
        base = pd.Timestamp("2025-01-01", tz="UTC")
        d = pd.DataFrame({"ADEP_mvt": ["TEST"]*n, "MVT_ID_mvt": np.arange(n),
                          "FLIGHT_ID_mvt": np.arange(n, dtype=float),
                          "MVT_TIME_UTC_mvt": pd.date_range(base, periods=n, freq="min"),
                          "SCHED_TIME_UTC_mvt": pd.date_range(base, periods=n, freq="min")})
        d.loc[40:44, "FLIGHT_ID_mvt"] = np.nan
        expected = C.baue(d)[C.MERKMALE]
        us = d.copy()
        for col in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"):
            us[col] = us[col].dt.as_unit("us")
        assert_frame_equal(expected, C.baue(us)[C.MERKMALE])
        poison = d.assign(y=-100000, TAXITIME_SEC_mvt=-123456,
                          BLOCK_TIME_UTC_mvt=base)
        assert_frame_equal(expected, C.baue(poison)[C.MERKMALE])


if __name__ == "__main__":
    unittest.main()
