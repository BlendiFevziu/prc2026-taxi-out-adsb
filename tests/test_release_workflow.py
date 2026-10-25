"""Fast release safety checks: no real training or upload."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_250 import release_common as R
from research_250 import release_pipeline as P
import numpy as np
import pandas as pd


class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value, dtype=float)

    def predict_proba(self, x):
        return np.tile([1-self.value, self.value], (len(x),1))


class ReleaseTest(unittest.TestCase):
    def sample(self):
        frame = pd.DataFrame({"ADEP_mvt": ["A", "B", "A"], "hat_aobt": [1,1,0],
                              "ersatz_s": [1000., 0., np.nan], "versp_min": [30.,-100.,20.],
                              "x": [1,2,3]})
        model = {"ziel_mit_aobt": "rest", "ziel_ohne_aobt": "mischung", "standquote": {},
            "merkmale_haupt": ["x"], "merkmale_rueck": ["x"],
            "mit_aobt": [Constant(100), Constant(200)], "anteil_flh": .2,
            "mit_aobt_flh": {"A": Constant(400)}, "anteil_cb": .6,
            "mit_aobt_cb": [Constant(300), Constant(500)], "anteil_netz": .25,
            "mit_aobt_netz": {}, "erf_klass": Constant(.5), "erf_hoch": 2.5,
            "ohne_aobt": Constant(200), "ohne_aobt_klass": Constant(.2),
            "ohne_aobt_hoehe": Constant(2000), "anteil_hm": .3, "hoch": 2.5}
        return frame, model

    def prediction(self, model, frame):
        with patch.object(R.SR, "wende_an", side_effect=lambda mod,x: x), \
             patch.object(R.H, "kat_str", side_effect=lambda x: x), \
             patch.object(R.NM, "sage_voraus", side_effect=lambda mod,x: np.full(len(x), 500.)):
            return R.predict(model, frame)

    def test_blending_order_reporting_gate_and_both_floors(self):
        frame, model = self.sample()
        pred = self.prediction(model, frame)
        residual = .75*(.4*(.8*150+.2*400)+.6*400)+.25*500
        expected = (1-.5**2.5)*(1000+residual)+.5**2.5*1800
        fallback = (1-.2**2.5)*200+.2**2.5*(.7*1200+.3*2000)
        np.testing.assert_allclose(pred, [expected, 300, fallback], rtol=1e-14)
        self.assertLess(fallback, 300)  # No main-group floor in fallback.
        model["ohne_aobt"] = Constant(-1000)
        self.assertEqual(self.prediction(model, frame)[2], 0)

    def test_unknown_airport_keeps_global_prediction(self):
        frame, model = self.sample()
        model["erf_klass"] = None
        frame.loc[1,"ersatz_s"] = 1000
        result = self.prediction(model, frame)
        expected = 1000+.75*(.4*150+.6*400)+.25*500
        self.assertEqual(result[1], expected)

    def test_rejects_nonfinite_prediction(self):
        frame, model = self.sample()
        model["mit_aobt"] = [Constant(np.nan)]
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.prediction(model, frame)

    def test_rejects_wrong_target(self):
        frame, model = self.sample()
        model["ziel_mit_aobt"] = "absolute"
        with self.assertRaisesRegex(ValueError, "target"):
            self.prediction(model, frame)

    def test_strict_source_start_month(self):
        self.assertEqual(P.month_of_file("training_2025-01-01_2025-02-01.parquet"), 1)
        self.assertEqual(P.month_of_file("training_2025-12-01_2026-01-01.parquet"), 12)
        for bad in ("training_2025-01-01_2025-03-01.parquet", "prefix_training_2025-01-01_2025-02-01.parquet"):
            with self.assertRaises(ValueError):
                P.month_of_file(bad)

    def test_checkpoint_identity_and_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            R.json_write(run/"run.json", {"id": 123})
            value = R.checkpoint(run, "baseline", "test", {"rounds": 7}, lambda: {"fitted": 1})
            def forbidden():
                raise AssertionError("Must not retrain verified checkpoint")
            self.assertEqual(value, R.checkpoint(run, "baseline", "test", {"rounds": 7}, forbidden))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                R.checkpoint(run, "baseline", "test", {"rounds": 8}, forbidden)

    def test_disjoint_ids(self):
        left = pd.DataFrame({"MVT_ID_mvt": [1,2]})
        R.split_check(left, pd.DataFrame({"MVT_ID_mvt": [3]}))
        with self.assertRaises(AssertionError):
            R.split_check(left, pd.DataFrame({"MVT_ID_mvt": [2]}))

    def test_untrimmed_metric_includes_worst_record(self):
        frame = pd.DataFrame({"hat_aobt": [1,1,0], "y": [0.,0.,10000.]})
        score = R.score(frame, np.zeros(3))
        self.assertEqual(score["mse"], 10000.**2/3)
        self.assertEqual(score["fallback_rmse"], 10000.)
        self.assertEqual(score["top5_sse_share"], 1.)

    def test_frozen_order_and_legacy_constants(self):
        self.assertEqual(P.STAGES, ("parity", "tune", "baseline", "candidate", "gate", "final", "export"))
        self.assertEqual((R.H.ANTEIL_CB, R.H.ANTEIL_NETZ, R.H.ANTEIL_FLH), (.6,.25,.2))
        self.assertEqual((R.H.ERF_HOCH, R.H.HOCH, R.H.ANTEIL_HM), (2.5,2.5,.3))
        self.assertEqual(R.H.CB_RUNDEN, 7900)
        self.assertEqual(R.NM.EPOCHEN, 12)
        model = R.tree(890, 0, 8).get_params()
        self.assertEqual((model["learning_rate"], model["num_leaves"], model["min_child_samples"]), (.03,255,50))


if __name__ == "__main__":
    unittest.main()
