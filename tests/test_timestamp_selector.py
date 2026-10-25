import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import check_grad

from research_240.timestamp_selector import (Design, NUMERIC, candidate_matrix, context,
    fit_gate, loss_gradient, metrics, outer_inner_masks, source_month_index, weights)


class TimestampSelectorTests(unittest.TestCase):
    def fixture(self):
        frame = pd.DataFrame({name: [1.0, 2.0, 3.0] for name in NUMERIC})
        frame["ADEP_mvt"] = ["A", "B", "A"]
        frame["versp_min"] = [-30, 60, 1500]
        frame["nt_anchor"] = [900, np.nan, 1200]
        parts = pd.DataFrame({"baseline": [1100,1200,1300], "normal": [1000,1100,1200],
                              "high": [5000,6000,7000], "p": [.1,.2,.3],
                              "q_stand": [.1,.1,.1], "q_stand_n": [2,3,4]})
        return frame, parts

    def test_candidate_masks_day_units_and_label_independence(self):
        frame, parts = self.fixture()
        c, mask = candidate_matrix(frame,parts)
        self.assertFalse(mask[0,3])
        self.assertFalse(mask[1,4])
        self.assertEqual(c[0,6],87300)
        self.assertEqual(c[2,7],3600)
        self.assertFalse(mask[2,8])
        frame["y"],frame["BLOCK_TIME_UTC_mvt"],frame["TAXITIME_SEC_mvt"] = 999999, "leak", -9
        c2,m2 = candidate_matrix(frame,parts)
        np.testing.assert_array_equal(c,c2)
        np.testing.assert_array_equal(mask,m2)
        self.assertTrue(np.isfinite(c).all())

    def test_analytic_gradient(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(9,3))
        c = rng.uniform(0,90000,size=(9,4))
        mask = np.ones(c.shape,bool)
        mask[1,2] = False
        c[1,2] = 0
        y = rng.uniform(0,90000,size=9)
        theta = rng.normal(size=9)*.1
        f = lambda t: loss_gradient(t,x,c,mask,y,.1)[0]
        g = lambda t: loss_gradient(t,x,c,mask,y,.1)[1]
        self.assertLess(check_grad(f,g,theta), 1e-3)
        w = weights(theta.reshape(3,3),x,mask)
        np.testing.assert_allclose(w.sum(axis=1),1)
        self.assertEqual(w[1,2],0)

    def test_train_only_design_and_unknown_airport(self):
        frame,parts = self.fixture()
        design = Design().fit(frame,parts)
        old = design.mean.copy()
        test = frame.copy()
        test["ADEP_mvt"] = "UNKNOWN"
        test["y"] = 1e20
        test["um_seit_ank"] = np.nan
        self.assertTrue(np.isfinite(design.transform(test,parts)).all())
        np.testing.assert_array_equal(old,design.mean)
        np.testing.assert_equal(context(frame,parts),context(frame.assign(y=999),parts))

    def test_inner_folds_never_use_outer_pair(self):
        months = np.array([2,3,4,5,6,8,9,10,11,12])
        covered = np.zeros(len(months),int)
        for pair, hold in outer_inner_masks(months):
            self.assertTrue(set(months[hold]).isdisjoint(months[~hold]))
            self.assertFalse(set(pair) & {1,7})
            covered += hold
        np.testing.assert_array_equal(covered,1)

    def test_source_file_start_month_not_calendar_or_end_month(self):
        paths = [Path(f"training_2025-{m:02d}-01_2025-{m%12+1:02d}-01.parquet") for m in range(1,13)]
        # Calendar timestamps are deliberately from another month, but never read.
        def read(path, columns):
            self.assertEqual(columns, ["MVT_ID_mvt"])
            return pd.DataFrame({"MVT_ID_mvt": [int(path.name[14:16])*100]})
        with patch("research_240.timestamp_selector.pd.read_parquet",side_effect=read):
            index = source_month_index(paths)
        self.assertEqual(index.loc[100],1)
        self.assertEqual(index.loc[1200],12)

    def test_duplicate_source_ids_rejected(self):
        paths = [Path(f"training_2025-{m:02d}-01_next.parquet") for m in range(1,13)]
        with patch("research_240.timestamp_selector.pd.read_parquet",return_value=pd.DataFrame({"MVT_ID_mvt":[1]})):
            with self.assertRaises(ValueError):
                source_month_index(paths)

    def test_metrics_export_is_json_serializable(self):
        frame = pd.DataFrame({"y":np.arange(120),"baseline":np.arange(120)+1,
                              "fold":["1_7"]*120,"monat":[1,7]*60})
        report = metrics(frame,frame.y.to_numpy())
        self.assertEqual(json.loads(json.dumps(report))["positive_pairs"],1)

    def test_direct_squared_error_fit_improves_training_objective(self):
        x = np.ones((30,1))
        c = np.tile([100,300],(30,1)).astype(float)
        mask = np.ones(c.shape,bool)
        y = np.full(30,300.)
        theta, info = fit_gate(x,c,mask,y,.001)
        before = loss_gradient(np.zeros(1),x,c,mask,y,.001)[0]
        self.assertTrue(info["success"])
        self.assertLess(loss_gradient(theta.ravel(),x,c,mask,y,.001)[0],before)


if __name__ == "__main__":
    unittest.main()
