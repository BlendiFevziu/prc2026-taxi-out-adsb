import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import check_grad

from research_240.main_timestamp_selector import (CLOCKS,NUMERIC,PRIORS,VIEWS,Design,
    candidates,context,linear_loss,predict_linear,predict_neural,soft_weights,source_months,summarize)


class MainTimestampSelectorTests(unittest.TestCase):
    def fixture(self,n=120):
        frame = pd.DataFrame({v:np.ones(n) for v in set(NUMERIC+CLOCKS)})
        frame["ADEP_mvt"],frame["RUNWAY_mvt"] = "TEST","01"
        frame["ersatz_s"] = 1000.
        frame["versp_min"] = 25.
        views = pd.DataFrame({v:np.full(n,1050.) for v in VIEWS})
        return frame,np.full(n,1100.),views

    def test_candidates_do_not_read_labels_or_ids(self):
        frame,base,views = self.fixture()
        c,m = candidates(frame,base,views)
        changed = frame.assign(y=-1e20,BLOCK_TIME_UTC_mvt="secret",TAXITIME_SEC_mvt=1e20,MVT_ID_mvt=999)
        c2,m2 = candidates(changed,base,views)
        np.testing.assert_array_equal(c,c2)
        np.testing.assert_array_equal(m,m2)
        np.testing.assert_array_equal(context(frame,c,m),context(changed,c,m))
        self.assertEqual(c[0,-2],87400)
        self.assertFalse(m[0,-1])

    def test_missing_candidates_have_zero_weight_and_floor_survives(self):
        frame,base,views = self.fixture()
        frame["ersatz_eobt_s"] = np.nan
        c,m = candidates(frame,base,views)
        w = soft_weights(np.zeros((len(c),len(PRIORS)-1)),m)
        self.assertTrue((w[~m]==0).all())
        np.testing.assert_allclose(w.sum(axis=1),1)
        self.assertTrue((predict_linear(np.zeros((1,len(PRIORS)-1)),np.ones((len(c),1)),c,m)>=300).all())

    def test_linear_gradient_including_floor(self):
        rng = np.random.default_rng(123)
        x = rng.normal(size=(8,3))
        c = rng.uniform(500,5000,size=(8,len(PRIORS)))
        c[0] = 100.
        mask = np.ones(c.shape,bool)
        mask[1,7] = False
        c[1,7] = 0
        y = rng.uniform(500,2500,size=8)
        theta = rng.normal(size=3*(len(PRIORS)-1))*.1
        f = lambda t:linear_loss(t,x,c,mask,y)[0]
        g = lambda t:linear_loss(t,x,c,mask,y)[1]
        self.assertLess(check_grad(f,g,theta),1e-3)

    def test_design_fits_training_only_and_handles_unknown_categories(self):
        frame,base,views = self.fixture()
        c,m = candidates(frame,base,views)
        d = Design().fit(frame,c,m)
        original = d.mean.copy()
        other = frame.assign(ADEP_mvt="UNKNOWN",RUNWAY_mvt="99",fid_abw=np.nan)
        self.assertTrue(np.isfinite(d.transform(other,c,m)).all())
        self.assertEqual(d.transform(frame,c,m).shape,d.transform(other,c,m).shape)
        np.testing.assert_array_equal(original,d.mean)
        json.dumps(d.state(),allow_nan=False)

    def test_source_file_membership_overrides_calendar(self):
        frames = {"jan":pd.DataFrame({"MVT_ID_mvt":[11,12]}),"jul":pd.DataFrame({"MVT_ID_mvt":[71,72]})}
        def read(path,columns):
            self.assertEqual(columns,["MVT_ID_mvt"])
            return frames[str(path)].copy()
        with patch("research_240.main_timestamp_selector.pd.read_parquet",side_effect=read):
            got = source_months(pd.DataFrame({"MVT_ID_mvt":[72,11,71],"monat":[1,7,1]}),{1:Path("jan"),7:Path("jul")})
        np.testing.assert_array_equal(got,[7,1,7])

    def test_summary_serialization_and_unchanged_fallback_accounting(self):
        frame = pd.DataFrame({"y":np.ones(120)*1000,"baseline":np.ones(120)*1100,
            "ADEP_mvt":["A"]*120,"source_month":[1,7]*60})
        got = summarize(frame,np.full(120,1000.),120,240,20000.)
        self.assertAlmostEqual(got["overall_rmse"],np.sqrt(15000.))
        self.assertEqual(got["positive_airports"],1)
        self.assertEqual(got["mse_gain_without_top100"],10000.)
        json.dumps(got,allow_nan=False)

    def test_neural_inference_matches_zero_logit_prior_without_training(self):
        import torch
        frame,base,views = self.fixture()
        c,m = candidates(frame,base,views)
        model = torch.nn.Linear(2,len(PRIORS)-1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        x = np.ones((len(c),2))
        got = predict_neural(model,x,c,m)
        expected = np.maximum((soft_weights(np.zeros((len(c),len(PRIORS)-1)),m)*c).sum(axis=1),300)
        np.testing.assert_allclose(got,expected,rtol=0,atol=1e-10)


if __name__ == "__main__":
    unittest.main()
