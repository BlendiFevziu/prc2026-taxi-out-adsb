"""Checks for research_280/features_nm_proxy.py. No data files, no model fits."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_280 import features_nm_proxy as NP  # noqa: E402

T0 = pd.Timestamp("2025-01-03 06:00", tz="UTC")


def frame(unit="ns"):
    """EHAM rows at 0, 10, 20, 20, 30, 70 min; plan carried over (AOBT == EOBT_1) where flag is 1.
    One LTFM row with the flag set must never reach EHAM, and one fallback row has no AOBT."""
    spec = [("EHAM", 0, 1), ("EHAM", 10, 0), ("EHAM", 20, 1), ("EHAM", 20, 0),
            ("EHAM", 30, 0), ("EHAM", 70, 1), ("LTFM", 15, 1), ("EHAM", 25, None)]
    rows = []
    for airport, minute, flag in spec:
        mvt = T0 + pd.Timedelta(minutes=minute)
        eobt = mvt - pd.Timedelta(minutes=20)
        aobt = pd.NaT if flag is None else (eobt if flag else eobt + pd.Timedelta(minutes=3))
        rows.append({"ADEP_mvt": airport, "MVT_TIME_UTC_mvt": mvt, "AOBT_3_flt": aobt,
                     "EOBT_1_flt": eobt, "IOBT_flt": eobt - pd.Timedelta(hours=1),
                     "LOBT_flt": eobt - pd.Timedelta(hours=1),
                     "ARVT_1_flt": mvt + pd.Timedelta(hours=2, seconds=17),
                     "ARVT_3_flt": mvt + pd.Timedelta(hours=2)})
    d = pd.DataFrame(rows)
    for c in d.columns:
        if c.endswith(("_mvt", "_flt")) and c != "ADEP_mvt":
            d[c] = pd.to_datetime(d[c], utc=True).dt.as_unit(unit)
    return d


class ProvenanceAndPrecisionTest(unittest.TestCase):
    def test_h1_flags_exact_equality(self):
        out = NP.baue(frame())
        self.assertEqual(out.loc[0, "nmp_aobt_eq_eobt1"], 1.0)
        self.assertEqual(out.loc[1, "nmp_aobt_eq_eobt1"], 0.0)
        self.assertEqual(out.loc[0, "nmp_aobt_eq_iobt"], 0.0)

    def test_h2_seconds_field(self):
        out = NP.baue(frame())
        self.assertEqual(out.loc[0, "nmp_arvt1_sec"], 1.0)   # 17 s
        self.assertEqual(out.loc[0, "nmp_arvt3_sec"], 0.0)   # whole minute
        self.assertEqual(out.loc[0, "nmp_aobt3_sec"], 0.0)

    def test_missing_aobt_gives_nan_and_is_not_a_source(self):
        out = NP.baue(frame())
        self.assertTrue(np.isnan(out.loc[7, "nmp_aobt_eq_eobt1"]))
        self.assertTrue(np.isnan(out.loc[7, "nmp_aobt3_sec"]))
        # The fallback row at minute 25 must not be counted by the row at minute 30.
        self.assertEqual(out.loc[4, "nmp_plan_n_60"], 4)


class RegimeTest(unittest.TestCase):
    def test_share_uses_strictly_earlier_rows_only(self):
        out = NP.baue(frame())
        # minute 30 sees 0 (1), 10 (0), 20 (1), 20 (0): share 2/4, never itself
        self.assertEqual(out.loc[4, "nmp_plan_n_60"], 4)
        self.assertAlmostEqual(out.loc[4, "nmp_plan_share_60"], 0.5)

    def test_same_instant_rows_do_not_see_each_other(self):
        out = NP.baue(frame())
        # both minute-20 rows see only 0 and 10
        self.assertEqual(out.loc[2, "nmp_plan_n_60"], 2)
        self.assertEqual(out.loc[3, "nmp_plan_n_60"], 2)

    def test_window_slides(self):
        out = NP.baue(frame())
        # minute 70 sees [10, 70): 10 (0), 20 (1), 20 (0), 30 (0)
        self.assertEqual(out.loc[5, "nmp_plan_n_60"], 4)
        self.assertAlmostEqual(out.loc[5, "nmp_plan_share_60"], 0.25)
        # the day window also reaches minute 0
        self.assertAlmostEqual(out.loc[5, "nmp_plan_share_day"], 2 / 5)

    def test_first_row_has_no_history(self):
        out = NP.baue(frame())
        self.assertEqual(out.loc[0, "nmp_plan_n_60"], 0)
        self.assertTrue(np.isnan(out.loc[0, "nmp_plan_share_60"]))

    def test_other_airport_never_leaks(self):
        out = NP.baue(frame())
        self.assertEqual(out.loc[6, "nmp_plan_n_60"], 0)
        self.assertTrue(np.isnan(out.loc[6, "nmp_plan_share_day"]))

    def test_day_window_resets_at_utc_midnight(self):
        d = frame().iloc[:2].copy()
        d["MVT_TIME_UTC_mvt"] = [pd.Timestamp("2025-01-03 23:50", tz="UTC"),
                                 pd.Timestamp("2025-01-04 00:10", tz="UTC")]
        d["EOBT_1_flt"] = d["MVT_TIME_UTC_mvt"] - pd.Timedelta(minutes=20)
        d["AOBT_3_flt"] = d["EOBT_1_flt"]
        out = NP.baue(d)
        self.assertEqual(out.iloc[1]["nmp_plan_n_60"], 1)         # 23:50 is inside the hour
        self.assertTrue(np.isnan(out.iloc[1]["nmp_plan_share_day"]))  # but on the previous day


class InvarianceTest(unittest.TestCase):
    def test_microsecond_storage_gives_identical_values(self):
        pd.testing.assert_frame_equal(NP.baue(frame("us")), NP.baue(frame("ns")))

    def test_row_order_does_not_matter(self):
        base = NP.baue(frame())
        shuffled = frame().sample(frac=1.0, random_state=7)
        pd.testing.assert_frame_equal(NP.baue(shuffled).sort_index(), base)

    def test_absolute_id_shift_does_not_matter(self):
        a = frame().assign(MVT_ID_mvt=np.arange(8) + 1_000.0)
        b = frame().assign(MVT_ID_mvt=np.arange(8) + 9_999_999.0)
        pd.testing.assert_frame_equal(NP.baue(a), NP.baue(b))

    def test_label_columns_are_ignored(self):
        base = frame()
        noisy = base.assign(TAXITIME_SEC_mvt=-1.0, y=123456.0,
                            BLOCK_TIME_UTC_mvt=pd.Timestamp("2000-01-01", tz="UTC"))
        pd.testing.assert_frame_equal(NP.baue(base), NP.baue(noisy))

    def test_missing_required_column_is_an_error(self):
        with self.assertRaises(KeyError):
            NP.baue(frame().drop(columns="ARVT_3_flt"))

    def test_families_partition_the_features(self):
        flat = [c for cols in NP.FAMILIES.values() for c in cols]
        self.assertEqual(sorted(flat), sorted(NP.MERKMALE))
        self.assertEqual(len(flat), len(set(flat)))


if __name__ == "__main__":
    unittest.main()
