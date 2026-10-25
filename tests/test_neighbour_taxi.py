"""Checks for research_240/features_neighbour_taxi.py. No data files, no model fits."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_240 import features_neighbour_taxi as NT  # noqa: E402


def frame(unit="ns", extra_linked_minute=None):
    t0 = pd.Timestamp("2025-01-03 06:00", tz="UTC")
    rows = [("EHAM", 0, 600.0, 1), ("EHAM", 10, 700.0, 1), ("EHAM", 20, 800.0, 1),
            ("EHAM", 30, 900.0, 1), ("EHAM", 15, np.nan, 0),
            ("EHAM", 200, 5000.0, 1),   # outside every window of the early rows
            ("LTFM", 15, 99999.0, 1)]   # other airport, must never leak
    if extra_linked_minute is not None:
        rows.append(("EHAM", extra_linked_minute, 850.0, 1))
    d = pd.DataFrame(rows, columns=["ADEP_mvt", "minutes", "ersatz_s", "hat_aobt"])
    d["MVT_TIME_UTC_mvt"] = (t0 + pd.to_timedelta(d.minutes, unit="m")).dt.as_unit(unit)
    return d.drop(columns="minutes")


class NeighbourTaxiTest(unittest.TestCase):
    def test_unlinked_row_takes_median_of_linked_neighbours(self):
        out = NT.baue(frame())
        self.assertEqual(out.loc[4, "nt_med30"], 750.0)
        self.assertEqual(out.loc[4, "nt_n60"], 4)
        self.assertEqual(out.loc[4, "nt_n180"], 4)

    def test_linked_row_excludes_itself(self):
        out = NT.baue(frame())
        self.assertEqual(out.loc[1, "nt_med30"], 800.0)
        self.assertEqual(out.loc[1, "nt_n60"], 3)

    def test_other_airport_never_leaks(self):
        out = NT.baue(frame())
        self.assertTrue(np.isnan(out.loc[6, "nt_med180"]))
        self.assertEqual(out.loc[6, "nt_n180"], 0)
        self.assertTrue(np.isnan(out.loc[5, "nt_med60"]))

    def test_anchor_requires_five_neighbours(self):
        self.assertTrue(np.isnan(NT.baue(frame()).loc[4, "nt_anchor"]))
        out = NT.baue(frame(extra_linked_minute=25))
        self.assertEqual(out.loc[4, "nt_n60"], 5)
        self.assertEqual(out.loc[4, "nt_anchor"], 800.0)

    def test_microsecond_storage_gives_identical_values(self):
        pd.testing.assert_frame_equal(NT.baue(frame("us")), NT.baue(frame("ns")))

    def test_label_columns_are_ignored(self):
        base = frame()
        noisy = base.assign(TAXITIME_SEC_mvt=-1.0, y=123456.0,
                            BLOCK_TIME_UTC_mvt=pd.Timestamp("2000-01-01", tz="UTC"))
        pd.testing.assert_frame_equal(NT.baue(base), NT.baue(noisy))

    def test_query_mask_only_fills_requested_rows(self):
        d = frame()
        mask = d.hat_aobt.eq(0).to_numpy()
        part, full = NT.baue(d, query=mask), NT.baue(d)
        pd.testing.assert_series_equal(part.loc[4], full.loc[4])
        self.assertTrue(part.drop(index=4).isna().all().all())


if __name__ == "__main__":
    unittest.main()
