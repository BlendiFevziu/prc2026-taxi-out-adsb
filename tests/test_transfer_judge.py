from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_290 import transfer_2025 as T  # noqa: E402

BLOCKS = (("d1", "d2"), ("d3", "d4"))


def frame(n_per_date, det_err, model_err, seed=0):
    """Covered rows: the detector's taxi-time error and the comparator's, per row, on four dates."""
    rng = np.random.default_rng(seed)
    rows, model = [], []
    for i, date in enumerate(["d1", "d2", "d3", "d4"]):
        for k in range(n_per_date[i]):
            rid = f"{date}-{k}"
            y = 600.0 + rng.normal(0, 60)
            block, mvt = 1_000_000.0, 1_000_000.0 + y
            det = y + det_err[i] * rng.standard_normal()      # detector taxi time = mvt - estimate
            rows.append({"date": date, "MVT_ID_mvt": rid, "lane": "main", "block_epoch": block, "mvt_epoch": mvt,
                         "estimate": mvt - det})
            model.append({"MVT_ID_mvt": rid, "y": y, "ersatz_s": y + 300.0,
                          "baseline": y + model_err[i] * rng.standard_normal()})
    return pd.DataFrame(rows), pd.DataFrame(model)


class JudgeTest(unittest.TestCase):
    def test_a_thin_block_gives_no_verdict(self):
        ok, model = frame([30, 30, 5, 5], [30] * 4, [200] * 4)
        report, _ = T.judge(ok, model, BLOCKS)
        self.assertEqual(report["verdict"], "insufficient")
        self.assertEqual([b["rows"] for b in report["per_block"]], [60, 10])

    def test_clear_win_passes_all_four(self):
        ok, model = frame([40] * 4, [30] * 4, [200] * 4)
        report, gain = T.judge(ok, model, BLOCKS)
        self.assertEqual(report["verdict"], "pass")
        self.assertTrue(all(report["checks"].values()))
        self.assertGreater(float(np.mean(gain)), 0)

    def test_losing_one_block_fails_v2(self):
        ok, model = frame([40] * 4, [30, 30, 400, 400], [200] * 4)
        report, _ = T.judge(ok, model, BLOCKS)
        self.assertFalse(report["checks"]["V2"])
        self.assertEqual(report["verdict"], "fail")

    def test_label_check_is_exact(self):
        ok, model = frame([25] * 4, [30] * 4, [200] * 4)
        report, _ = T.judge(ok, model, BLOCKS)
        self.assertLess(report["label_check_max_diff_s"], 1e-6)

    def test_missing_comparator_row_stops(self):
        ok, model = frame([25] * 4, [30] * 4, [200] * 4)
        with self.assertRaises(SystemExit):
            T.judge(ok, model.iloc[1:], BLOCKS)


if __name__ == "__main__":
    unittest.main()
