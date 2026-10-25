"""Leave-one-month-out test of research_500_midnight/PREREG_crossing.md. No fit, no upload.
Run from the repository root:  python3 -u -B research_500_midnight/crossing_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule as R  # noqa: E402
import audit_crossing as A  # noqa: E402

N, REF = 344841, 255.32
OUT = R.ROOT / "research_500_midnight/results/crossing_test.json"
V69 = R.ROOT / ".cache/research_500/v69/candidate_v69.parquet"


def in_cell(delay, value, crossing):
    return (np.asarray(crossing) >= 1) & (np.asarray(delay) > 3600) & (np.asarray(delay) <= 8 * 3600) & (np.asarray(value) < 0.5 * np.asarray(delay))


def main():
    d = A.rows_2025()
    d = d[in_cell(d.delay, d.shipped, d.crossing)].reset_index(drop=True)
    d["is_copy"] = d.kind.eq("copy")
    y, ship = d[R.T].to_numpy(float), d.shipped.to_numpy(float)
    new = np.empty(len(d))
    p_by_month = {}
    for m in sorted(d.month.unique()):
        p = float(d.loc[d.month.ne(m), "is_copy"].mean())
        p_by_month[int(m)] = p
        i = d.month.eq(m).to_numpy()
        new[i] = ship[i] + p * (d.delay.to_numpy(float)[i] - ship[i])
    gain = (ship - y) ** 2 - (new - y) ** 2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-5:]] = False
    by_month = {int(m): float(gain[d.month.eq(m).to_numpy()].sum()) for m in sorted(d.month.unique())}
    p_all = float(d.is_copy.mean())
    rk = R.annotate(pd.read_parquet(R.ROOT / "data/ranking.parquet", columns=R.COLS))
    rk = rk[rk.PHASE_mvt.eq("DEP") & rk.ADEP_mvt.eq("LIRF") & rk.AOBT_3_flt.isna()].reset_index(drop=True)
    rk["v69"] = rk[R.ID].map(pd.read_parquet(V69).set_index(R.ID)[R.T]).astype(float)
    rows26 = int(in_cell(rk.delay, rk.v69, rk.crossing).sum())
    mse = rows26 * float(gain.mean()) / N
    report = {"prereg": "research_500_midnight/PREREG_crossing.md", "rows_2025": int(len(d)),
              "kinds_2025": {k: int(v) for k, v in d.kind.value_counts().items()}, "p_by_month": p_by_month, "p_all_2025": p_all,
              "gain_sse": float(gain.sum()), "gain_per_row": float(gain.mean()), "without_best5": float(gain[keep].sum()),
              "by_month": by_month, "rows_2026": rows26,
              "planning_s": float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))}
    report["rules"] = {"pooled_positive": report["gain_sse"] > 0, "without_best5_positive": report["without_best5"] > 0,
                       "two_thirds_of_months": sum(v > 0 for v in by_month.values()) >= 2 * len(by_month) / 3,
                       "planning_at_least_0_1_s": report["planning_s"] >= 0.1}
    report["verdict"] = "pass" if all(report["rules"].values()) else "fail"
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
