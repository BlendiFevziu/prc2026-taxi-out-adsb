"""Fallback lane: where the SHIPPED model's own out-of-sample error sits, and how much copy headroom it leaves.

Codex's correction (STATUS.md, 23 September): the old "92.88 % of fallback error in 481 rows" was squared error about
each airport's MEDIAN, not the shipped model's residuals. This measures the residuals themselves, using the six-pair
out-of-sample predictions of the production anchored fallback (`prod_anchor`,
.cache/research_240/fallback_anchor_production/predictions_{pair}.parquet; v46 recipe; v50 adds a Rome gate on 8 rows).

Classes, from training labels (description of 2025 only; nothing here is fitted):
  copy       |BLOCK - SCHED| <= 60 s: the airport wrote the scheduled time into the block field, so y = MVT - BLOCK is
             within a minute of versp = MVT - SCHED, which is known at rank time
  extreme    not copy, y above 2 h
  normal     everything else
Oracle: the shipped prediction replaced by versp on copy rows only (the value a perfect copy classifier would add).
Official scale: fallback rows are 5,290 of 344,841 scored rows; v54 = 264.2328.

Prior work this replicates (do not re-run it): research_260/fallback_regime_census.py (same 1,435 copy rows, 15.4 %),
research_260/day_shift_regime_screen.py and fallback_copy_gate_screen.py (both rejected), research_280/day_shift_candidates.py
and fallback_gate_calibration.py (rejected). STATUS.md, "The fallback tail is closed for now" (16 September).

Run from the repository root:  python3 -u -B research_300/fallback_headroom.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / ".cache/research_240/fallback_anchor_production"
PAIRS = ("1_7", "2_8", "3_9", "4_10", "5_11", "6_12")
OFFICIAL, N_SCORED, N_FALLBACK = 264.2328, 344841, 5290
OUT = ROOT / "research_300/results/fallback_headroom.json"


def official_gain(delta_sse_per_fallback_row):
    delta = delta_sse_per_fallback_row * N_FALLBACK / N_SCORED
    return float(OFFICIAL - np.sqrt(max(OFFICIAL ** 2 - delta, 0.0)))


def main():
    pred = pd.concat([pd.read_parquet(PREDS / f"predictions_{p}.parquet",
                                      columns=["MVT_ID_mvt", "y", "prod_anchor", "ADEP_mvt", "fold"]) for p in PAIRS])
    raw = pd.concat(pd.read_parquet(p, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"])
                    for p in sorted((ROOT / "data").glob("training_2025-*.parquet")))
    d = pred.merge(raw, on="MVT_ID_mvt", how="left")
    t = {c: pd.to_datetime(d[c], utc=True) for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt")}
    d["versp"] = (t["MVT_TIME_UTC_mvt"] - t["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    d["block_minus_sched"] = (t["BLOCK_TIME_UTC_mvt"] - t["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    d["cls"] = np.where(d.block_minus_sched.abs() <= 60, "copy", np.where(d.y > 7200, "extreme", "normal"))
    e = (d.prod_anchor - d.y) ** 2
    med = d.groupby(["fold", "ADEP_mvt"]).y.transform("median")
    e_med = (med - d.y) ** 2
    n = len(d)
    rep = {"fallback_rows_2025_out_of_sample": int(n), "shipped_rmse": float(np.sqrt(e.mean())),
           "airport_median_rmse": float(np.sqrt(e_med.mean())), "classes": {}}
    for c, g in d.groupby("cls"):
        k = d.cls.eq(c)
        rep["classes"][c] = {"rows": int(k.sum()), "share_of_shipped_sse": float(e[k].sum() / e.sum()),
                             "share_of_sse_about_median": float(e_med[k].sum() / e_med.sum()),
                             "shipped_rmse": float(np.sqrt(e[k].mean()))}
    copy = d.cls.eq("copy").to_numpy()
    big = copy & (d.versp.to_numpy() > 7200)
    oracle = np.where(copy, d.versp.to_numpy(), d.prod_anchor.to_numpy())
    e_or = (oracle - d.y) ** 2
    rep["copy_rows_with_versp_over_2h"] = {"rows": int(big.sum()), "share_of_shipped_sse": float(e[big].sum() / e.sum()),
                                          "airports": d.loc[big, "ADEP_mvt"].value_counts().to_dict()}
    rep["oracle_perfect_copy_classifier"] = {
        "rmse": float(np.sqrt(e_or.mean())), "sse_saved_share": float(1 - e_or.sum() / e.sum()),
        "official_scale_gain_s": official_gain(float(e.mean() - e_or.mean()))}
    top = np.sort(e.to_numpy())[::-1]
    rep["largest_rows_share_of_shipped_sse"] = {str(k): float(top[:k].sum() / e.sum()) for k in (1, 3, 10, 84)}
    rep["airports"] = {}
    for a, g in d.groupby("ADEP_mvt"):
        k = d.ADEP_mvt.eq(a).to_numpy()
        rep["airports"][a] = {"rows": int(k.sum()), "share_of_shipped_sse": float(e[k].sum() / e.sum()),
                              "copy_rows": int((copy & k).sum()), "copy_share_of_shipped_sse": float(e[copy & k].sum() / e.sum()),
                              "extreme_share_of_shipped_sse": float(e[k & d.cls.eq("extreme").to_numpy()].sum() / e.sum())}
    OUT.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
