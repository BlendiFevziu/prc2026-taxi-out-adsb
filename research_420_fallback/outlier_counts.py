"""Fallback lane: do the submission's outliers look like the training outliers? Counts and sizes, per airport. No fit.

Prompted by a leading contestant's public tip ("force the outliers in the submission to look as much like the training
set outliers as possible") and his answer that most of his lead is in the rows without AOBT_3.

2025: the shipped fallback's six-pair out-of-sample predictions (`prod_anchor`, .cache/research_240/
fallback_anchor_production; the v50 Rome gate on 8 rows is not in them) against the labels. 2026: v63's values on the
5,290 scored rows without AOBT_3 and the times known at rank time; no 2026 label exists or is read.

  outlier            label above 2 h
  predicted outlier  prediction above 2 h
  late               versp = MVT - SCHED above 2 h (known at rank time; the precondition for a large copy value)
  copy               |BLOCK - SCHED| <= 60 s (2025 only), so y equals versp to within a minute
  commit             prediction within 10 % of versp on a late row (the model has taken the copy value)
Run from the repository root:  python3 -u -B research_420_fallback/outlier_counts.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / ".cache/research_240/fallback_anchor_production"
PAIRS = ("1_7", "2_8", "3_9", "4_10", "5_11", "6_12")
V63 = ROOT / ".cache/research_360/v63/candidate_v63.parquet"
OUT = ROOT / "research_420_fallback/results/outlier_counts.json"
ID, T, H2 = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 7200.0


def secs(a, b):
    return (pd.to_datetime(a, utc=True) - pd.to_datetime(b, utc=True)).dt.total_seconds().to_numpy()


def frame_2025():
    pred = pd.concat([pd.read_parquet(PREDS / f"predictions_{p}.parquet", columns=[ID, "y", "prod_anchor", "ADEP_mvt"]) for p in PAIRS])
    raw = pd.concat(pd.read_parquet(p, columns=[ID, "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"])
                    for p in sorted((ROOT / "data").glob("training_2025-*.parquet")))
    d = pred.merge(raw, on=ID, how="left", validate="one_to_one")
    d["versp"] = secs(d.MVT_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)
    d["is_copy"] = np.abs(secs(d.BLOCK_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)) <= 60
    d["pred"] = d.prod_anchor.astype(float)
    return d.rename(columns={"ADEP_mvt": "airport"})


def frame_2026():
    rk = pd.read_parquet(ROOT / "data/ranking.parquet", columns=[ID, "PHASE_mvt", T, "AOBT_3_flt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    d = rk[rk.PHASE_mvt.eq("DEP") & rk[T].isna() & rk.AOBT_3_flt.isna()].copy()
    v = pd.read_parquet(V63).set_index(ID)[T]
    d["pred"] = v.loc[d[ID]].to_numpy(float)
    d["versp"] = secs(d.MVT_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)
    return d.rename(columns={"ADEP_mvt": "airport"})


def describe(d, labelled):
    late = d.versp > H2
    commit = late & (np.abs(d.pred - d.versp) <= 0.1 * d.versp)
    out = {"rows": int(len(d)), "late_share": float(late.mean()), "predicted_outlier_share": float((d.pred > H2).mean()),
           "commit_share": float(commit.mean()), "commit_share_of_late": float(commit[late].mean()) if late.any() else None,
           "predicted_outlier_size_q50_q90": [float(x) for x in np.percentile(d.pred[d.pred > H2], [50, 90])] if (d.pred > H2).any() else None}
    if labelled:
        outl = d.y > H2
        out.update({"outlier_share": float(outl.mean()), "copy_share": float(d.is_copy.mean()),
                    "late_copy_share_of_late": float(d.is_copy[late].mean()) if late.any() else None,
                    "outlier_size_q50_q90": [float(x) for x in np.percentile(d.y[outl], [50, 90])] if outl.any() else None,
                    "outliers_caught": float((d.pred[outl] > H2).mean()) if outl.any() else None,
                    "predicted_outliers_true": float((d.y[d.pred > H2] > H2).mean()) if (d.pred > H2).any() else None,
                    "copy_rows_committed": float(commit[d.is_copy & late].mean()) if (d.is_copy & late).any() else None})
    return out


def main():
    a, b = frame_2025(), frame_2026()
    rep = {"pooled": {"2025": describe(a, True), "2026": describe(b, False)}, "airports": {}}
    for ap in sorted(set(a.airport) | set(b.airport)):
        rep["airports"][ap] = {"2025": describe(a[a.airport.eq(ap)], True), "2026": describe(b[b.airport.eq(ap)], False)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2) + "\n")
    f = lambda x: "  -  " if x is None else f"{100 * x:5.1f}"
    print(f"{'':6s}| 2025 rows  late% outl% pred>2h% caught% predTrue% copy|late% commit|late% copyCommitted% | 2026 rows  late% pred>2h% commit|late%")
    for ap, r in [("ALL", rep["pooled"])] + list(rep["airports"].items()):
        x, y = r["2025"], r["2026"]
        print(f"{ap:6s}| {x['rows']:9,d} {f(x['late_share'])} {f(x['outlier_share'])} {f(x['predicted_outlier_share'])}   {f(x['outliers_caught'])}   "
              f"{f(x['predicted_outliers_true'])}     {f(x['late_copy_share_of_late'])}       {f(x['commit_share_of_late'])}        {f(x['copy_rows_committed'])}     "
              f"| {y['rows']:9,d} {f(y['late_share'])} {f(y['predicted_outlier_share'])}   {f(y['commit_share_of_late'])}")
    for k in ("2025", "2026"):
        print(k, "predicted outlier size q50/q90:", rep["pooled"][k]["predicted_outlier_size_q50_q90"],
              "| true outlier size q50/q90:" if k == "2025" else "", rep["pooled"][k].get("outlier_size_q50_q90", ""))


if __name__ == "__main__":
    main()
