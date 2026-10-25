"""DESCRIPTIVE development-data error audit of v56's partial-track correction; no model is fitted here.

Data: v56's own development test dates (January 4, 13, 20, 28 and July 9, 11, 16, 27, 2025), where its three saved
real-seed models are out of sample. The predictions are the faithful rebuild (research_320/rebuild_v56_partial.py:
the three real models byte-identical to Codex's), i.e. exactly v56's correction: comparator + 0.5 x mean seed
correction, floor 300, on partial_ground rows.

Cohort, as v56's release policy treats it (research_310/RELEASE_RESULT.md, V55_BASE_ADDENDUM.md):
  the seven deployed airport/month cells only (EDDF, EGLL January; EHAM July; LEBL, LSZH both months; Codex owns
  EDDF July, EGLL July and EHAM January), partial_ground rows only, and without the rows the guarded Amsterdam
  detector estimates in EHAM July (v56 keeps the detector's value there). Munich, fallback and NM rows are outside
  the five-airport main-lane scope by construction on these 2025 dates.

It reports where v56's remaining error sits, by cell and by ordered stop/move episode pattern (research_320/
episodes.py, computed on the same points as the 23 summaries), next to the comparator's error in the same rows.
This audit informs one pre-registered experiment; its dates are never used to evaluate that experiment.
Run from the repository root:  python3 -u -B research_320/audit_v56_errors.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import extract_episodes as X  # noqa: E402
from research_320.episodes import EPISODE_FEATURES  # noqa: E402

TEST = ("2025-01-04", "2025-01-13", "2025-01-20", "2025-01-28", "2025-07-09", "2025-07-11", "2025-07-16", "2025-07-27")
DEPLOYED = {("EDDF", "01"), ("EGLL", "01"), ("EHAM", "07"), ("LEBL", "01"), ("LEBL", "07"), ("LSZH", "01"), ("LSZH", "07")}
REBUILD = HERE / ".cache/research_320/v56_rebuild"
OUT = HERE / "research_320/results/audit_v56_dev.json"
BINS = {
    "stop_count": ([-0.5, 0.5, 1.5, 2.5, 99], ["0", "1", "2", "3+"]),
    "stopped_after_stand_s": ([-1, 0.5, 300, 900, 1e9], ["0", "<5 min", "5-15 min", ">15 min"]),
    "last_stop_end_age_s": ([-1, 120, 300, 600, 1e9], ["<2 min", "2-5 min", "5-10 min", ">10 min"]),
    "last_stop_stand_m": ([-1, 80, 500, 1e9], ["at stand <=80 m", "80-500 m", ">500 m"]),
    "longest_stop_s": ([-1, 120, 600, 1800, 1e9], ["<2 min", "2-10 min", "10-30 min", ">30 min"]),
}


def detector_rows():
    """IDs the guarded Amsterdam detector estimates on the July test dates (v56 keeps them)."""
    from research_300 import tow_guard_test as T
    from research_290 import stand_detector as SD
    from research_300 import geometry_compare as GC
    import offblock_from_tracks as OB
    mapping, maps = OB.callsign_mapping(), (SD.stand_table("EHAM"), GC.xplane_table("EHAM")[0])
    ids = set()
    for date in [d for d in TEST if d[5:7] == "07"]:
        est = T.estimates(date, "EHAM", mapping, maps)
        ids |= set(est.loc[est.guarded.notna(), "MVT_ID_mvt"])
    return ids


def summary(frame, pred_col):
    e_v56, e_cmp = (frame[pred_col] - frame.y) ** 2, (frame.comparator - frame.y) ** 2
    return {"rows": int(len(frame)), "comparator_rmse": float(np.sqrt(e_cmp.mean())) if len(frame) else None,
            "v56_rmse": float(np.sqrt(e_v56.mean())) if len(frame) else None,
            "v56_gain_mse": float((e_cmp - e_v56).mean()) if len(frame) else None,
            "v56_sse_share": float(e_v56.sum()), "mean_residual_s": float((frame.y - frame[pred_col]).mean()) if len(frame) else None}


def main():
    pred = pd.read_parquet(REBUILD / "predictions.parquet")
    feats = X.extract(TEST, "dev_test")
    check = pred[["MVT_ID_mvt", "category"] + list(X.v56_modules().FEATURES)].merge(
        feats[["MVT_ID_mvt", "category"] + list(X.v56_modules().FEATURES)], on="MVT_ID_mvt", suffixes=("", "_x"),
        validate="one_to_one")
    parity = bool(len(check) == len(pred) and (check.category == check.category_x).all() and all(
        np.allclose(check[c].to_numpy(float), check[c + "_x"].to_numpy(float), equal_nan=True, rtol=0, atol=0)
        for c in X.v56_modules().FEATURES))
    if not parity:
        raise SystemExit("episode extraction does not reproduce v56's feature rows")
    d = pred.merge(feats[["MVT_ID_mvt"] + list(EPISODE_FEATURES)], on="MVT_ID_mvt", validate="one_to_one")
    d["month"] = d.date.str[5:7]
    in_cell = np.array([(a, m) in DEPLOYED for a, m in zip(d.airport, d.month)])
    detector = detector_rows()
    cohort = d[in_cell & d.category.eq("partial_ground") & ~d.MVT_ID_mvt.isin(detector)].copy()
    total_sse = float(((cohort.real - cohort.y) ** 2).sum())
    report = {"parity_with_v56_rows": parity, "detector_rows_excluded": int(d.MVT_ID_mvt.isin(detector).sum()),
              "cohort": summary(cohort, "real"), "cells": {}, "episodes": {}}
    for (a, m), g in cohort.groupby(["airport", "month"]):
        report["cells"][f"{a}-{m}"] = summary(g, "real")
    for rec in report["cells"].values():
        rec["v56_sse_share"] = rec["v56_sse_share"] / total_sse
    se = ((cohort.real - cohort.y) ** 2).to_numpy()
    report["concentration"] = {
        "top100_share": float(np.sort(se)[-100:].sum() / se.sum()),
        "worst_airport_day_share": float(cohort.assign(se=se).groupby(["airport", "date"]).se.sum().max() / se.sum())}
    for feature, (edges, labels) in BINS.items():
        col = pd.cut(cohort[feature], edges, labels=labels)
        rows = {}
        for label in labels + ["missing"]:
            g = cohort[cohort[feature].isna()] if label == "missing" else cohort[col.eq(label)]
            if len(g):
                rec = summary(g, "real")
                rec["v56_sse_share"] = rec["v56_sse_share"] / total_sse
                rec["row_share"] = len(g) / len(cohort)
                rows[label] = rec
        report["episodes"][feature] = rows
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    c = report["cohort"]
    print(f"cohort {c['rows']} rows, comparator {c['comparator_rmse']:.2f} -> v56 {c['v56_rmse']:.2f} s "
          f"(gain {c['v56_gain_mse']:.0f} MSE); detector rows excluded {report['detector_rows_excluded']}; "
          f"top-100 share {report['concentration']['top100_share']:.3f}")
    for cell, r in report["cells"].items():
        print(f"  {cell}: {r['rows']:5d} rows, {r['comparator_rmse']:.1f} -> {r['v56_rmse']:.1f} s, SSE share {r['v56_sse_share']:.3f}")
    for feature, rows in report["episodes"].items():
        print(" ", feature)
        for label, r in rows.items():
            print(f"     {label:>16s}: rows {r['rows']:5d} ({r['row_share']:.2f}) mean resid {r['mean_residual_s']:+7.1f} s  "
                  f"cmp {r['comparator_rmse']:6.1f} v56 {r['v56_rmse']:6.1f}  SSE share {r['v56_sse_share']:.3f}")


if __name__ == "__main__":
    main()
