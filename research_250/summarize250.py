"""Aggregate paired component predictions, preserving the official-score distinction."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

def summarize(frames, baseline, candidate):
    if not frames:
        return None
    pooled = pd.concat(frames, ignore_index=True)
    assert pooled.MVT_ID_mvt.is_unique, "A movement must appear in exactly one outer holdout"
    e0 = (pooled[baseline]-pooled.y)**2
    e1 = (pooled[candidate]-pooled.y)**2
    pooled["baseline_sse"], pooled["candidate_sse"] = e0, e1
    pooled["day"] = pd.to_datetime(pooled.MVT_TIME_UTC_mvt, utc=True).dt.floor("D")
    folds = []
    for fold, group in pooled.groupby("fold", sort=True):
        mse0, mse1 = group.baseline_sse.mean(), group.candidate_sse.mean()
        folds.append({"fold": fold, "rows": len(group),
                      "baseline_rmse": float(np.sqrt(mse0)),
                      "candidate_rmse": float(np.sqrt(mse1)),
                      "rmse_gain": float(np.sqrt(mse0)-np.sqrt(mse1)),
                      "mse_gain": float(mse0-mse1)})
    airports = []
    for airport, group in pooled.groupby("ADEP_mvt", observed=True):
        airports.append({"airport": str(airport), "rows": len(group),
                         "mse_gain": float(group.baseline_sse.mean()-group.candidate_sse.mean())})
    # Descriptive uncertainty within the observed year; this does not model 2026
    # distribution shift. Resample whole airport-days, not individual flights.
    groups = pooled.groupby(["ADEP_mvt", "day"], observed=True).agg(
        n=("y", "size"), b=("baseline_sse", "sum"), c=("candidate_sse", "sum"))
    a = groups.to_numpy(float)
    rng = np.random.default_rng(250)
    draws = []
    for _ in range(2000):
        sample = a[rng.integers(0, len(a), len(a))].sum(axis=0)
        draws.append((sample[1]-sample[2])/sample[0])
    return {"rows": len(pooled), "baseline_rmse": float(np.sqrt(e0.mean())),
            "candidate_rmse": float(np.sqrt(e1.mean())),
            "mse_gain": float(e0.mean()-e1.mean()),
            "positive_folds": sum(f["mse_gain"] > 0 for f in folds),
            "folds": folds, "airports": airports,
            "airport_day_bootstrap_mse_gain_95pct": np.quantile(draws, [.025, .975]).tolist(),
            "bootstrap_note": "descriptive resampling of observed airport-days; excludes future-year/model-selection uncertainty",
            "baseline_top5_sse_share": float(e0.nlargest(5).sum()/e0.sum()),
            "candidate_top5_sse_share": float(e1.nlargest(5).sum()/e1.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--main-results", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fallback, main = [], []
    for first in range(1, 7):
        pair = f"{first}_{first+6}"
        p = args.results / f"predictions_{pair}.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            f["fold"] = pair
            fallback.append(f[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "y", "fold",
                               "v44_fallback", "correct_time_added"]])
        if first == 1:
            p = args.results / f"main_predictions_{pair}.parquet"
        else:
            p = (args.main_results or args.results) / f"time_cv_predictions_{pair}.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            f["fold"] = pair
            main.append(f[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "y", "fold",
                           "v44_main", "units_replaced"]])
    result = {"official_best_rmse": 272.334, "target_rmse": 250.,
              "official_score_unchanged": True,
              "required_overall_mse_reduction": 272.334**2-250**2,
              "validation": "paired local component predictions; no full-ensemble six-pair CV and no new submission",
              "fallback": summarize(fallback, "v44_fallback", "correct_time_added"),
              "main_single_lightgbm": summarize(main, "v44_main", "units_replaced")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
