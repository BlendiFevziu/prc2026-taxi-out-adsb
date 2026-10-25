"""Production-faithful validation for the local FLIGHT_ID neighbour-gap family.

The gap family is now integrated into ``main_model.py``.  This harness removes it
from the baseline to preserve the original ablation.  It never writes a model or a
submission.  Feature references are
built independently for the learning months and held-out months, matching the final
training/ranking boundary.

Usage:
    python3 fid_gap_cv.py 1 7
"""
import glob
import re
import sys
import time
from pathlib import Path

import neural_model  # Load torch before tree libraries (macOS OpenMP).
import lightgbm as lgb
import numpy as np
import pandas as pd

import features_geometry as MG
import main_model as H


def month_of_file(path):
    """Read only the *start* month, never the end date, from a monthly filename."""
    match = re.fullmatch(r"training_2025-(\d{2})-01_.*\.parquet", Path(path).name)
    if match is None:
        raise ValueError(f"unerwarteter Trainingsdateiname: {path}")
    return int(match.group(1))


def add_neighbour_features(d):
    """Add the pre-specified local-gap family without targets or external inputs."""
    out = d.copy()
    out["_row"] = np.arange(len(out))
    out["_fid"] = pd.to_numeric(out["FLIGHT_ID_mvt"], errors="coerce")
    out["_time"] = pd.to_datetime(out["MVT_TIME_UTC_mvt"]).astype("int64") / 1e9
    out = out.sort_values(["ADEP_mvt", "MVT_TIME_UTC_mvt"], kind="stable")
    group = out.groupby("ADEP_mvt", observed=True)
    out["fid_luecke_vor"] = group["_fid"].diff()
    out["fid_luecke_nach"] = group["_fid"].shift(-1) - out["_fid"]
    out["fid_zeit_vor"] = group["_time"].diff()
    out["fid_zeit_nach"] = group["_time"].shift(-1) - out["_time"]
    out["fid_rate_vor"] = out.fid_luecke_vor / out.fid_zeit_vor.clip(lower=1)
    out["fid_rate_nach"] = out.fid_luecke_nach / out.fid_zeit_nach.clip(lower=1)
    out["fid_span201"] = group["_fid"].transform(
        lambda values: values.rolling(201, center=True, min_periods=20).max()
        - values.rolling(201, center=True, min_periods=20).min()
    )
    # This is a feature-definition robustness bound, not target-driven tuning.  It is
    # recomputed from the available movement table at both train and scoring time.
    for column in NEIGHBOUR_FEATURES:
        low, high = out[column].quantile([0.001, 0.999])
        out[column] = out[column].clip(low, high)
    return out.sort_values("_row", kind="stable").drop(columns=["_row", "_fid", "_time"])


NEIGHBOUR_FEATURES = [
    "fid_luecke_vor", "fid_luecke_nach", "fid_zeit_vor", "fid_zeit_nach",
    "fid_rate_vor", "fid_rate_nach", "fid_span201",
]


BASELINE_FEATURES = [f for f in H.MERKMALE_HAUPT if f not in NEIGHBOUR_FEATURES]
CANDIDATE_FEATURES = BASELINE_FEATURES + NEIGHBOUR_FEATURES


def align_categories(train, valid):
    """Give LightGBM identical categorical dictionaries on both sides of the split."""
    for column in H.KAT:
        categories = pd.unique(pd.concat(
            [train[column].astype(str), valid[column].astype(str)], ignore_index=True
        ))
        train[column] = pd.Categorical(train[column].astype(str), categories=categories)
        valid[column] = pd.Categorical(valid[column].astype(str), categories=categories)


def score(train, valid, features):
    assert len(features) == len(set(features)), "duplicate feature names"
    model = lgb.LGBMRegressor(
        objective="l2", n_estimators=1400, learning_rate=0.03, num_leaves=255,
        min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        verbose=-1, n_jobs=-1, random_state=177,
    )
    model.fit(
        train[features], train.rest,
        eval_set=[(valid[features], valid.rest)], eval_metric="rmse",
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    prediction = valid.ersatz_s.to_numpy() + model.predict(valid[features])
    mse = float(np.mean((prediction - valid.y.to_numpy()) ** 2))
    return mse, model.best_iteration_


def main(months):
    if len(months) != 2 or any(month < 1 or month > 12 for month in months):
        raise SystemExit("zwei Monatsnummern zwischen 1 und 12 angeben, z.B. 1 7")
    months = tuple(months)
    paths = sorted(glob.glob("data/training_2025-*.parquet"))
    learning_paths = [path for path in paths if month_of_file(path) not in months]
    held_out_paths = [path for path in paths if month_of_file(path) in months]
    print(f"Lernen: {[month_of_file(path) for path in learning_paths]} | "
          f"Halten: {[month_of_file(path) for path in held_out_paths]}", flush=True)
    train = add_neighbour_features(H.baue(learning_paths, MG.alle()))
    valid = add_neighbour_features(H.baue(held_out_paths, MG.alle()))
    align_categories(train, valid)
    train = train[train.hat_aobt.eq(1)].copy()
    valid = valid[valid.hat_aobt.eq(1)].copy()
    train["rest"] = train.y - train.ersatz_s
    valid["rest"] = valid.y - valid.ersatz_s
    start = time.time()
    baseline_mse, baseline_round = score(train, valid, BASELINE_FEATURES)
    candidate_mse, candidate_round = score(train, valid, CANDIDATE_FEATURES)
    result = {
        "months": months,
        "rows_train": len(train), "rows_valid": len(valid),
        "baseline_rmse": round(baseline_mse ** 0.5, 6),
        "candidate_rmse": round(candidate_mse ** 0.5, 6),
        "rmse_gain": round(baseline_mse ** 0.5 - candidate_mse ** 0.5, 6),
        "mse_delta": round(baseline_mse - candidate_mse, 3),
        "baseline_round": baseline_round, "candidate_round": candidate_round,
        "seconds": round(time.time() - start, 1),
    }
    print(result)


if __name__ == "__main__":
    main([int(value) for value in sys.argv[1:]])
