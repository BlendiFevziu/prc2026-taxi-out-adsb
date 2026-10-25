"""Isolated validation of future same-stand arrival features.

The organiser explicitly permits using every provided movement row, including rows
after the departure being predicted.  Arrival values stay observable in the ranking
file: their block time and taxi-in time remain supplied; ARVT_3 is available where
reported (342,141 of 344,693 ranking arrivals in the September 2026 snapshot).  This harness asks a
strictly bounded question: does the *next* arrival at the same stand add information
on top of the current production feature set, which already contains the previous
arrival at that stand?

Learning and held-out month pairs are built independently.  The script writes neither
a model nor a submission.

Usage:
    python3 future_arrivals_cv.py 1 7
"""
import glob
import sys
import time

import neural_model  # Load torch before tree libraries (macOS OpenMP).
import lightgbm as lgb
import numpy as np
import pandas as pd

import features_geometry as MG
import main_model as H
from fid_gap_cv import align_categories, month_of_file


FUTURE_ARRIVAL_FEATURES = [
    "um_ver_naechste_ank", "um_taxi_naechste_ank",
    "um_versp_naechste_ank", "um_luecke_naechste_ank",
]

RAW_COLUMNS = [
    "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "ARVT_3_flt", "TAXITIME_SEC_mvt",
]


def seconds_after(a, b):
    """Return a-b in seconds, preserving missing timestamps as NaN."""
    return (pd.to_datetime(a, utc=True) - pd.to_datetime(b, utc=True)).dt.total_seconds()


def add_future_arrivals(paths, dep):
    """Attach the next known arrival at each departure's airport/stand.

    Only fields that remain supplied for ranking arrivals are read.  In particular,
    this never reads a departure target or a held-out training target.
    """
    raw = pd.concat(
        [pd.read_parquet(path, columns=RAW_COLUMNS) for path in paths],
        ignore_index=True,
    )
    phase = raw["PHASE_mvt"].astype(str).str.upper()
    arrivals = raw[phase.str.startswith("ARR")].copy()
    arrivals["stand"] = arrivals["ADES_mvt"].astype(str) + "|" + arrivals["STAND_mvt"].astype(str)
    arrivals["arrival_time"] = pd.to_datetime(arrivals["MVT_TIME_UTC_mvt"], utc=True)
    arrivals["arrival_block"] = pd.to_datetime(arrivals["BLOCK_TIME_UTC_mvt"], utc=True)
    arrivals["arrival_ver"] = seconds_after(arrivals["ARVT_3_flt"], arrivals["BLOCK_TIME_UTC_mvt"])
    arrivals["arrival_taxi"] = pd.to_numeric(arrivals["TAXITIME_SEC_mvt"], errors="coerce")
    arrivals["arrival_versp"] = seconds_after(arrivals["MVT_TIME_UTC_mvt"], arrivals["SCHED_TIME_UTC_mvt"])
    arrivals = arrivals.dropna(subset=["arrival_time"])

    out = dep.copy()
    out["_row"] = np.arange(len(out))
    out["stand"] = out["ADEP_mvt"].astype(str) + "|" + out["STAND_mvt"].astype(str)
    out["departure_time"] = pd.to_datetime(out["MVT_TIME_UTC_mvt"], utc=True)
    left = out[["_row", "stand", "departure_time"]].sort_values("departure_time")
    right = arrivals[[
        "stand", "arrival_time", "arrival_ver", "arrival_taxi", "arrival_versp",
    ]].sort_values("arrival_time")
    joined = pd.merge_asof(
        left,
        right,
        left_on="departure_time",
        right_on="arrival_time",
        by="stand",
        direction="forward",
        allow_exact_matches=False,
    ).sort_values("_row")

    out["um_ver_naechste_ank"] = joined["arrival_ver"].to_numpy()
    out["um_taxi_naechste_ank"] = joined["arrival_taxi"].to_numpy()
    out["um_versp_naechste_ank"] = joined["arrival_versp"].to_numpy()
    out["um_luecke_naechste_ank"] = (
        joined["arrival_time"] - joined["departure_time"]
    ).dt.total_seconds().to_numpy()
    return out.drop(columns=["_row", "stand", "departure_time"])


def score(train, valid, features):
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
    return float(np.mean((prediction - valid.y.to_numpy()) ** 2)), model.best_iteration_


def main(months):
    if len(months) != 2 or any(month < 1 or month > 12 for month in months):
        raise SystemExit("zwei Monatsnummern zwischen 1 und 12 angeben, z.B. 1 7")
    paths = sorted(glob.glob("data/training_2025-*.parquet"))
    learning = [path for path in paths if month_of_file(path) not in months]
    held_out = [path for path in paths if month_of_file(path) in months]
    print(f"Lernen: {[month_of_file(path) for path in learning]} | "
          f"Halten: {[month_of_file(path) for path in held_out]}", flush=True)
    train = add_future_arrivals(learning, H.baue(learning, MG.alle()))
    valid = add_future_arrivals(held_out, H.baue(held_out, MG.alle()))
    align_categories(train, valid)
    train = train[train.hat_aobt.eq(1)].copy()
    valid = valid[valid.hat_aobt.eq(1)].copy()
    train["rest"] = train.y - train.ersatz_s
    valid["rest"] = valid.y - valid.ersatz_s
    started = time.time()
    baseline, baseline_round = score(train, valid, H.MERKMALE_HAUPT)
    candidate, candidate_round = score(train, valid, H.MERKMALE_HAUPT + FUTURE_ARRIVAL_FEATURES)
    print({
        "months": tuple(months),
        "baseline_rmse": round(baseline ** 0.5, 6),
        "candidate_rmse": round(candidate ** 0.5, 6),
        "rmse_gain": round(baseline ** 0.5 - candidate ** 0.5, 6),
        "mse_delta": round(baseline - candidate, 3),
        "baseline_round": baseline_round, "candidate_round": candidate_round,
        "seconds": round(time.time() - started, 1),
    })


if __name__ == "__main__":
    main([int(value) for value in sys.argv[1:]])
