"""Hold-out test for a frozen interaction extension of the FLIGHT_ID sequence layer.

This remains an experiment: it does not alter ``main_model.py`` or write a model.
It builds learning and held-out features independently and uses the strict monthly file
selector from ``fid_gap_cv.py``.
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


INTERACTIONS = [
    "fid_beschleunigung", "fid_rate_asym", "fid_z201",
    "fid_rel_mvt_vor", "fid_rel_mvt_nach", "fid_zeit_summe",
]


def add_interactions(d):
    out = d.copy()
    out["_row"] = np.arange(len(out))
    out["_mvtid"] = pd.to_numeric(out["MVT_ID_mvt"], errors="coerce")
    out = out.sort_values(["ADEP_mvt", "MVT_TIME_UTC_mvt"], kind="stable")
    group = out.groupby("ADEP_mvt", observed=True)
    mvt_vor = group["_mvtid"].diff().abs()
    mvt_nach = (group["_mvtid"].shift(-1) - out["_mvtid"]).abs()
    out["fid_beschleunigung"] = out.fid_luecke_nach - out.fid_luecke_vor
    out["fid_rate_asym"] = out.fid_rate_nach - out.fid_rate_vor
    out["fid_z201"] = out.fid_abw2 / out.fid_span201.replace(0, np.nan)
    out["fid_rel_mvt_vor"] = out.fid_luecke_vor / (mvt_vor + 1)
    out["fid_rel_mvt_nach"] = out.fid_luecke_nach / (mvt_nach + 1)
    out["fid_zeit_summe"] = out.fid_zeit_vor + out.fid_zeit_nach
    for column in INTERACTIONS:
        lower, upper = out[column].quantile([0.001, 0.999])
        out[column] = out[column].clip(lower, upper)
    return out.sort_values("_row", kind="stable").drop(columns=["_row", "_mvtid"])


def score(train, valid, features):
    model = lgb.LGBMRegressor(
        objective="l2", n_estimators=1400, learning_rate=0.03, num_leaves=255,
        min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        verbose=-1, n_jobs=-1, random_state=177,
    )
    model.fit(train[features], train.rest, eval_set=[(valid[features], valid.rest)],
              eval_metric="rmse", callbacks=[lgb.early_stopping(80, verbose=False)])
    prediction = valid.ersatz_s.to_numpy() + model.predict(valid[features])
    return float(np.mean((prediction - valid.y.to_numpy()) ** 2)), model.best_iteration_


def main(months):
    if len(months) != 2:
        raise SystemExit("zwei Monatsnummern angeben, z.B. 1 7")
    paths = sorted(glob.glob("data/training_2025-*.parquet"))
    learning = [path for path in paths if month_of_file(path) not in months]
    held_out = [path for path in paths if month_of_file(path) in months]
    print(f"Lernen: {[month_of_file(path) for path in learning]} | "
          f"Halten: {[month_of_file(path) for path in held_out]}", flush=True)
    train = add_interactions(H.baue(learning, MG.alle()))
    valid = add_interactions(H.baue(held_out, MG.alle()))
    align_categories(train, valid)
    train = train[train.hat_aobt.eq(1)].copy()
    valid = valid[valid.hat_aobt.eq(1)].copy()
    train["rest"] = train.y - train.ersatz_s
    valid["rest"] = valid.y - valid.ersatz_s
    started = time.time()
    baseline, baseline_round = score(train, valid, H.MERKMALE_HAUPT)
    candidate, candidate_round = score(train, valid, H.MERKMALE_HAUPT + INTERACTIONS)
    print({
        "months": tuple(months), "baseline_rmse": round(baseline ** 0.5, 6),
        "candidate_rmse": round(candidate ** 0.5, 6),
        "rmse_gain": round(baseline ** 0.5 - candidate ** 0.5, 6),
        "mse_delta": round(baseline - candidate, 3),
        "baseline_round": baseline_round, "candidate_round": candidate_round,
        "seconds": round(time.time() - started, 1),
    })


if __name__ == "__main__":
    main([int(value) for value in sys.argv[1:]])
