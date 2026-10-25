"""Strict hold-out screen for source-agreement features.

The target is a discrepancy between movement and Network-Manager records, so this tests
whether their categorical agreement adds information beyond the validated event-order
layer.  It is experimental and never writes a model or submission.
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
from fid_gap_cv import month_of_file


FEATURES = [
    "src_adep_gleich", "src_ades_flt_gleich", "src_ades_filed_gleich",
    "src_typ_gleich", "src_regel_gleich", "src_ruf_gleich", "src_ruf_len",
    "src_ruf_pre", "src_ruf_num_gleich", "src_ruf_num_len_diff",
]


def add_source_agreement(d):
    out = d.copy()

    def normal(values):
        return (values.astype("string").fillna("<NA>").str.upper()
                .str.replace(r"[^A-Z0-9]", "", regex=True))

    flight = normal(out["FLIGHT_mvt"])
    callsign = normal(out["CALLSIGN_flt"])
    for left, right, name in [
        ("ADEP_mvt", "ADEP_flt", "src_adep_gleich"),
        ("ADES_mvt", "ADES_flt", "src_ades_flt_gleich"),
        ("ADES_mvt", "ADES_FILED_flt", "src_ades_filed_gleich"),
        ("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt", "src_typ_gleich"),
        ("FLIGHT_RULE_mvt", "FLIGHT_RULE_flt", "src_regel_gleich"),
    ]:
        out[name] = (normal(out[left]) == normal(out[right])).astype("int8")
    out["src_ruf_gleich"] = (flight == callsign).astype("int8")
    out["src_ruf_len"] = callsign.str.len().astype("int8")
    out["src_ruf_pre"] = callsign.str[:3].astype("string")
    flight_number = flight.str.extract(r"(\d+)", expand=False).fillna("")
    callsign_number = callsign.str.extract(r"(\d+)", expand=False).fillna("")
    out["src_ruf_num_gleich"] = (flight_number == callsign_number).astype("int8")
    out["src_ruf_num_len_diff"] = (flight_number.str.len() - callsign_number.str.len()).astype("int8")
    return out


def align_categories(train, valid):
    for column in H.KAT + ["src_ruf_pre"]:
        categories = pd.unique(pd.concat(
            [train[column].astype(str), valid[column].astype(str)], ignore_index=True
        ))
        train[column] = pd.Categorical(train[column].astype(str), categories=categories)
        valid[column] = pd.Categorical(valid[column].astype(str), categories=categories)


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
    train = add_source_agreement(H.baue(learning, MG.alle()))
    valid = add_source_agreement(H.baue(held_out, MG.alle()))
    align_categories(train, valid)
    train = train[train.hat_aobt.eq(1)].copy()
    valid = valid[valid.hat_aobt.eq(1)].copy()
    train["rest"] = train.y - train.ersatz_s
    valid["rest"] = valid.y - valid.ersatz_s
    started = time.time()
    baseline, baseline_round = score(train, valid, H.MERKMALE_HAUPT)
    candidate, candidate_round = score(train, valid, H.MERKMALE_HAUPT + FEATURES)
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
