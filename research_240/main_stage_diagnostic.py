"""Pre-registered January/July main-lane diagnostic: stand-runway reference taxi times as main features.

Pre-specified 13.09.2026, before any fit, after fallback_reference_screen.py rejected the same references in
the fallback lane. Run outside the repository, then moved into research_240/. No leaderboard feedback.

Why the main lane: the main target is rest = y - ersatz_s = AOBT_3 - BLOCK_TIME, the reporting gap. An NM taxi
time far below the stand's physical minimum means AOBT_3 is implausibly late; far above it, AOBT_3 is early or
the day's queue was long. The 91 v45 main features carry ersatz_s, the straight-line distance and the runway,
but no stand-level taxi norm.

Tables: the v45 release cache (.cache/clock_queue_release_20260912/data), learning months 2-6 and 8-12,
held-out January/July, main rows (hat_aobt 1, y present).
References from the learning months only, over linked departures with 0 < ersatz_s <= 120 min, as in
fallback_reference_screen.py: 10th percentile and median per airport, stand and runway (at least 10 flights at
or below the 10th percentile), else per airport and runway, else per airport.
Added columns (STAGE_MAIN): ref_p10, ref_p50, ref_log_n, ref_level, ex_p10 = ersatz_s - ref_p10,
ex_p50 = ersatz_s - ref_p50.
Model: research_250/queue_screen.py single LightGBM unchanged (1400 rounds, learning rate 0.03, 255 leaves,
min_child_samples 50, subsample and colsample 0.8, seed 177, early stopping 80 on the held-out fold, research
categorical encoding), prediction max(ersatz_s + model, 300).
  baseline   the 91 v45 main features, B.main_features(H.MERKMALE_HAUPT)
  candidate  the 91 plus STAGE_MAIN
Replication check (reported): baseline RMSE within 0.05 s of queue_screen_1_7.json units_plus_queue.
Decision to run a six-pair main screen: (1) RMSE gain >= 0.3 s (the queue screen's bar), (2) the MSE gain stays
positive without its 100 most favourable rows, (3) January and July both gain. Otherwise stop.
Diagnostic only; not full ensemble; not a leaderboard forecast; no submission.

Run:
  python3 -B research_240/main_stage_diagnostic.py --project <repository> --output <folder>
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--jobs", type=int, default=8)
args = parser.parse_args()
project, output = args.project.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from research_250 import candidate_features as B

RELEASE = project / ".cache/clock_queue_release_20260912/data"
FEATURES = B.main_features(H.MERKMALE_HAUPT)
REFERENCE = ["ref_p10", "ref_p50", "ref_log_n", "ref_level"]
STAGE_MAIN = REFERENCE + ["ex_p10", "ex_p50"]
LEVELS = (("stand", ["airport", "stand", "runway"], 2), ("runway", ["airport", "runway"], 1), ("airport", ["airport"], 0))
MIN_BELOW = 10
EXTRA = ["y", "ersatz_s", "hat_aobt", "versp_min", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "MVT_ID_mvt"]
QUEUE_RMSE = json.loads((project / "research_250/results/queue_screen_1_7.json").read_text())["scores"]["units_plus_queue"]["rmse"]
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), project / "main_model.py", Path(B.__file__).resolve()]}
started = time.time()


def text(values):
    s = pd.Series(values).reset_index(drop=True)
    return np.where(s.notna().to_numpy(), s.astype(str).to_numpy(), "").astype(object)


def load(months):
    path = RELEASE / ("months_" + "_".join(map(str, months)) + ".parquet")
    wanted = list(dict.fromkeys(FEATURES + EXTRA))
    missing = [c for c in wanted if c not in set(pq.read_schema(path).names)]
    assert not missing, missing
    return pd.read_parquet(path, columns=wanted)


def reference_tables(frame):
    p = frame.ersatz_s.to_numpy(float)
    keep = frame.hat_aobt.eq(1).to_numpy() & (p > 0) & (p <= 7200)
    base = pd.DataFrame({"airport": frame.ADEP_mvt.astype(str).to_numpy()[keep], "stand": text(frame.STAND_mvt)[keep],
                         "runway": text(frame.RUNWAY_mvt)[keep], "p": p[keep]})
    tables = {}
    for name, keys, _ in LEVELS:
        data = base
        for k in keys[1:]:
            data = data[data[k].to_numpy() != ""]
        grouped = data.groupby(keys, sort=False).p
        table = pd.DataFrame({"p10": grouped.quantile(.1), "p50": grouped.median(), "n": grouped.size()})
        if name == "stand":
            joined = data.join(table.p10.rename("q"), on=keys)
            below = joined[joined.p.to_numpy() <= joined.q.to_numpy()].groupby(keys, sort=False).size()
            table = table[below.reindex(table.index).fillna(0).to_numpy() >= MIN_BELOW]
        tables[name] = table
    return tables


def lookup(airport, stand, runway, tables):
    keys = pd.DataFrame({"airport": np.asarray(airport, object), "stand": np.asarray(stand, object),
                         "runway": np.asarray(runway, object)})
    out = np.full((len(keys), 4), np.nan)
    for name, cols, level in LEVELS:
        joined = keys.join(tables[name], on=cols)
        fill = np.isnan(out[:, 1]) & joined.p50.notna().to_numpy()
        out[fill, 0] = joined.p10.to_numpy(float)[fill]
        out[fill, 1] = joined.p50.to_numpy(float)[fill]
        out[fill, 2] = np.log1p(joined.n.to_numpy(float)[fill])
        out[fill, 3] = level
    return pd.DataFrame(out, columns=REFERENCE)


def add_stage(frame, tables):
    ref = lookup(frame.ADEP_mvt.astype(str).to_numpy(), text(frame.STAND_mvt), text(frame.RUNWAY_mvt), tables)
    for c in REFERENCE:
        frame[c] = ref[c].to_numpy()
    frame["ex_p10"] = frame.ersatz_s.to_numpy(float) - frame.ref_p10.to_numpy()
    frame["ex_p50"] = frame.ersatz_s.to_numpy(float) - frame.ref_p50.to_numpy()
    return frame


learn = load([2, 3, 4, 5, 6, 8, 9, 10, 11, 12])
tables = reference_tables(learn)
train = add_stage(learn.loc[learn.hat_aobt.eq(1) & learn.y.notna()].reset_index(drop=True), tables)
del learn
held = load([1, 7])
valid = add_stage(held.loc[held.hat_aobt.eq(1) & held.y.notna()].reset_index(drop=True), tables)
del held
assert train.MVT_ID_mvt.is_unique and valid.MVT_ID_mvt.is_unique and set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))
for col in H.KAT:
    if col in train.columns:
        cats = pd.unique(train[col].astype(str))
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)
target = train.y.to_numpy() - train.ersatz_s.to_numpy()
valid_target = valid.y.to_numpy() - valid.ersatz_s.to_numpy()
print("LOADED", len(train), len(valid), f"{time.time()-started:.0f}s", flush=True)


def fit(features, name):
    before = time.time()
    model = lgb.LGBMRegressor(objective="l2", n_estimators=1400, learning_rate=.03, num_leaves=255, min_child_samples=50,
                              subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=args.jobs,
                              random_state=177)
    model.fit(train[features], target, eval_set=[(valid[features], valid_target)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(300)])
    prediction = np.maximum(valid.ersatz_s.to_numpy() + model.predict(valid[features]), 300)
    print("FIT", name, len(features), "best_iteration", model.best_iteration_, f"{time.time()-before:.0f}s", flush=True)
    return model, prediction, time.time()-before


base_model, base_pred, base_seconds = fit(FEATURES, "baseline_91")
cand_model, cand_pred, cand_seconds = fit(FEATURES + STAGE_MAIN, "candidate_97")
y = valid.y.to_numpy(float)
e0, e1 = (base_pred-y)**2, (cand_pred-y)**2
gain = e0 - e1
keep = np.ones(len(gain), bool)
keep[np.argsort(gain)[::-1][:100]] = False
raw_month = pd.to_datetime(valid.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
month = np.where(np.isin(raw_month, [1, 7]), raw_month, raw_month % 12 + 1)
copied = (np.abs(y - valid.versp_min.to_numpy(float)*60) <= 10)
airports = valid.ADEP_mvt.astype(str).to_numpy()
importance = cand_model.booster_.feature_importance("gain")
names = FEATURES + STAGE_MAIN
report = {"validation": "single-LightGBM January/July main diagnostic, research encoding; not full ensemble or leaderboard",
          "rows_train": int(len(train)), "rows_valid": int(len(valid)),
          "reference_level_share_valid": {str(k): float(v) for k, v in pd.Series(valid.ref_level).value_counts(normalize=True).items()},
          "baseline": {"rmse": float(np.sqrt(e0.mean())), "best_iteration": int(base_model.best_iteration_), "seconds": base_seconds,
                       "queue_screen_rmse": QUEUE_RMSE, "replicates_queue_screen": bool(abs(np.sqrt(e0.mean())-QUEUE_RMSE) <= .05)},
          "candidate": {"rmse": float(np.sqrt(e1.mean())), "best_iteration": int(cand_model.best_iteration_), "seconds": cand_seconds},
          "rmse_gain": float(np.sqrt(e0.mean())-np.sqrt(e1.mean())), "mse_gain": float(gain.mean()),
          "mse_gain_without_top100_rows": float(gain[keep].mean()),
          "january_mse_gain": float(gain[month == 1].mean()), "july_mse_gain": float(gain[month == 7].mean()),
          "copied_rows": {"rows": int(copied.sum()), "mse_gain_contribution": float(gain[copied].sum()/len(gain))},
          "not_copied_rows": {"rows": int((~copied).sum()), "mse_gain_contribution": float(gain[~copied].sum()/len(gain))},
          "airport_mse_gain": {a: {"rows": int((airports == a).sum()), "mse_gain": float(gain[airports == a].mean()),
                                   "baseline_rmse": float(np.sqrt(e0[airports == a].mean())),
                                   "candidate_rmse": float(np.sqrt(e1[airports == a].mean()))} for a in sorted(set(airports))},
          "stage_importance_gain_share": {n: float(importance[names.index(n)]/importance.sum()) for n in STAGE_MAIN},
          "stage_importance_rank": {n: int((importance > importance[names.index(n)]).sum()+1) for n in STAGE_MAIN},
          "code_sha256": CODE}
checks = {"1_rmse_gain_at_least_0_3_s": report["rmse_gain"] >= .3,
          "2_survives_top100_removal": report["mse_gain_without_top100_rows"] > 0,
          "3_january_and_july_gain": report["january_mse_gain"] > 0 and report["july_mse_gain"] > 0}
report["decision"] = {"checks": checks, "run_six_pair_screen": bool(all(checks.values()))}
report["seconds"] = time.time()-started
(output / "main_stage_diagnostic.json").write_text(json.dumps(report, indent=2))
pd.DataFrame({"MVT_ID_mvt": valid.MVT_ID_mvt.to_numpy(), "ADEP_mvt": airports, "y": y, "baseline_91": base_pred,
              "candidate_97": cand_pred}).to_parquet(output / "main_stage_predictions_1_7.parquet", index=False)
print("SUMMARY", json.dumps({k: report[k] for k in ("baseline", "candidate", "rmse_gain", "mse_gain", "mse_gain_without_top100_rows",
                                                    "january_mse_gain", "july_mse_gain", "copied_rows", "not_copied_rows",
                                                    "stage_importance_gain_share", "stage_importance_rank", "decision")}, indent=2),
      flush=True)
print("AIRPORTS", json.dumps({a: round(v["mse_gain"]) for a, v in report["airport_mse_gain"].items()}), flush=True)
