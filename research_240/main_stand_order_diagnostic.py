"""Pre-registered January/July main-lane diagnostic: stand order against the NM off-block time.

Pre-specified 13.09.2026, before any fit, after stand_order_residuals.py found that the complete v45 holdout ensemble
under-predicts where another arrival blocks in at the departure's stand shortly before AOBT_3 while no departure left
that stand since the arrival before (mean residual +63 to +100 s in those cells, cross-fitted cell-bias bound 81 MSE,
enrichment 24 to 57 times for not-copied rows whose block time is over 30 min before AOBT_3). Run outside the
repository, then moved into research_240/. No leaderboard feedback.

Added columns (STAND_ORDER), computed separately on each source split from its own raw files, label-free at rank time
(arrival in-block times and stands are complete in ranking.parquet):
  so_s1       in-block time of the last arrival at the same airport and STAND_mvt at or before MVT_TIME, minus AOBT_3
  so_s2       the same for the second-to-last arrival
  so_between  departures from the same stand with MVT_TIME strictly between that second-to-last in-block and MVT_TIME
Model and data exactly as main_stage_diagnostic.py: queue-screen LightGBM (1400 rounds, learning rate 0.03, 255 leaves,
min_child_samples 50, subsample and colsample 0.8, seed 177, early stopping 80 on the fold), research encoding, v45
release cache, learning months 2-6 and 8-12, held-out January/July main rows, prediction max(ersatz_s + model, 300).
Baseline: the saved baseline_91 predictions of main_stage_diagnostic.py, aligned by MVT_ID and checked (RMSE 223.68022).
Decision to run a six-pair main screen: (1) RMSE gain >= 0.3 s, (2) MSE gain positive without its 100 most favourable
rows, (3) January and July both gain. Otherwise stop. Diagnostic only; not full ensemble; no submission.

Run:
  python3 -B research_240/main_stand_order_diagnostic.py --project <repository> --baseline <main_stage_predictions_1_7.parquet> --output <folder>
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
parser.add_argument("--baseline", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--jobs", type=int, default=8)
args = parser.parse_args()
project, output, baseline_path = args.project.resolve(), args.output.resolve(), args.baseline.resolve()
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS
from fid_gap_cv import month_of_file
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from research_250 import candidate_features as B

RELEASE = project / ".cache/clock_queue_release_20260912/data"
FEATURES = B.main_features(H.MERKMALE_HAUPT)
STAND_ORDER = ["so_s1", "so_s2", "so_between"]
EXTRA = ["y", "ersatz_s", "hat_aobt", "versp_min", "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "MVT_ID_mvt"]
RAW = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]
EDGES = [-np.inf, -3600, -1800, -900, -300, 0, 300, 900, np.inf]
ORIGIN = pd.Timestamp("2025-01-01", tz="UTC")
BASELINE_RMSE = 223.68022001367427
paths = sorted((project / "data").glob("training_2025-*.parquet"))
assert len(paths) == 12 and {month_of_file(p) for p in paths} == set(range(1, 13))
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(), project / "main_model.py", Path(B.__file__).resolve()]}
started = time.time()


def seconds(values):
    return (pd.to_datetime(values, utc=True) - ORIGIN).dt.total_seconds().to_numpy(float)


def stand_events(months):
    raw = pd.concat([pd.read_parquet(p, columns=RAW) for p in paths if month_of_file(p) in months], ignore_index=True)
    phase = raw.PHASE_mvt.astype(str).str.upper()
    arr, dep = raw[phase.str.startswith("ARR").to_numpy()], raw[phase.str.startswith("DEP").to_numpy()]
    events = []
    for keys, times in (((arr.ADES_mvt.astype(str) + "|" + arr.STAND_mvt.astype(str)).to_numpy(), seconds(arr.BLOCK_TIME_UTC_mvt)),
                        ((dep.ADEP_mvt.astype(str) + "|" + dep.STAND_mvt.astype(str)).to_numpy(), seconds(dep.MVT_TIME_UTC_mvt))):
        frame = pd.DataFrame({"k": keys, "t": times}).dropna().sort_values(["k", "t"], kind="stable")
        events.append({k: g.t.to_numpy() for k, g in frame.groupby("k", sort=False)})
    return events


def add_stand_order(frame, months):
    arrivals, departures = stand_events(months)
    keys = (frame.ADEP_mvt.astype(str) + "|" + frame.STAND_mvt.astype(str)).to_numpy()
    mvt = seconds(frame.MVT_TIME_UTC_mvt)
    aobt = mvt - frame.ersatz_s.to_numpy(float)
    n = len(frame)
    t1, t2, between = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for key, idx in pd.Series(np.arange(n)).groupby(keys).indices.items():
        a = arrivals.get(key)
        if a is None:
            continue
        t = mvt[idx]
        pos = np.searchsorted(a, t, side="right")
        t1[idx] = np.where(pos >= 1, a[np.maximum(pos - 1, 0)], np.nan)
        second = np.where(pos >= 2, a[np.maximum(pos - 2, 0)], np.nan)
        t2[idx] = second
        d = departures.get(key, np.array([]))
        lo = np.searchsorted(d, np.nan_to_num(second, nan=np.inf), side="right")
        hi = np.searchsorted(d, t, side="left")
        between[idx] = np.where(pos >= 2, hi - lo, np.nan)
    frame["so_s1"], frame["so_s2"], frame["so_between"] = t1 - aobt, t2 - aobt, between
    return frame


def load(months):
    path = RELEASE / ("months_" + "_".join(map(str, months)) + ".parquet")
    wanted = list(dict.fromkeys(FEATURES + EXTRA))
    missing = [c for c in wanted if c not in set(pq.read_schema(path).names)]
    assert not missing, missing
    frame = pd.read_parquet(path, columns=wanted)
    return frame.loc[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)


learn_months, held_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], [1, 7]
train = add_stand_order(load(learn_months), learn_months)
valid = add_stand_order(load(held_months), held_months)
assert train.MVT_ID_mvt.is_unique and valid.MVT_ID_mvt.is_unique and set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))
for col in H.KAT:
    if col in train.columns:
        cats = pd.unique(train[col].astype(str))
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)
y = valid.y.to_numpy(float)
saved = pd.read_parquet(baseline_path)
assert saved.MVT_ID_mvt.is_unique and set(saved.MVT_ID_mvt) == set(valid.MVT_ID_mvt)
saved = saved.set_index("MVT_ID_mvt").loc[valid.MVT_ID_mvt]
assert np.array_equal(saved.y.to_numpy(float), y)
base_pred = saved.baseline_91.to_numpy(float)
assert abs(np.sqrt(np.mean((base_pred - y)**2)) - BASELINE_RMSE) < 1e-9
coverage = {c: {"train": float(np.isfinite(train[c].to_numpy(float)).mean()), "valid": float(np.isfinite(valid[c].to_numpy(float)).mean())} for c in STAND_ORDER}
print("LOADED", len(train), len(valid), json.dumps(coverage), f"{time.time()-started:.0f}s", flush=True)

features = FEATURES + STAND_ORDER
before = time.time()
model = lgb.LGBMRegressor(objective="l2", n_estimators=1400, learning_rate=.03, num_leaves=255, min_child_samples=50,
                          subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=args.jobs, random_state=177)
model.fit(train[features], train.y.to_numpy() - train.ersatz_s.to_numpy(),
          eval_set=[(valid[features], y - valid.ersatz_s.to_numpy())], eval_metric="rmse",
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(300)])
cand_pred = np.maximum(valid.ersatz_s.to_numpy() + model.predict(valid[features]), 300)
print("FIT candidate_stand_order", len(features), "best_iteration", model.best_iteration_, f"{time.time()-before:.0f}s", flush=True)

e0, e1 = (base_pred - y)**2, (cand_pred - y)**2
gain = e0 - e1
keep = np.ones(len(gain), bool)
keep[np.argsort(gain)[::-1][:100]] = False
raw_month = pd.to_datetime(valid.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
month = np.where(np.isin(raw_month, [1, 7]), raw_month, raw_month % 12 + 1)
s1_bin = pd.cut(pd.Series(valid.so_s1.to_numpy(float)), EDGES).astype(str).to_numpy()
b = valid.so_between.to_numpy(float)
cell = np.char.add(np.char.add(s1_bin.astype(str), " | "), np.where(np.isnan(b), "no2arr", np.where(b == 0, "b0", "b1+")).astype(str))
airports = valid.ADEP_mvt.astype(str).to_numpy()
importance = model.booster_.feature_importance("gain")
report = {"validation": "single-LightGBM January/July main diagnostic, research encoding; not full ensemble or leaderboard",
          "rows_train": int(len(train)), "rows_valid": int(len(valid)), "coverage": coverage,
          "baseline": {"rmse": float(np.sqrt(e0.mean())), "source": str(baseline_path)},
          "candidate": {"rmse": float(np.sqrt(e1.mean())), "best_iteration": int(model.best_iteration_), "seconds": time.time()-before},
          "rmse_gain": float(np.sqrt(e0.mean()) - np.sqrt(e1.mean())), "mse_gain": float(gain.mean()),
          "mse_gain_without_top100_rows": float(gain[keep].mean()),
          "january_mse_gain": float(gain[month == 1].mean()), "july_mse_gain": float(gain[month == 7].mean()),
          "cell_mse_gain_contribution": {c: {"rows": int((cell == c).sum()), "contribution": float(gain[cell == c].sum() / len(gain))} for c in sorted(set(cell))},
          "airport_mse_gain": {a: float(gain[airports == a].mean()) for a in sorted(set(airports))},
          "importance_gain_share": {c: float(importance[features.index(c)] / importance.sum()) for c in STAND_ORDER},
          "importance_rank": {c: int((importance > importance[features.index(c)]).sum() + 1) for c in STAND_ORDER},
          "code_sha256": CODE}
checks = {"1_rmse_gain_at_least_0_3_s": report["rmse_gain"] >= .3, "2_survives_top100_removal": report["mse_gain_without_top100_rows"] > 0,
          "3_january_and_july_gain": report["january_mse_gain"] > 0 and report["july_mse_gain"] > 0}
report["decision"] = {"checks": checks, "run_six_pair_screen": bool(all(checks.values()))}
report["seconds"] = time.time() - started
(output / "main_stand_order_diagnostic.json").write_text(json.dumps(report, indent=2))
pd.DataFrame({"MVT_ID_mvt": valid.MVT_ID_mvt.to_numpy(), "y": y, "baseline_91": base_pred, "candidate_stand_order": cand_pred}).to_parquet(
    output / "main_stand_order_predictions_1_7.parquet", index=False)
print("SUMMARY", json.dumps({k: report[k] for k in ("coverage", "baseline", "candidate", "rmse_gain", "mse_gain", "mse_gain_without_top100_rows",
                                                    "january_mse_gain", "july_mse_gain", "airport_mse_gain", "importance_gain_share",
                                                    "importance_rank", "decision")}, indent=2), flush=True)
print("CELLS", json.dumps({c: round(v["contribution"], 1) for c, v in report["cell_mse_gain_contribution"].items()}), flush=True)
