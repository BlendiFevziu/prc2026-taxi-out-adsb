"""Pre-registered January/July diagnostic: summaries of the ordered neighbour tensor as main-lane tree features.

Pre-specified 14.09.2026 before any fit. No leaderboard feedback. v47's gain comes from the origin-grouped neighbour sequence
(24 movements, +/-12 rows within 90 minutes) read by the neural quarter; the tree models, 75 % of the blend, never see it.
This diagnostic asks whether fixed summaries of the same tensors carry information the 91 v45 features lack.

Inputs: the origin tensors and main-row order of research_260/sequence_release_v47.py validate-prepare
(.cache/sequence_release_v47/validate, learning months 2-6 and 8-12, held-out 1 and 7), the v45 release cache tables, and the
saved 91-feature January/July single-LightGBM predictions of research_260/main_stage_diagnostic.py as the baseline
(223.680 s; it reproduced the queue screen within 0.02 s).
Summaries over each departure's valid neighbours (tensor fields of research_260/sequence_features.py, scaled as there):
  nb_valid, nb_departures, nb_arrivals
  nb_dep_nm_taxi_mean, nb_dep_nm_taxi_sd, nb_dep_nm_taxi_before, nb_dep_nm_taxi_after (departures with AOBT_3; before/after
  by relative time), nb_nearest_dep_nm_taxi, nb_nearest_dep_dt
  nb_same_runway_dep_nm_taxi, nb_same_runway, nb_same_stand, nb_same_operator, nb_same_type
  nb_arr_taxi_mean, nb_arr_nm_gap_mean, nb_arr_copy_share (arrival records)
  nb_sched_delay_mean, nb_fid_near (neighbours within 1,000 flight IDs), nb_fid_min_abs
Model: research_250/queue_screen.py single LightGBM unchanged (1400 rounds, learning rate 0.03, 255 leaves, min_child_samples 50,
subsample and colsample 0.8, seed 177, early stopping 80 on the held-out fold, research encoding), max(ersatz + model, 300),
features = the 91 v45 main features plus the 21 summaries.
Decision to plan a six-pair confirmation: (1) RMSE gain >= 0.3 s against the saved baseline, (2) MSE gain positive without its
100 most favourable rows, (3) January and July both gain. Otherwise stop. Diagnostic only; shipping would need a full retrain.

Run from the repository root:  python3 -u -B research_260/neighbour_aggregates_diagnostic.py --baseline <main_stage_predictions_1_7.parquet>
"""
import argparse
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--jobs", type=int, default=8)
parser.add_argument("--chunk", type=int, default=250000)
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main_model as H  # torch before tree libraries on macOS
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from research_250 import candidate_features as B
from research_260.airport_ablation import sha, write_json, log

PROJECT = Path(__file__).resolve().parents[1]
VALIDATE = PROJECT / ".cache/sequence_release_v47/validate"
RELEASE = PROJECT / ".cache/clock_queue_release_20260912/data"
FEATURES = B.main_features(H.MERKMALE_HAUPT)
EXTRA = ["y", "ersatz_s", "hat_aobt", "MVT_TIME_UTC_mvt", "MVT_ID_mvt"]
SUMMARIES = ["nb_valid", "nb_departures", "nb_arrivals", "nb_dep_nm_taxi_mean", "nb_dep_nm_taxi_sd", "nb_dep_nm_taxi_before",
             "nb_dep_nm_taxi_after", "nb_nearest_dep_nm_taxi", "nb_nearest_dep_dt", "nb_same_runway_dep_nm_taxi", "nb_same_runway",
             "nb_same_stand", "nb_same_operator", "nb_same_type", "nb_arr_taxi_mean", "nb_arr_nm_gap_mean", "nb_arr_copy_share",
             "nb_sched_delay_mean", "nb_fid_near", "nb_fid_min_abs"]
OUTPUT = PROJECT / "research_260/results/neighbour_aggregates_diagnostic.json"
started = time.time()


def masked_mean(values, mask):
    count = mask.sum(1)
    return np.where(count > 0, (values * mask).sum(1) / np.maximum(count, 1), np.nan), count


def summarise(xs, ms):
    out = {name: np.full(len(xs), np.nan, np.float32) for name in SUMMARIES}
    for lo in range(0, len(xs), args.chunk):
        x, m = xs[lo:lo + args.chunk].astype(np.float32), ms[lo:lo + args.chunk]
        rel, dep, arr, taxi, has_aobt = x[:, :, 0], x[:, :, 1] > .5, x[:, :, 2] > .5, x[:, :, 3], x[:, :, 4] > .5
        arr_taxi, arr_gap, arr_copy = x[:, :, 5], x[:, :, 6], x[:, :, 8]
        same_stand, same_rwy, same_op, same_type = x[:, :, 9] > .5, x[:, :, 10] > .5, x[:, :, 11] > .5, x[:, :, 15] > .5
        sdel, fid, has_fid = x[:, :, 12], x[:, :, 13], x[:, :, 14] > .5
        dep_ok = m & dep & has_aobt
        s = slice(lo, lo + len(x))
        out["nb_valid"][s], out["nb_departures"][s], out["nb_arrivals"][s] = m.sum(1), (m & dep).sum(1), (m & arr).sum(1)
        mean, count = masked_mean(taxi, dep_ok)
        out["nb_dep_nm_taxi_mean"][s] = mean
        sq, _ = masked_mean(taxi ** 2, dep_ok)
        out["nb_dep_nm_taxi_sd"][s] = np.where(count > 1, np.sqrt(np.maximum(sq - mean ** 2, 0)), np.nan)
        out["nb_dep_nm_taxi_before"][s] = masked_mean(taxi, dep_ok & (rel < 0))[0]
        out["nb_dep_nm_taxi_after"][s] = masked_mean(taxi, dep_ok & (rel > 0))[0]
        distance = np.where(dep_ok, np.abs(rel), np.inf)
        nearest = distance.argmin(1)
        has_nearest = np.isfinite(distance.min(1))
        rows = np.arange(len(x))
        out["nb_nearest_dep_nm_taxi"][s] = np.where(has_nearest, taxi[rows, nearest], np.nan)
        out["nb_nearest_dep_dt"][s] = np.where(has_nearest, rel[rows, nearest], np.nan)
        out["nb_same_runway_dep_nm_taxi"][s] = masked_mean(taxi, dep_ok & same_rwy)[0]
        out["nb_same_runway"][s], out["nb_same_stand"][s] = (m & same_rwy).sum(1), (m & same_stand).sum(1)
        out["nb_same_operator"][s], out["nb_same_type"][s] = (m & same_op).sum(1), (m & same_type).sum(1)
        arr_ok = m & arr
        out["nb_arr_taxi_mean"][s] = masked_mean(arr_taxi, arr_ok)[0]
        out["nb_arr_nm_gap_mean"][s] = masked_mean(arr_gap, arr_ok & (x[:, :, 7] > .5))[0]
        out["nb_arr_copy_share"][s] = masked_mean(arr_copy, arr_ok)[0]
        out["nb_sched_delay_mean"][s] = masked_mean(sdel, m)[0]
        fid_ok = m & has_fid
        out["nb_fid_near"][s] = (fid_ok & (np.abs(fid) <= .1)).sum(1)
        out["nb_fid_min_abs"][s] = np.where(fid_ok.any(1), np.where(fid_ok, np.abs(fid), np.inf).min(1), np.nan)
    return pd.DataFrame(out)


def table(months, side):
    path = RELEASE / ("months_" + "_".join(map(str, months)) + ".parquet")
    wanted = list(dict.fromkeys(FEATURES + EXTRA))
    frame = pd.read_parquet(path, columns=wanted)
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)
    rows = pd.read_parquet(VALIDATE / f"{side}_rows.parquet", columns=["MVT_ID_mvt"])
    if not np.array_equal(rows.MVT_ID_mvt.to_numpy(), frame.MVT_ID_mvt.to_numpy()):
        raise ValueError(f"{side} row order differs from the validate arrays")
    xs = np.load(VALIDATE / f"{side}_origin_xs.npy", mmap_mode="r")
    ms = np.load(VALIDATE / f"{side}_origin_ms.npy", mmap_mode="r")
    summary = summarise(xs, ms)
    for name in SUMMARIES:
        frame[name] = summary[name].to_numpy()
    return frame


train, valid = table([2, 3, 4, 5, 6, 8, 9, 10, 11, 12], "train"), table([1, 7], "valid")
for col in H.KAT:
    if col in train.columns:
        cats = pd.unique(train[col].astype(str))
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)
log(f"LOADED {len(train)} {len(valid)} summaries in {time.time() - started:.0f}s")
features = FEATURES + SUMMARIES
model = lgb.LGBMRegressor(objective="l2", n_estimators=1400, learning_rate=.03, num_leaves=255, min_child_samples=50, subsample=.8,
                          subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=args.jobs, random_state=177)
model.fit(train[features], train.y - train.ersatz_s, eval_set=[(valid[features], valid.y - valid.ersatz_s)], eval_metric="rmse",
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(300)])
candidate = np.maximum(valid.ersatz_s.to_numpy() + model.predict(valid[features]), 300)
base = pd.read_parquet(args.baseline).set_index("MVT_ID_mvt").loc[valid.MVT_ID_mvt]
y = valid.y.to_numpy(float)
np.testing.assert_allclose(base.y.to_numpy(float), y, atol=0, rtol=0)
b = base.baseline_91.to_numpy(float)
gain = (b - y) ** 2 - (candidate - y) ** 2
keep = np.ones(len(gain), bool)
keep[np.argsort(gain)[::-1][:100]] = False
month = pd.to_datetime(valid.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
month = np.where(np.isin(month, [1, 7]), month, month % 12 + 1)
importance = model.booster_.feature_importance("gain")
report = {"validation": "single-LightGBM January/July main diagnostic; not full ensemble or leaderboard",
          "baseline_rmse": float(np.sqrt(np.mean((b - y) ** 2))), "candidate_rmse": float(np.sqrt(np.mean((candidate - y) ** 2))),
          "best_iteration": int(model.best_iteration_), "mse_gain": float(gain.mean()), "mse_gain_without_top100_rows": float(gain[keep].mean()),
          "january_mse_gain": float(gain[month == 1].mean()), "july_mse_gain": float(gain[month == 7].mean()),
          "summary_importance_gain_share": {n: float(importance[features.index(n)] / importance.sum()) for n in SUMMARIES},
          "summary_importance_rank": {n: int((importance > importance[features.index(n)]).sum() + 1) for n in SUMMARIES},
          "baseline_sha256": sha(args.baseline), "seconds": time.time() - started}
report["rmse_gain"] = report["baseline_rmse"] - report["candidate_rmse"]
checks = {"1_rmse_gain_at_least_0_3_s": report["rmse_gain"] >= .3, "2_survives_top100_removal": report["mse_gain_without_top100_rows"] > 0,
          "3_january_and_july_gain": report["january_mse_gain"] > 0 and report["july_mse_gain"] > 0}
report["decision"] = {"checks": checks, "plan_six_pair_confirmation": bool(all(checks.values()))}
write_json(OUTPUT, report)
print("SUMMARY", json.dumps({k: report[k] for k in ("baseline_rmse", "candidate_rmse", "rmse_gain", "mse_gain", "mse_gain_without_top100_rows",
                                                    "january_mse_gain", "july_mse_gain", "best_iteration", "decision")}, indent=1), flush=True)
print("TOP SUMMARIES", sorted(report["summary_importance_rank"].items(), key=lambda kv: kv[1])[:8], flush=True)
