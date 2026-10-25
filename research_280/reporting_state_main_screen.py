"""Pre-registered January/July main-lane screen: the airport's reporting state as tree features. Fixed 16.09.2026.

Why. The main lane is 98.5 % of rows and about 40 % of all squared error, and its target is the disagreement between
NM's off-block clock and the airport's written block time. Nothing in the 91 features measures how the airport is
writing timestamps at that moment. The arrival rows make it observable and stay observable at rank time, where they
carry BLOCK_TIME, TAXITIME and SCHED_TIME at 100 % while departures carry none of them.

Measured beforehand over all 2.085 million 2025 departures (research_280/reporting_state_signal.py), arrivals of the
same airport in a two-hour bucket:
  share of arrival block times with a non-zero seconds field, from 0.877 to 1.000 across quintiles: the main-lane
  target's RMSE falls 415.5 -> 310.1 s and the median |rest| 179 -> 146 s; fallback copy rate 8.67 % -> 4.90 % and
  extreme rate 6.14 % -> 1.37 %
  arrival copy rate, across quintiles: fallback copy rate 2.93 % -> 14.33 %, extreme rate 1.94 % -> 10.42 %
Rome, the airport that carries the tail, has an arrival copy rate of 12.1 % against 4 to 5.5 % elsewhere.

Not the same as the rejected own-arrival copy flag of research_270, which asked whether the SAME flight's arrival was
a copy (5.87 % against 5.60 %, dead). This is the airport's state in the window, not the individual flight.

Features under test, from features_reporting_state.py: rs_sek_90, rs_kopie_90, rs_nmfehlt_90, rs_n_90, rs_sek_tag,
rs_kopie_tag. No target is read on the departure side.

Protocol, unchanged from the repository's main-lane screens (research_250/queue_screen.py, and identical to
research_260/neighbour_aggregates_diagnostic.py): one LightGBM, 1400 rounds, learning rate 0.03, 255 leaves,
min_child_samples 50, subsample and colsample 0.8, seed 177, early stopping 80 on the held-out fold; learning months
2-6 and 8-12, held out 1 and 7; target y - ersatz_s, prediction max(ersatz_s + model, 300). Baseline and candidate
are fitted in the same run on the same rows, so no saved baseline is relied upon.

Decision, to plan a six-pair confirmation:
  1 RMSE gain >= 0.3 s, the bar the repository uses for main-lane screens
  2 MSE gain positive without its 100 most favourable rows
  3 January and July both gain
Otherwise stop. Diagnostic scale only; shipping would need a full retrain and the release checks.

Run from the repository root:  python3 -u -B research_280/reporting_state_main_screen.py
"""
import argparse
import glob
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--jobs", type=int, default=8)
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main_model as H  # torch before tree libraries on macOS
import features_reporting_state as RS
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from research_250 import candidate_features as B

PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT / ".cache/clock_queue_release_20260912/data"
OUTPUT = PROJECT / "research_280/results/reporting_state_main_screen.json"
FEATURES = B.main_features(H.MERKMALE_HAUPT)
EXTRA = ["y", "ersatz_s", "hat_aobt", "MVT_TIME_UTC_mvt", "MVT_ID_mvt", "ADEP_mvt"]
started = time.time()


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def reporting_state():
    """rs_* for every 2025 departure, built from the raw movement rows including arrivals."""
    frames = [pq.ParquetFile(p).read(columns=RS.RAW_COLUMNS + ["MVT_ID_mvt"], use_threads=False).to_pandas()
              for p in sorted(glob.glob(str(PROJECT / "data/training_2025-*.parquet")))]
    raw = pd.concat(frames, ignore_index=True)
    dep = raw.loc[raw.PHASE_mvt.astype(str).str.upper().str.startswith("DEP")].reset_index(drop=True)
    built = RS.baue(raw, dep)
    built["MVT_ID_mvt"] = dep.MVT_ID_mvt.to_numpy()
    log(f"reporting state built for {len(built):,} departures")
    return built.set_index("MVT_ID_mvt")


def table(months, state):
    frame = pd.read_parquet(RELEASE / ("months_" + "_".join(map(str, months)) + ".parquet"),
                            columns=list(dict.fromkeys(FEATURES + EXTRA)))
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)
    for column in RS.MERKMALE:
        frame[column] = frame.MVT_ID_mvt.map(state[column]).to_numpy(float)
    return frame


state = reporting_state()
train = table([2, 3, 4, 5, 6, 8, 9, 10, 11, 12], state)
valid = table([1, 7], state)
for column in H.KAT:
    if column in train.columns:
        categories = pd.unique(train[column].astype(str))
        train[column] = pd.Categorical(train[column].astype(str), categories=categories)
        valid[column] = pd.Categorical(valid[column].astype(str), categories=categories)
coverage = {c: float(valid[c].notna().mean()) for c in RS.MERKMALE}
log(f"rows {len(train):,} train, {len(valid):,} valid; coverage {json.dumps({k: round(v, 4) for k, v in coverage.items()})}")

y_train = (train.y - train.ersatz_s).to_numpy()
y_valid_target = (valid.y - valid.ersatz_s).to_numpy()
y = valid.y.to_numpy(float)
ersatz = valid.ersatz_s.to_numpy(float)
predictions, models = {}, {}
for name, columns in (("baseline", FEATURES), ("candidate", FEATURES + RS.MERKMALE)):
    model = lgb.LGBMRegressor(objective="l2", n_estimators=1400, learning_rate=.03, num_leaves=255,
                              min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.8,
                              verbose=-1, n_jobs=args.jobs, random_state=177)
    model.fit(train[columns], y_train, eval_set=[(valid[columns], y_valid_target)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(300)])
    predictions[name] = np.maximum(ersatz + model.predict(valid[columns]), 300)
    models[name] = model
    log(f"{name}: rmse {np.sqrt(np.mean((predictions[name] - y) ** 2)):.4f}, best iteration {model.best_iteration_}")

gain = (predictions["baseline"] - y) ** 2 - (predictions["candidate"] - y) ** 2
keep = np.ones(len(gain), bool)
keep[np.argsort(gain)[::-1][:100]] = False
month = pd.to_datetime(valid.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
importance = models["candidate"].booster_.feature_importance("gain")
columns = FEATURES + RS.MERKMALE
report = {"validation": "single-LightGBM January/July main-lane screen; not the full ensemble and not a leaderboard forecast",
          "rows": int(len(valid)), "coverage": coverage,
          "baseline_rmse": float(np.sqrt(np.mean((predictions["baseline"] - y) ** 2))),
          "candidate_rmse": float(np.sqrt(np.mean((predictions["candidate"] - y) ** 2))),
          "best_iteration": {k: int(v.best_iteration_) for k, v in models.items()},
          "mse_gain": float(gain.mean()), "mse_gain_without_top100_rows": float(gain[keep].mean()),
          "january_mse_gain": float(gain[month == 1].mean()), "july_mse_gain": float(gain[month == 7].mean()),
          "rows_improved": int((gain > 0).sum()), "rows_worsened": int((gain < 0).sum()),
          "airport_mse_gain": {a: float(gain[valid.ADEP_mvt.astype(str).to_numpy() == a].mean())
                               for a in sorted(valid.ADEP_mvt.astype(str).unique())},
          "new_feature_importance_rank": {c: int((importance > importance[columns.index(c)]).sum() + 1) for c in RS.MERKMALE},
          "new_feature_gain_share": {c: float(importance[columns.index(c)] / importance.sum()) for c in RS.MERKMALE},
          "seconds": time.time() - started}
report["rmse_gain"] = report["baseline_rmse"] - report["candidate_rmse"]
checks = {"1_rmse_gain_at_least_0_3_s": report["rmse_gain"] >= .3,
          "2_survives_top100_removal": report["mse_gain_without_top100_rows"] > 0,
          "3_january_and_july_gain": report["january_mse_gain"] > 0 and report["july_mse_gain"] > 0}
report["decision"] = {"checks": checks, "plan_six_pair_confirmation": bool(all(checks.values()))}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
print("\nSUMMARY", json.dumps({k: report[k] for k in ("baseline_rmse", "candidate_rmse", "rmse_gain", "mse_gain",
                                                     "mse_gain_without_top100_rows", "january_mse_gain",
                                                     "july_mse_gain", "rows_improved", "rows_worsened", "decision")},
                              indent=1))
print("RANKS", report["new_feature_importance_rank"])
print("AIRPORTS", {a: round(v) for a, v in report["airport_mse_gain"].items()})
