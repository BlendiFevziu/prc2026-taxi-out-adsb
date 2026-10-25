"""Pre-registered check: does the research harness's categorical re-encoding change the main lane?

Pre-specified 13.09.2026 (HANDOFF.md next action 2), before any fit. Pair 1_7 from the v45 release
cache: learn months 2-6 and 8-12, test January and July, main-lane rows only. Both fits use the
release tree (LightGBM l2, 1,106 rounds, learning rate 0.03, 255 leaves, min_child_samples 50,
subsample 0.8, colsample 0.8, seed 0, n_jobs 8) on the 91 v45 main features, target y - ersatz_s.
  production  category dtypes and missing values as stored (main_model.py, release_common.py)
  research    categoricals re-encoded as research_250/queue_screen.py does: string values,
              categories in first-appearance order of the training table, missing as "nan"
Replication gate: the production fit must reproduce the saved release tree lgb_0 on the test rows
within 1e-6 s. Metrics: main RMSE of max(ersatz_s + residual, 300) and without the floor,
per-airport RMSE, RMS and maximum prediction difference, category counts.
Decision: if the two main RMSEs differ by less than 0.3 s, the caveat is closed; otherwise the
earlier main-lane rejections within that margin are listed for a production re-screen.

Run from the repository root:
  python3 research_240/encoding_check_main.py --project . --output .cache/research_240/encoding_check_main.json
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
parser.add_argument("--release", default=Path(".cache/clock_queue_release_20260912"), type=Path)
parser.add_argument("--jobs", type=int, default=8)
args = parser.parse_args()
project, output = args.project.resolve(), args.output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project))
os.chdir(project)
release = args.release if args.release.is_absolute() else project / args.release

import main_model as H  # torch before tree libraries on macOS
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_250 import candidate_features as B

FEATURES = B.main_features(H.MERKMALE_HAUPT)
assert len(FEATURES) == 91, len(FEATURES)
ROUNDS, SEED, BAR = 1106, 0, 0.3


def load(months):
    frame = pd.read_parquet(release / "data" / ("months_" + "_".join(map(str, months)) + ".parquet"))
    return frame.loc[frame.hat_aobt.eq(1)].copy()


def tree():
    return lgb.LGBMRegressor(objective="l2", n_estimators=ROUNDS, learning_rate=.03, num_leaves=255,
                             min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.8,
                             verbose=-1, n_jobs=args.jobs, random_state=SEED)


started = time.time()
train, valid = load([2, 3, 4, 5, 6, 8, 9, 10, 11, 12]), load([1, 7])
assert set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))
target = train.y.to_numpy(float) - train.ersatz_s.to_numpy(float)
y, ersatz = valid.y.to_numpy(float), valid.ersatz_s.to_numpy(float)
airports = valid.ADEP_mvt.astype(str).to_numpy()

before = time.time()
residual_production = tree().fit(train[FEATURES], target).predict(valid[FEATURES])
production_seconds = time.time() - before
saved = joblib.load(release / "candidate/checkpoints/lgb_0.pkl")
saved = saved["model"] if isinstance(saved, dict) else saved
gap = float(np.abs(residual_production - saved.predict(valid[FEATURES])).max())
report = {"validation": "single release LightGBM tree, pair 1_7, main lane; encoding comparison, not an ensemble result",
          "features": len(FEATURES), "train_rows": int(len(train)), "test_rows": int(len(valid)),
          "replication_gate": {"max_abs_difference_s": gap, "passed": gap <= 1e-6},
          "production_fit_seconds": production_seconds}
print("REPLICATION", json.dumps(report["replication_gate"]), flush=True)
if not report["replication_gate"]["passed"]:
    output.write_text(json.dumps(report, indent=2))
    sys.exit("Production fit does not reproduce the saved release tree lgb_0")

x_train, x_valid = train[FEATURES].copy(), valid[FEATURES].copy()
categories = {}
for col in [c for c in H.KAT if c in FEATURES]:
    order = pd.unique(train[col].astype(str))
    x_train[col] = pd.Categorical(train[col].astype(str), categories=order)
    x_valid[col] = pd.Categorical(valid[col].astype(str), categories=order)
    categories[col] = {"production_train_categories": int(len(train[col].cat.categories)),
                       "research_train_categories": int(len(order)),
                       "missing_share_train": float(train[col].isna().mean()),
                       "missing_share_test": float(valid[col].isna().mean()),
                       "research_test_rows_unseen_in_train": int(x_valid[col].isna().sum())}
before = time.time()
residual_research = tree().fit(x_train, target).predict(x_valid)
research_seconds = time.time() - before


def rmse(residual, mask=None, floor=True):
    prediction = ersatz + residual
    if floor:
        prediction = np.maximum(prediction, 300)
    squared = (prediction - y) ** 2
    return float(np.sqrt(squared.mean() if mask is None else squared[mask].mean()))


difference = residual_research - residual_production
report.update({
    "research_fit_seconds": research_seconds, "categories": categories,
    "main_rmse": {"production": rmse(residual_production), "research": rmse(residual_research),
                  "production_no_floor": rmse(residual_production, floor=False),
                  "research_no_floor": rmse(residual_research, floor=False)},
    "prediction_difference": {"rms_s": float(np.sqrt(np.mean(difference ** 2))), "max_abs_s": float(np.abs(difference).max()),
                              "share_over_60_s": float((np.abs(difference) > 60).mean())},
    "by_airport": {a: {"rows": int((airports == a).sum()), "production": rmse(residual_production, airports == a),
                       "research": rmse(residual_research, airports == a)} for a in sorted(set(airports))}})
change = report["main_rmse"]["research"] - report["main_rmse"]["production"]
report["decision"] = {"rmse_research_minus_production_s": change, "bar_s": BAR, "caveat_closed": bool(abs(change) < BAR)}
report["seconds"] = time.time() - started
report["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__).resolve(), project / "main_model.py", Path(B.__file__).resolve())}
output.write_text(json.dumps(report, indent=2))
print("RESULT", json.dumps({"main_rmse": report["main_rmse"], "prediction_difference": report["prediction_difference"],
                            "decision": report["decision"]}), flush=True)
