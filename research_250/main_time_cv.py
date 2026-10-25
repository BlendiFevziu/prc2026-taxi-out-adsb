"""Locked v44 vs corrected-time single-LightGBM temporal confirmation.

Uses full core tables made independently by fallback250_cv.py. It never reads
ranking labels, changes a model bundle, trains the full ensemble, or uploads.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--core-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--pairs", default="2,8;3,9;4,10;5,11;6,12")
parser.add_argument("--jobs", type=int, default=8)
args = parser.parse_args()
for key in ("project", "core_dir", "output"):
    setattr(args, key, getattr(args, key).resolve())
os.environ.setdefault("MPLCONFIGDIR", str(args.output / "matplotlib"))
sys.path.insert(0, str(args.project.resolve()))
os.chdir(args.project)
import main_model as H
import lightgbm as lgb
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features_time_units as U

lookup = {}
for manifest_path in sorted(args.core_dir.glob("core_*.json")):
    manifest = json.loads(manifest_path.read_text())
    months = tuple(sorted(map(int, manifest["source_months"])))
    path = args.core_dir / manifest["core_file"]
    if months in lookup:
        raise ValueError(f"Ambiguous cached reference for months {months}; use a clean core directory")
    lookup[months] = path
args.output.mkdir(parents=True, exist_ok=True)
all_months = set(range(1, 13))
for pair in [tuple(map(int, p.split(","))) for p in args.pairs.split(";")]:
    assert len(pair) == 2 and len(set(pair)) == 2 and set(pair) <= all_months
    suffix = "_".join(map(str, pair))
    score_path = args.output / f"time_cv_result_{suffix}.json"
    pred_path = args.output / f"time_cv_predictions_{suffix}.parquet"
    if score_path.exists() or pred_path.exists():
        raise FileExistsError(f"Use a new output directory to preserve {score_path}")
    fitting = lookup[tuple(sorted(all_months-set(pair)))]
    testing = lookup[tuple(sorted(pair))]
    train = pd.read_parquet(fitting)
    valid = pd.read_parquet(testing)
    assert set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))
    train = train.loc[train.hat_aobt.eq(1) & train.y.notna()].copy()
    valid = valid.loc[valid.hat_aobt.eq(1) & valid.y.notna()].copy()
    for col in H.KAT:
        cats = pd.unique(train[col].astype(str))
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)
    target = train.y.to_numpy()-train.ersatz_s.to_numpy()
    valid_target = valid.y.to_numpy()-valid.ersatz_s.to_numpy()
    out = valid[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "y"]].copy()
    report = {"validation": "single-LightGBM main component, not full ensemble or leaderboard",
              "pair": pair, "seed": 177, "training_rows": len(train), "validation_rows": len(valid),
              "source_tables": [str(fitting.resolve()), str(testing.resolve())],
              "replacement_columns": U.CORRECTED_COLUMNS, "scores": {}}
    for name in ["v44_main", "units_replaced"]:
        x, xv = train[H.MERKMALE_HAUPT].copy(), valid[H.MERKMALE_HAUPT].copy()
        if name == "units_replaced":
            for col in U.CORRECTED_COLUMNS:
                x[col], xv[col] = train["tu_"+col].to_numpy(), valid["tu_"+col].to_numpy()
        started = time.time()
        print("MAIN FIT", pair, name, flush=True)
        model = lgb.LGBMRegressor(objective="l2", n_estimators=1400,
            learning_rate=.03, num_leaves=255, min_child_samples=50,
            subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1,
            n_jobs=args.jobs, random_state=177)
        model.fit(x, target, eval_set=[(xv, valid_target)], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(300)])
        out[name] = np.maximum(valid.ersatz_s.to_numpy()+model.predict(xv), 300)
        e = (out[name]-valid.y)**2
        report["scores"][name] = {"rmse": float(np.sqrt(e.mean())), "mse": float(e.mean()),
            "top5_sse_share": float(e.nlargest(5).sum()/e.sum()),
            "best_iteration": model.best_iteration_, "seconds": time.time()-started}
        print("MAIN RESULT", pair, name, json.dumps(report["scores"][name]), flush=True)
        del x, xv, model
        gc.collect()
    report["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__), Path(U.__file__), Path(H.__file__)]}
    out.to_parquet(pred_path, index=False)
    score_path.write_text(json.dumps(report, indent=2))
    print("MAIN SUMMARY", json.dumps(report), flush=True)
    del train, valid, out
    gc.collect()
