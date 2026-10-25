"""Locked correctly timed queue ablation, above the unit-corrected main model.

January/July was the exploratory screen. Other source-month pairs confirm the
same configuration; no parameters are selected from their results.
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
parser.add_argument("--core-train", type=Path, required=True)
parser.add_argument("--core-valid", type=Path, required=True)
parser.add_argument("--baseline-predictions", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--jobs", type=int, default=8)
args = parser.parse_args()
for key in ("project", "core_train", "core_valid", "baseline_predictions", "output"):
    setattr(args, key, getattr(args, key).resolve())
train_source = json.loads(args.core_train.with_suffix(".json").read_text())
valid_source = json.loads(args.core_valid.with_suffix(".json").read_text())
pair = sorted(valid_source["source_months"])
assert len(pair) == len(set(pair)) == 2
assert set(train_source["source_months"]) == set(range(1, 13))-set(pair)
suffix = "_".join(map(str, pair))
score_path = args.output/f"queue_result_{suffix}.json"
pred_path = args.output/f"queue_predictions_{suffix}.parquet"
if score_path.exists() or pred_path.exists():
    raise FileExistsError(f"Use a new output directory to preserve {score_path}")
os.environ.setdefault("MPLCONFIGDIR", str(args.output / "matplotlib"))
sys.path.insert(0, str(args.project))
os.chdir(args.project)
import main_model as H
import lightgbm as lgb
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features_time_units as U
import features_correct_queue as Q
import features_queue as legacy_queue


def table(path):
    core = pd.read_parquet(path)
    source = json.loads(path.with_suffix(".json").read_text())
    raw = pd.concat([pd.read_parquet(args.project/"data"/name, columns=Q.RAW_COLUMNS)
                     for name in source["source_files"]], ignore_index=True)
    fixed = Q.baue(raw, core[Q.DEP_COLUMNS])
    del raw
    for col in Q.MERKMALE:
        core[col] = fixed[col].to_numpy()
    del fixed
    gc.collect()
    return core.loc[core.hat_aobt.eq(1) & core.y.notna()].copy()


train, valid = table(args.core_train), table(args.core_valid)
assert train.MVT_ID_mvt.is_unique and valid.MVT_ID_mvt.is_unique
assert set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))
base = pd.read_parquet(args.baseline_predictions).set_index("MVT_ID_mvt", verify_integrity=True)
assert set(base.index) == set(valid.MVT_ID_mvt)
base = base.loc[valid.MVT_ID_mvt]
np.testing.assert_array_equal(base.y.to_numpy(), valid.y.to_numpy())
for col in H.KAT:
    cats = pd.unique(train[col].astype(str))
    train[col] = pd.Categorical(train[col].astype(str), categories=cats)
    valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)
features = H.MERKMALE_HAUPT+Q.MERKMALE
x, xv = train[features].copy(), valid[features].copy()
for col in U.CORRECTED_COLUMNS:
    x[col], xv[col] = train["tu_"+col].to_numpy(), valid["tu_"+col].to_numpy()
target = train.y.to_numpy()-train.ersatz_s.to_numpy()
valid_target = valid.y.to_numpy()-valid.ersatz_s.to_numpy()
before = time.time()
print("FIT correct-clock queue on corrected-time baseline", len(features), flush=True)
model = lgb.LGBMRegressor(objective="l2", n_estimators=1400,
    learning_rate=.03, num_leaves=255, min_child_samples=50,
    subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1,
    n_jobs=args.jobs, random_state=177)
model.fit(x, target, eval_set=[(xv, valid_target)], eval_metric="rmse",
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(300)])
out = valid[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "y"]].copy()
out["units_plus_queue"] = np.maximum(valid.ersatz_s.to_numpy()+model.predict(xv), 300)
out = out.merge(base[["v44_main","units_replaced"]], left_on="MVT_ID_mvt", right_index=True, validate="one_to_one")
report = {"validation":"single-LightGBM main component, not full ensemble or leaderboard",
          "role":"exploratory screen" if pair == [1,7] else "locked temporal confirmation",
          "source_months":pair, "training_rows":len(train),
          "source_tables":[str(args.core_train),str(args.core_valid)],
          "baseline_predictions":str(args.baseline_predictions),
          "features":features, "seed":177, "best_iteration":model.best_iteration_,
          "seconds":time.time()-before, "rows":len(out), "scores":{}}
for name in ("v44_main", "units_replaced", "units_plus_queue"):
    e = (out[name]-out.y)**2
    report["scores"][name] = {"rmse":float(np.sqrt(e.mean())), "mse":float(e.mean()),
                               "top5_sse_share":float(e.nlargest(5).sum()/e.sum())}
report["incremental_rmse_gain"] = report["scores"]["units_replaced"]["rmse"]-report["scores"]["units_plus_queue"]["rmse"]
report["prespecified_main_rmse_gain_bar"] = .3
report["clears_screen_bar"] = report["incremental_rmse_gain"] >= .3
report["code_sha256"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__),Path(Q.__file__),Path(U.__file__),Path(H.__file__),Path(legacy_queue.__file__))}
args.output.mkdir(parents=True, exist_ok=True)
out.to_parquet(pred_path, index=False)
score_path.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
