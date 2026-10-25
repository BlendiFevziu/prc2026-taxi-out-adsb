"""Small, fixed reporting-feature ablations on independently built core tables.

Use fallback250_cv.py to build each outer split. No raw ID or timestamp is a model
column. The whole departure reference is used before selecting the fallback group.
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
parser.add_argument("--core-train", required=True, type=Path)
parser.add_argument("--core-valid", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--jobs", type=int, default=8)
args = parser.parse_args()
for key in ("project", "core_train", "core_valid", "output"):
    setattr(args, key, getattr(args, key).resolve())
os.environ.setdefault("MPLCONFIGDIR", str(args.output / "matplotlib"))
sys.path.insert(0, str(args.project.resolve()))
os.chdir(args.project)
import main_model as H  # torch before LightGBM on macOS
import features_stand_rate as SQ
import lightgbm as lgb
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features_record_time as R


def get_table(path):
    core = pd.read_parquet(path)
    feature = R.baue(core[R.RAW_COLUMNS])
    mask = core.hat_aobt.eq(0) & core.y.notna()
    small = core.loc[mask].copy()
    for col in R.MERKMALE:
        small[col] = feature.loc[mask, col].to_numpy()
    return small.reset_index(drop=True)


def prepare(train, valid):
    train, valid = train.copy(), valid.copy()
    train["z"] = train.y.gt(H.SCHWELLE_EXTREM).astype(int)
    mapping = SQ.passe_an(train)
    train, valid = SQ.ausser_faltung(train), SQ.wende_an(mapping, valid)
    for col in H.KAT:
        if col in train:
            cats = pd.unique(train[col].astype(str))
            train[col] = pd.Categorical(train[col].astype(str), categories=cats)
            valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)
    return train, valid


def fit(train, valid, features):
    x, xv = train[features], valid[features]
    shared = dict(learning_rate=.05, verbose=-1, n_jobs=args.jobs, random_state=0)
    classifier = lgb.LGBMClassifier(objective="binary", n_estimators=600,
                     num_leaves=31, min_child_samples=20, **shared)
    normal = lgb.LGBMRegressor(objective="l2", n_estimators=400,
                     num_leaves=31, min_child_samples=20, **shared)
    height = lgb.LGBMRegressor(objective="l2", n_estimators=300,
                     num_leaves=15, min_child_samples=10, **shared)
    classifier.fit(x, train.z)
    normal.fit(x[train.z.eq(0)], train.loc[train.z.eq(0), "y"])
    height.fit(x[train.z.eq(1)], train.loc[train.z.eq(1), "y"])
    q = classifier.predict_proba(xv)[:, 1] ** H.HOCH
    high = (1-H.ANTEIL_HM)*valid.versp_min.to_numpy()*60 + H.ANTEIL_HM*height.predict(xv)
    return np.maximum((1-q)*normal.predict(xv) + q*high, 0)


train, valid = get_table(args.core_train), get_table(args.core_valid)
assert set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))
train, valid = prepare(train, valid)
missing = ["rc_missing_share_15m", "rc_missing_run_size"]
base = H.MERKMALE_RUECK.copy()
unit = base + ["tu_auslast"]
variants = {"v44_fallback": base, "correct_time_added": unit,
            "missingness_pair": unit + missing,
            "record_time": unit + R.MERKMALE,
            "record_time_and_missingness": unit + R.MERKMALE + missing}
out = valid[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "y", "monat"]].copy()
report = {"validation": "exploratory fallback-only temporal ablation; not a leaderboard score",
          "learning_timestamp_months": sorted(map(int, train.monat.unique())),
          "validation_months": sorted(map(int, valid.monat.unique())),
          "split_definition": "source files supplied by the caller; distinct record IDs checked",
          "rows_train": len(train), "rows_valid": len(valid),
          "source_tables": [str(args.core_train.resolve()), str(args.core_valid.resolve())],
          "features": variants, "scores": {}}
for name, cols in variants.items():
    before = time.time()
    print("FIT", name, len(cols), flush=True)
    out[name] = fit(train, valid, cols)
    e = (out[name]-valid.y)**2
    report["scores"][name] = {"rmse": float(np.sqrt(e.mean())), "mse": float(e.mean()),
        "top5_sse_share": float(e.nlargest(5).sum()/e.sum()),
        "seconds": time.time()-before}
    report["scores"][name]["mse_gain"] = report["scores"]["v44_fallback"]["mse"]-e.mean()
    print("RESULT", name, json.dumps(report["scores"][name]), flush=True)
args.output.mkdir(parents=True, exist_ok=True)
suffix = "_".join(map(str, report["validation_months"]))
out.to_parquet(args.output / f"reporting_predictions_{suffix}.parquet", index=False)
report["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [Path(__file__), Path(R.__file__), Path(H.__file__), Path(SQ.__file__)]}
(args.output / f"reporting_result_{suffix}.json").write_text(json.dumps(report, indent=2))
