"""Main-branch target views, blended against the saved v45 full-ensemble holdout.

Pre-specified 12.09.2026, before any fit. No feature build: the release run's cached
v45-schema tables are reused, so inputs match the scored v45 schema exactly.

Folds
  outer  months_2_3_4_5_6_8_9_10_11_12 -> months_1_7   v45 full-ensemble holdout exists
  inner  months_2_4_5_6_8_10_11_12     -> months_3_9   confirmation (the release used this
                                                        split only to select LightGBM length)

Views, one LightGBM each: v45 main recipe (lr .03, 255 leaves, min 50, subsample .8,
colsample .8, seed 177), fixed 1,106 rounds as selected by the release on inner data,
no early stopping on any held-out fold, 91 v45 main features, prediction floor 300 s.
  rest     y - ersatz_s          the v45 target, the matched component
  direct   y
  eobt     y - ersatz_eobt_s     rows without that anchor keep the rest prediction
  sched    y - versp_min*60
  median4  y - anker_median      median of the AOBT/EOBT/LOBT/IOBT taxi anchors

Control: an extra rest model blended into v45. Adding one more ordinary LightGBM may
help by itself, so a view counts as complementary only if it beats that control.

Decision rule at the pre-specified weight w = 0.2, fixed now. A view advances to
ensemble integration only if (i) blending it into the v45 ensemble improves Jan/Jul
main MSE, (ii) that gain exceeds the rest control's, (iii) blending it into the matched
rest component also improves March/September main MSE, and (iv) at least 7 of 10
airports improve on Jan/Jul. The in-sample optimal weight is reported, never used.

Replication sanity: the outer rest view must land within 1.0 s of the research_250
single-LightGBM Jan/Jul result 223.698 s, otherwise the run stops.

Component screen only. No submission, no upload, no change to the frozen release.
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
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--run", default=Path(".cache/clock_queue_release_20260912"), type=Path)
parser.add_argument("--jobs", type=int, default=8)
parser.add_argument("--rounds", type=int, default=1106)
parser.add_argument("--folds", default="outer,inner")
args = parser.parse_args()
project, output = args.project.resolve(), args.output.resolve()
run = args.run if args.run.is_absolute() else (project / args.run)
output.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_250 import candidate_features as B

FEATURES = B.main_features(H.MERKMALE_HAUPT)
assert len(FEATURES) == 91 == len(set(FEATURES))
FOLDS = {"outer": ("months_2_3_4_5_6_8_9_10_11_12", "months_1_7"),
         "inner": ("months_2_4_5_6_8_10_11_12", "months_3_9")}
VIEWS = ["rest", "direct", "eobt", "sched", "median4"]
FIXED_W, GATE_W, FLOOR = (0.1, 0.2, 0.3), 0.2, 300.0
REFERENCE_OUTER_REST, REFERENCE_TOLERANCE = 223.697629244, 1.0
MAIN_ROWS_OFFICIAL, ROWS_OFFICIAL, V45_OFFICIAL = 339551, 344841, 269.4753
COLUMNS = list(dict.fromkeys(FEATURES + ["ersatz_s", "ersatz_eobt_s", "versp_min", "anker_median",
                                         "hat_aobt", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y", "ADEP_mvt"]))
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__).resolve(), project / "main_model.py",
                  project / "research_250/candidate_features.py")}


def load(name):
    d = pd.read_parquet(run / "data" / f"{name}.parquet", columns=COLUMNS)
    d = d[d.hat_aobt.eq(1) & d.y.notna()].reset_index(drop=True)
    assert d.MVT_ID_mvt.is_unique and np.isfinite(d.ersatz_s).all()
    return d


def align(train, valid):
    for col in H.KAT:
        if col in FEATURES:
            cats = pd.unique(train[col].astype(str))
            train[col] = pd.Categorical(train[col].astype(str), categories=cats)
            valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)


def anchor(d, view):
    return {"rest": lambda: d.ersatz_s.to_numpy(float),
            "direct": lambda: np.zeros(len(d)),
            "eobt": lambda: d.ersatz_eobt_s.to_numpy(float),
            "sched": lambda: d.versp_min.to_numpy(float)*60,
            "median4": lambda: d.anker_median.to_numpy(float)}[view]()


def fit_view(train, valid, view, fallback=None):
    a_tr, a_va = anchor(train, view), anchor(valid, view)
    ok, has = np.isfinite(a_tr), np.isfinite(a_va)
    model = lgb.LGBMRegressor(objective="l2", n_estimators=args.rounds, learning_rate=.03,
                              num_leaves=255, min_child_samples=50, subsample=.8, subsample_freq=1,
                              colsample_bytree=.8, verbose=-1, n_jobs=args.jobs, random_state=177)
    x = train[FEATURES] if ok.all() else train.loc[ok, FEATURES]
    model.fit(x, train.y.to_numpy(float)[ok]-a_tr[ok])
    del x
    gc.collect()
    prediction = np.full(len(valid), np.nan)
    xv = valid[FEATURES] if has.all() else valid.loc[has, FEATURES]
    prediction[has] = a_va[has] + model.predict(xv)
    if not has.all():
        prediction[~has] = fallback[~has]
    del model, xv
    gc.collect()
    return np.maximum(prediction, FLOOR), int(ok.sum()), int((~has).sum())


def mse(p, y):
    return float(np.mean((p-y)**2))


def blend(reference, candidate, y, airports):
    se_ref = (reference-y)**2
    worst = se_ref >= np.quantile(se_ref, .99)
    e0, e1 = reference-y, candidate-y
    d = e0-e1
    w_star = float(np.clip(np.sum(e0*d)/np.sum(d*d), 0, 1)) if np.sum(d*d) > 0 else 0.0
    out = {"reference_mse": float(se_ref.mean()), "candidate_mse": mse(candidate, y),
           "error_correlation": float(np.corrcoef(e0, e1)[0, 1]),
           "w_star_in_sample_not_used": w_star,
           "mse_gain_at_w_star_in_sample_not_used": float(se_ref.mean()-mse((1-w_star)*reference+w_star*candidate, y))}
    for w in FIXED_W:
        out[f"mse_gain_w{w}"] = float(se_ref.mean()-mse((1-w)*reference+w*candidate, y))
    se_gate = ((1-GATE_W)*reference+GATE_W*candidate-y)**2
    gains = {a: float(se_ref[airports == a].mean()-se_gate[airports == a].mean()) for a in np.unique(airports)}
    out["airport_mse_gains_at_gate_w"] = gains
    out["airports_improving_at_gate_w"] = int(sum(g > 0 for g in gains.values()))
    out["baseline_worst_1pct_mse_gain_at_gate_w"] = float(se_ref[worst].mean()-se_gate[worst].mean())
    out["other_99pct_mse_gain_at_gate_w"] = float(se_ref[~worst].mean()-se_gate[~worst].mean())
    return out


report = {"validation": "single-LightGBM target views blended into the saved v45 Jan/Jul full ensemble; "
                        "not a new full ensemble and not a leaderboard forecast",
          "features": FEATURES, "rounds": args.rounds, "seed": 177, "gate_w": GATE_W,
          "fixed_w": list(FIXED_W), "code_sha256": CODE, "folds": {}}
started = time.time()
for fold in args.folds.split(","):
    train_name, valid_name = FOLDS[fold]
    train, valid = load(train_name), load(valid_name)
    assert set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))
    align(train, valid)
    y = valid.y.to_numpy(float)
    airports = valid.ADEP_mvt.astype(str).to_numpy()
    print("FOLD", fold, train_name, len(train), "->", valid_name, len(valid), flush=True)
    predictions, views = {}, {}
    for view in VIEWS:
        before = time.time()
        predictions[view], fit_rows, missing = fit_view(train, valid, view, predictions.get("rest"))
        views[view] = {"rmse": float(np.sqrt(mse(predictions[view], y))), "mse": mse(predictions[view], y),
                       "fit_rows": fit_rows, "valid_rows_without_anchor": missing,
                       "seconds": time.time()-before}
        print("VIEW", fold, view, json.dumps(views[view]), flush=True)
        if fold == "outer" and view == "rest":
            gap = abs(views[view]["rmse"]-REFERENCE_OUTER_REST)
            views[view]["replication_gap_seconds"] = gap
            if gap > REFERENCE_TOLERANCE:
                (output / "replication_failed.json").write_text(json.dumps(views, indent=2))
                sys.exit(f"Outer rest view {views[view]['rmse']:.3f} is {gap:.3f} s from {REFERENCE_OUTER_REST}")
    frame = valid[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y"]].copy()
    frame["ADEP_mvt"] = airports
    for view in VIEWS:
        frame[view] = predictions[view]
    frame.to_parquet(output / f"predictions_{fold}.parquet", index=False)
    errors = np.vstack([predictions[v]-y for v in VIEWS])
    fold_report = {"train": train_name, "valid": valid_name, "valid_main_rows": int(len(valid)),
                   "views": views,
                   "view_error_correlations": {a: {b: float(np.corrcoef(errors[i], errors[j])[0, 1])
                                                   for j, b in enumerate(VIEWS)} for i, a in enumerate(VIEWS)},
                   "blend_into_rest_component": {v: blend(predictions["rest"], predictions[v], y, airports)
                                                 for v in VIEWS[1:]}}
    if fold == "outer":
        ensemble = (pd.read_parquet(run / "candidate/holdout_predictions.parquet")
                    .set_index("MVT_ID_mvt", verify_integrity=True).loc[valid.MVT_ID_mvt])
        assert (ensemble.hat_aobt.to_numpy() == 1).all()
        np.testing.assert_allclose(ensemble.y.to_numpy(float), y)
        v45 = ensemble.prediction.to_numpy(float)
        fold_report["v45_ensemble_main_rmse"] = float(np.sqrt(mse(v45, y)))
        fold_report["blend_into_v45_ensemble"] = {v: blend(v45, predictions[v], y, airports) for v in VIEWS}
    report["folds"][fold] = fold_report
    (output / f"result_{fold}.json").write_text(json.dumps(fold_report, indent=2))
    del train, valid, predictions, errors
    gc.collect()

if {"outer", "inner"} <= set(report["folds"]):
    outer, inner = report["folds"]["outer"], report["folds"]["inner"]
    key = f"mse_gain_w{GATE_W}"
    control = outer["blend_into_v45_ensemble"]["rest"][key]
    report["decision"] = {}
    for view in VIEWS[1:]:
        gain = outer["blend_into_v45_ensemble"][view][key]
        checks = {"i_improves_v45_jan_jul": gain > 0,
                  "ii_beats_rest_control": gain > control,
                  "iii_improves_component_mar_sep": inner["blend_into_rest_component"][view][key] > 0,
                  "iv_at_least_7_airports": outer["blend_into_v45_ensemble"][view]["airports_improving_at_gate_w"] >= 7}
        overall_gain = gain*MAIN_ROWS_OFFICIAL/ROWS_OFFICIAL
        report["decision"][view] = {**{k: bool(v) for k, v in checks.items()},
                                    "advance": bool(all(checks.values())),
                                    "jan_jul_main_mse_gain_at_gate_w": gain,
                                    "rest_control_gain_at_gate_w": control,
                                    "mechanical_overall_rmse_if_transfer_from_v45_not_forecast":
                                        float(np.sqrt(max(V45_OFFICIAL**2-overall_gain, 0)))}
report["seconds"] = time.time()-started
(output / "summary.json").write_text(json.dumps(report, indent=2))
print("SUMMARY", json.dumps(report.get("decision", {}), indent=2), flush=True)
