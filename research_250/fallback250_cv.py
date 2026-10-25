"""Isolated, temporal fallback experiments toward RMSE 250.

No training targets cross the OUTER month-pair boundary. Raw feature references
are built separately on the ten fitting months and two validation months.
Both full core and small fallback feature tables are cached. Scored model bundles
are untouched. These are component experiments, not full-ensemble validation.
"""
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--pairs", default="1,7;2,8;3,9;4,10;5,11;6,12")
parser.add_argument("--variants", default="context,excess,gate")
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--main-screen", action="store_true")
args = parser.parse_args()
project = args.project.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS
import features_geometry as MG
import features_stand_rate as MSQ
from fid_gap_cv import month_of_file
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
C = importlib.import_module("features_reporting_context")
U = importlib.import_module("features_time_units")
variants = set(args.variants.split(","))
pairs = [tuple(map(int, p.split(","))) for p in args.pairs.split(";")]
assert all(len(p) == 2 and len(set(p)) == 2 and all(1 <= m <= 12 for m in p) for p in pairs)
base_features = H.MERKMALE_RUECK.copy()
context_features = base_features + C.MERKMALE
unit_features = base_features + [f for f in U.MERKMALE if f in ("tu_auslast",)]
all_features = unit_features + C.MERKMALE
assert len(context_features) == len(set(context_features))
paths = sorted((project / "data").glob("training_2025-*.parquet"))
assert len(paths) == 12 and {month_of_file(p) for p in paths} == set(range(1, 13))
geometry = MG.alle()

def fingerprint(selected):
    h = hashlib.sha256()
    for p in sorted(project.glob("*.py")) + [Path(C.__file__), Path(U.__file__), Path(__file__)]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    for p in selected:
        h.update(str(p).encode())
        h.update(str((p.stat().st_size, p.stat().st_mtime_ns)).encode())
    return h.hexdigest()[:16]

def table(selected):
    key = fingerprint(selected)
    cached = output / ("fallback_" + key + ".parquet")
    if cached.exists():
        return pd.read_parquet(cached)
    started = time.time()
    print("BUILD", [month_of_file(p) for p in selected], flush=True)
    full = H.baue([str(p) for p in selected], geometry)
    context = C.baue(full[C.RAW_COLUMNS])
    assert context.index.equals(full.index)
    raw = pd.concat([pd.read_parquet(p, columns=U.RAW_COLUMNS) for p in selected],
                    ignore_index=True)
    fixed = U.corrected_values(raw, full[U.DEP_COLUMNS])
    del raw
    core_columns = list(dict.fromkeys(
        [f for f in H.MERKMALE_HAUPT + base_features if f not in MSQ.MERKMALE]
        + ["MVT_ID_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt",
           "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "y"]))
    for f in C.MERKMALE:
        full[f] = context[f].to_numpy()
    for f in U.CORRECTED_COLUMNS:
        full["tu_" + f] = fixed[f].to_numpy()
    del context, fixed
    full[core_columns + C.MERKMALE + U.MERKMALE].to_parquet(
        output / ("core_" + key + ".parquet"), index=False)
    manifest = {"source_months": [month_of_file(p) for p in selected],
                "source_files": [p.name for p in selected],
                "core_file": "core_" + key + ".parquet",
                "cache_fingerprint": key,
                "split_definition": "source-file start month; movement timestamps can cross a file boundary"}
    (output / ("core_" + key + ".json")).write_text(json.dumps(manifest, indent=2))
    mask = full.hat_aobt.eq(0) & full.y.notna()
    # Stand labels and timestamps are retained for encoding and diagnostics only.
    keep = list(dict.fromkeys([f for f in base_features if f not in MSQ.MERKMALE]
                              + ["MVT_ID_mvt", "FLIGHT_ID_mvt", "STAND_mvt",
                                 "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                                 "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "y",
                                 "monat", "hat_aobt"]))
    small = full.loc[mask, keep + C.MERKMALE + U.MERKMALE].copy()
    del full
    gc.collect()
    small.to_parquet(cached, index=False)
    print("BUILT", len(small), "rows", round(time.time()-started, 1), "seconds", flush=True)
    return small

def prepare(train, valid):
    train, valid = train.copy(), valid.copy()
    train["z"] = (train.y > H.SCHWELLE_EXTREM).astype(int)
    mapping = MSQ.passe_an(train)
    train = MSQ.ausser_faltung(train)
    valid = MSQ.wende_an(mapping, valid)
    for col in H.KAT:
        if col in train:
            cats = pd.unique(train[col].astype(str))
            train[col] = pd.Categorical(train[col].astype(str), categories=cats)
            valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)
    return train, valid

def reg(rounds=300, leaves=15, minimum=20, **kw):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05,
                            num_leaves=leaves, min_child_samples=minimum,
                            verbose=-1, n_jobs=args.jobs, random_state=0, **kw)

def fit_parts(train, valid, features):
    train, valid = prepare(train, valid)
    x, xv = train[features], valid[features]
    classifier = lgb.LGBMClassifier(objective="binary", n_estimators=600,
                    learning_rate=.05, num_leaves=31, min_child_samples=20,
                    verbose=-1, n_jobs=args.jobs, random_state=0)
    classifier.fit(x, train.z)
    normal = reg(400, 31).fit(x[train.z.eq(0)], train.loc[train.z.eq(0), "y"])
    height = reg(300, 15, 10).fit(x[train.z.eq(1)], train.loc[train.z.eq(1), "y"])
    return pd.DataFrame({
        "normal": normal.predict(xv),
        "height": (1-H.ANTEIL_HM)*valid.versp_min.to_numpy()*60
                  + H.ANTEIL_HM*height.predict(xv),
        "p": classifier.predict_proba(xv)[:, 1],
    }, index=valid.index)

def combine(parts):
    q = parts.p.to_numpy() ** H.HOCH
    return np.maximum((1-q)*parts.normal.to_numpy() + q*parts.height.to_numpy(), 0)

def predict_excess(train, valid, features):
    train, valid = prepare(train, valid)
    x, xv = train[features], valid[features]
    y = train.y.to_numpy()
    cap = float(H.SCHWELLE_EXTREM)
    # E[Y|X] = E[min(Y, cap)|X] + E[max(Y-cap, 0)|X].
    # Both estimators use all training rows; there is no classification threshold
    # applied to a prediction and no inference from validation labels.
    lower = reg(400, 31, 50).fit(x, np.minimum(y, cap))
    upper = reg(300, 15, 50).fit(x, np.maximum(y-cap, 0))
    return np.maximum(lower.predict(xv) + np.maximum(upper.predict(xv), 0), 0)

def gate_frame(raw, parts):
    # Small fixed feature set limits capacity for the rare, high-impact examples.
    cols = ["ADEP_mvt", "versp_min", "stunde", "tag_versch", "id_abw",
            "id_abw2", "ank_med_30min", "ank_med_90min", "tu_auslast"] + C.MERKMALE
    frame = raw[cols].copy()
    for col in ["normal", "height", "p"]:
        frame["gate_" + col] = parts[col].to_numpy()
    frame["gate_delta"] = parts.height.to_numpy() - parts.normal.to_numpy()
    return frame

def predict_gate(train, valid, valid_parts, features):
    # All gate-training anchors are label-out-of-fold. In each inner split the
    # stand target encoding is also re-fitted without those prediction targets.
    oof = pd.DataFrame(index=train.index, columns=["normal", "height", "p"], dtype=float)
    available = set(train.monat.unique())
    for first in range(1, 7):
        months = available & {first, first+6}
        if not months:
            continue
        hold = train.monat.isin(months)
        oof.loc[hold] = fit_parts(train.loc[~hold], train.loc[hold], features)
    assert oof.notna().all().all()
    x = gate_frame(train, oof)
    xv = gate_frame(valid, valid_parts)
    cats = pd.unique(x.ADEP_mvt.astype(str))
    x["ADEP_mvt"] = pd.Categorical(x.ADEP_mvt.astype(str), categories=cats)
    xv["ADEP_mvt"] = pd.Categorical(xv.ADEP_mvt.astype(str), categories=cats)
    delta = oof.height.to_numpy()-oof.normal.to_numpy()
    useful = np.abs(delta) >= 60
    # This weighted regression is algebraically the actual squared prediction loss:
    # (y - normal - q*delta)^2 = delta^2*((y-normal)/delta - q)^2.
    target = (train.y.to_numpy()[useful]-oof.normal.to_numpy()[useful])/delta[useful]
    weight = delta[useful]**2
    weight /= weight.mean()
    model = reg(120, 7, 200, reg_lambda=100).fit(x.loc[useful], target, sample_weight=weight)
    q = np.clip(model.predict(xv), 0, 1)
    gate = np.maximum(valid_parts.normal.to_numpy()
                      + q*(valid_parts.height.to_numpy()-valid_parts.normal.to_numpy()), 0)
    return gate

def summarize(frame, prediction):
    errors = (prediction-frame.y.to_numpy())**2
    order = np.argsort(errors)[::-1]
    return {"rmse": float(np.sqrt(errors.mean())), "mse": float(errors.mean()),
            "top5_sse_share": float(errors[order[:5]].sum()/errors.sum()),
            "rmse_without_5_largest_errors": float(np.sqrt(np.delete(errors, order[:5]).mean()))}

all_frames = []
started = time.time()
for pair in pairs:
    train = table([p for p in paths if month_of_file(p) not in pair]).reset_index(drop=True)
    valid = table([p for p in paths if month_of_file(p) in pair]).reset_index(drop=True)
    # File membership defines the outer fold. Two records in the March/September
    # source files have February/August movement timestamps three seconds before
    # midnight. Check record identity, not timestamp-month disjointness.
    assert set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))
    print("FIT", pair, len(train), len(valid), flush=True)
    base_parts = fit_parts(train, valid, base_features)
    predictions = {"v44_fallback": combine(base_parts)}
    predictions["correct_time_added"] = combine(fit_parts(train, valid, unit_features))
    expanded_parts = None
    if "context" in variants or "gate" in variants:
        expanded_parts = fit_parts(train, valid, all_features)
        predictions["reporting_context"] = combine(expanded_parts)
    if "excess" in variants:
        predictions["excess_expectation"] = predict_excess(train, valid, all_features)
    if "gate" in variants:
        predictions["crossfit_mse_gate"] = predict_gate(train, valid, expanded_parts, all_features)
        predictions["half_context_half_gate"] = .5*predictions["reporting_context"] + .5*predictions["crossfit_mse_gate"]
    frame = valid[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "y", "monat"]].copy()
    for name, prediction in predictions.items():
        frame[name] = prediction
    frame["fold"] = str(pair)
    all_frames.append(frame)
    result = {name: summarize(valid, prediction) for name, prediction in predictions.items()}
    for name in result:
        result[name]["mse_gain_vs_v44"] = result["v44_fallback"]["mse"]-result[name]["mse"]
    print("RESULT", json.dumps({"pair":pair, "scores":result}), flush=True)
    frame.to_parquet(output / ("predictions_" + "_".join(map(str, pair)) + ".parquet"), index=False)
    (output / ("result_" + "_".join(map(str, pair)) + ".json")).write_text(json.dumps(result, indent=2))
    del train, valid, base_parts, expanded_parts
    gc.collect()

pooled = pd.concat(all_frames, ignore_index=True)
metrics = {name: summarize(pooled, pooled[name].to_numpy()) for name in predictions}
baseline_mse = metrics["v44_fallback"]["mse"]
for name in metrics:
    gain = baseline_mse-metrics[name]["mse"]
    hypothetical_mse = 272.334**2 - (5290/344841)*gain
    metrics[name]["mse_gain_vs_v44"] = gain
    metrics[name]["mechanical_overall_rmse_if_transfer"] = float(np.sqrt(max(hypothetical_mse, 0)))
report = {"outer_pairs":pairs, "seed":0, "jobs":args.jobs, "baseline_features":base_features,
          "candidate_features":all_features, "unit_features":unit_features,
          "enabled_variants":sorted(variants-{ "" }), "metrics":metrics, "seconds":time.time()-started,
          "validation":"fallback-only temporal CV; full ensemble not fitted; mechanical score is NOT a leaderboard forecast",
          "cache_fingerprint":fingerprint(paths)}
(output / "pooled.json").write_text(json.dumps(report, indent=2))
print("POOLED", json.dumps(report), flush=True)

if args.main_screen:
    for pair in pairs:
        learning = [p for p in paths if month_of_file(p) not in pair]
        held = [p for p in paths if month_of_file(p) in pair]
        train = pd.read_parquet(output / ("core_" + fingerprint(learning) + ".parquet"))
        valid = pd.read_parquet(output / ("core_" + fingerprint(held) + ".parquet"))
        train = train[train.hat_aobt.eq(1)].copy()
        valid = valid[valid.hat_aobt.eq(1)].copy()
        for col in H.KAT:
            cats = pd.unique(train[col].astype(str))
            train[col] = pd.Categorical(train[col].astype(str), categories=cats)
            valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)
        target = train.y.to_numpy() - train.ersatz_s.to_numpy()
        valid_target = valid.y.to_numpy() - valid.ersatz_s.to_numpy()
        names = ["v44_main", "units_replaced", "units_added", "units_and_context"]
        main_results = {}
        main_predictions = valid[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "y"]].copy()
        for name in names:
            columns = H.MERKMALE_HAUPT.copy()
            if name in ("units_added", "units_and_context"):
                columns += U.MERKMALE
            if name == "units_and_context":
                columns += C.MERKMALE
            x, xv = train[columns].copy(), valid[columns].copy()
            if name == "units_replaced":
                for col in U.CORRECTED_COLUMNS:
                    x[col] = train["tu_" + col].to_numpy()
                    xv[col] = valid["tu_" + col].to_numpy()
            before = time.time()
            print("MAIN FIT", pair, name, len(columns), flush=True)
            model = lgb.LGBMRegressor(objective="l2", n_estimators=1400,
                learning_rate=.03, num_leaves=255, min_child_samples=50,
                subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1,
                n_jobs=args.jobs, random_state=177)
            model.fit(x, target, eval_set=[(xv, valid_target)], eval_metric="rmse",
                      callbacks=[lgb.early_stopping(80, verbose=False),
                                 lgb.log_evaluation(300)])
            prediction = np.maximum(valid.ersatz_s.to_numpy() + model.predict(xv), 300)
            main_predictions[name] = prediction
            main_results[name] = summarize(valid, prediction)
            main_results[name]["best_iteration"] = model.best_iteration_
            main_results[name]["seconds"] = time.time()-before
            print("MAIN RESULT", name, json.dumps(main_results[name]), flush=True)
            del x, xv, model
            gc.collect()
        main_predictions.to_parquet(output / ("main_predictions_" + "_".join(map(str,pair)) + ".parquet"), index=False)
        (output / ("main_result_" + "_".join(map(str,pair)) + ".json")).write_text(json.dumps(main_results, indent=2))
        print("MAIN SUMMARY", pair, json.dumps(main_results), flush=True)
        del train, valid
        gc.collect()
