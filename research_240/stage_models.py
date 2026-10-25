"""Per-stage taxi-out models on the v45 release-cache January/July split (diagnostic, no candidate).

Run outside the repository while another agent worked in it, then moved into research_240/.
No leaderboard feedback, no 2026 label; outputs go to --output.

Stages per departure with AOBT_3 (main lane):
  S1 reporting gap   g = y - (MVT - AOBT_3)                                        measured
  S2 reference       r = EUROCONTROL-style 10th percentile of p = MVT - AOBT_3 per airport, stand and
                         runway (at least 10 flights at or below it, 0 < p <= 120 min), from the learning
                         months only, with airport-runway and airport fallbacks                    estimated
  S3 queue and delay e = p - r                                                     measured minus estimate
Models, one LightGBM each with fixed settings (600 rounds, learning rate 0.05, 127 leaves, seed 0):
  S1 on the 91 v45 main features (the main-lane target)
  S2 on airport, stand, runway, aircraft type, wake category, operator, market segment and distance
  S3 on runway and airport traffic counts, arrival taxi medians, hour, minute of day, weekday, month and
     delay context (no feature of the flight's own AOBT_3)
Scoring on January/July: S1 against measured g; S3 against e_test = p - r (learning-month reference);
S2 against the test months' own reference for groups with enough flights; and p_hat = r_hat + e_hat
against measured p, the physical taxi a flight without AOBT_3 would need, compared with a learning-month
airport-hour median of p. Errors by airport, hour band and congestion (runway departures in the previous
10 min); gain importances per model.

Run:
  python3 -B research_240/stage_models.py --project <repository> --output <folder>
"""
import argparse
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
from research_250 import candidate_features as B

RELEASE = project / ".cache/clock_queue_release_20260912/data"
S1_FEATURES = B.main_features(H.MERKMALE_HAUPT)
S2_FEATURES = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
               "MARKET_SEGMENT_flt", "strecke_m"]
S3_FEATURES = ["ADEP_mvt", "RUNWAY_mvt", "qt_bahn_vor_10", "qt_bahn_nach_10", "qt_bahn_vor_30", "qt_bahn_nach_30",
               "qt_bahn_luecke_vor", "qt_bahn_luecke_nach", "qt_bahn_ank_vor_10", "qt_bahn_anteil_60", "qt_flh_vor_10",
               "qt_bahn_rang_10", "tu_auslast", "ank_med_30min", "ank_anz_30min", "ank_med_90min", "ank_anz_90min",
               "stunde", "minute_tag", "wtag", "monat", "lage_50", "lage_200", "versp_min"]
FINE, COARSE, AIRPORT = ["airport", "stand", "runway"], ["airport", "runway"], ["airport"]


def load(months):
    frame = pd.read_parquet(RELEASE / ("months_" + "_".join(map(str, months)) + ".parquet"))
    main = frame.loc[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)
    main["airport"], main["stand"], main["runway"] = (main.ADEP_mvt.astype(str), main.STAND_mvt.astype(str),
                                                      main.RUNWAY_mvt.astype(str))
    main["p"] = main.ersatz_s.to_numpy(float)
    main["g"] = main.y.to_numpy(float) - main.p
    main["valid"] = (main.p > 0) & (main.p <= 7200)
    return main


def reference_table(frame, min_below=10):
    base = frame[frame.valid]
    tables = {}
    for name, keys in (("fine", FINE), ("coarse", COARSE), ("airport", AIRPORT)):
        q = base.groupby(keys).p.quantile(.1).rename("ref")
        joined = base.join(q, on=keys)
        below = joined[joined.p <= joined.ref].groupby(keys).size().rename("below")
        tables[name] = pd.concat([q, below], axis=1)
        tables[name] = tables[name][tables[name].below.fillna(0) >= (min_below if name == "fine" else 1)]
    return tables


def lookup(frame, tables):
    ref = pd.Series(np.nan, index=frame.index)
    level = pd.Series("none", index=frame.index)
    for name, keys in (("fine", FINE), ("coarse", COARSE), ("airport", AIRPORT)):
        values = frame.join(tables[name].ref.rename("r_" + name), on=keys)["r_" + name]
        fill = ref.isna() & values.notna()
        ref[fill], level[fill] = values[fill], name
    return ref.to_numpy(float), level.to_numpy()


def model():
    return lgb.LGBMRegressor(objective="l2", n_estimators=600, learning_rate=.05, num_leaves=127, min_child_samples=50,
                             subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=args.jobs,
                             random_state=0)


def metrics(truth, prediction, mask=None):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    ok = np.isfinite(truth) & np.isfinite(prediction) & (np.ones(len(truth), bool) if mask is None else mask)
    if ok.sum() < 30:
        return None
    err = prediction[ok] - truth[ok]
    var = np.var(truth[ok])
    return {"rows": int(ok.sum()), "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)), "r2": float(1 - np.mean(err ** 2) / var) if var > 0 else None}


def breakdown(frame, truth, prediction):
    hours = frame.stunde.to_numpy(float)
    busy = frame.qt_bahn_vor_10.to_numpy(float)
    return {"overall": metrics(truth, prediction),
            "by_airport": {a: metrics(truth, prediction, (frame.airport == a).to_numpy()) for a in sorted(frame.airport.unique())},
            "by_hour_band": {f"{lo:02d}-{hi:02d}": metrics(truth, prediction, (hours >= lo) & (hours < hi))
                             for lo, hi in ((0, 6), (6, 10), (10, 14), (14, 18), (18, 24))},
            "by_runway_departures_prev_10min": {label: metrics(truth, prediction, cond(busy)) for label, cond in
                                                (("0", lambda b: b == 0), ("1-2", lambda b: (b >= 1) & (b <= 2)),
                                                 ("3-5", lambda b: (b >= 3) & (b <= 5)), ("6+", lambda b: b >= 6))}}


def importance(fitted, features, top=10):
    gain = fitted.booster_.feature_importance("gain")
    order = np.argsort(gain)[::-1][:top]
    return {features[i]: round(float(gain[i] / gain.sum()), 4) for i in order}


started = time.time()
train, test = load([2, 3, 4, 5, 6, 8, 9, 10, 11, 12]), load([1, 7])
assert set(train.MVT_ID_mvt).isdisjoint(set(test.MVT_ID_mvt))
tables = reference_table(train)
train["r"], train["r_level"] = lookup(train, tables)
test["r"], test["r_level"] = lookup(test, tables)
train["e"], test["e"] = train.p - train.r, test.p - test.r
report = {"validation": "diagnostic stage models, v45 release-cache split (learn 2-6 and 8-12, test January/July), main lane",
          "train_rows": int(len(train)), "test_rows": int(len(test)),
          "reference_level_share_test": {k: float(v) for k, v in pd.Series(test.r_level).value_counts(normalize=True).items()},
          "stages": {}}

print("S1 reporting gap", flush=True)
s1 = model().fit(train[S1_FEATURES], train.g)
g_hat = s1.predict(test[S1_FEATURES])
report["stages"]["S1_reporting_gap"] = {"target_median_test_s": float(np.median(test.g)), "importance": importance(s1, S1_FEATURES),
                                        **breakdown(test, test.g, g_hat)}

fit_rows = train.valid.to_numpy() & np.isfinite(train.r.to_numpy())
print("S2 reference", flush=True)
s2 = model().fit(train.loc[fit_rows, S2_FEATURES], train.loc[fit_rows, "r"])
r_hat = s2.predict(test[S2_FEATURES])
test_tables = reference_table(test)
test_ref = test.join(test_tables["fine"].ref.rename("r_test"), on=FINE)["r_test"].to_numpy(float)
report["stages"]["S2_reference"] = {
    "note": "scored against the test months' own fine reference where it is valid; lookup = learning-month reference",
    "model": breakdown(test, test_ref, r_hat), "lookup": breakdown(test, test_ref, test.r.to_numpy(float)),
    "importance": importance(s2, S2_FEATURES)}

print("S3 queue and delay", flush=True)
s3 = model().fit(train.loc[fit_rows, S3_FEATURES], train.loc[fit_rows, "e"])
e_hat = s3.predict(test[S3_FEATURES])
valid_test = test.valid.to_numpy()
report["stages"]["S3_queue_and_delay"] = {"target_median_test_s": float(np.nanmedian(test.e[valid_test])),
                                          "importance": importance(s3, S3_FEATURES),
                                          **breakdown(test[valid_test], test.e[valid_test], e_hat[valid_test])}

p_hat = r_hat + e_hat
naive = test.join(train[train.valid].groupby(["airport", "stunde"]).p.median().rename("p_naive"), on=["airport", "stunde"])["p_naive"]
report["physical_taxi_without_aobt"] = {
    "note": "p_hat = S2 + S3 against measured MVT - AOBT_3 on valid test flights; baseline = learning-month airport-hour median",
    "stage_models": breakdown(test[valid_test], test.p[valid_test], p_hat[valid_test]),
    "airport_hour_median": breakdown(test[valid_test], test.p[valid_test], naive.to_numpy(float)[valid_test])}
report["seconds"] = time.time() - started
(output / "stage_models.json").write_text(json.dumps(report, indent=2))

fmt = lambda m: "-" if m is None else f"RMSE {m['rmse']:.0f}  MAE {m['mae']:.0f}  R2 {m['r2']:.2f}"
print("S1 reporting gap:", fmt(report["stages"]["S1_reporting_gap"]["overall"]))
print("S2 reference model:", fmt(report["stages"]["S2_reference"]["model"]["overall"]), "| lookup:", fmt(report["stages"]["S2_reference"]["lookup"]["overall"]))
print("S3 queue and delay:", fmt(report["stages"]["S3_queue_and_delay"]["overall"]))
print("physical taxi without AOBT, stage models:", fmt(report["physical_taxi_without_aobt"]["stage_models"]["overall"]),
      "| airport-hour median:", fmt(report["physical_taxi_without_aobt"]["airport_hour_median"]["overall"]))
for stage in ("S1_reporting_gap", "S3_queue_and_delay"):
    rows = report["stages"][stage]["by_airport"]
    worst = sorted(((a, m["rmse"]) for a, m in rows.items() if m), key=lambda t: -t[1])[:4]
    print(f"{stage} worst airports by RMSE:", [(a, round(v)) for a, v in worst])
print("S3 by congestion:", {k: (None if v is None else round(v["rmse"])) for k, v in report["stages"]["S3_queue_and_delay"]["by_runway_departures_prev_10min"].items()})
print("importance S3:", report["stages"]["S3_queue_and_delay"]["importance"])
print(f"seconds {report['seconds']:.0f}")
