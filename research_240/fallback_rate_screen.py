"""Six-pair fallback screen: legacy auslast as a scale-free per-available-day rate.

Pre-specified 12.09.2026, before any fit. Motivation: legacy auslast counts departures over
+/-20.83 days, so its scale follows the table length (0.61 of training scale in the 2026
ranking file). The label-free sensitivity check showed that restoring the training scale
moves v45's scored fallback on 162 extreme-weight rows by a mean 1,745 s, 68 of the 72 rows
moving over 15 minutes at LIRF.

Variants, all on the fallback_anchor screen's cached tables (no builds):
  v44_fallback  exact v44/v45 recipe; gate: research_250/results/fallback_six_pair.json
  rate          legacy auslast replaced by features_auslast_rate.rate, same column name
  anchor        nt_anchor_normal; gate: that screen's result_<pair>.json
  anchor_rate   anchor on the rate tables

Decision rules, fixed now. A comparison passes only if, pooled over six pairs:
  (1) MSE gain over its reference is positive,
  (2) the gain survives removing its 100 most favourable rows,
  (3) at least 4 of 6 pairs gain,
  (4) January and July both gain inside pair 1_7.
rate is judged against v44_fallback; anchor_rate against anchor. Coverage spans come from
each table's source months. Fallback-only screen; not full ensemble; no submission.
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
parser.add_argument("--screen", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--bootstrap", type=int, default=2000)
args = parser.parse_args()
project, screen, output = args.project.resolve(), args.screen.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS
import features_stand_rate as MSQ
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_240 import features_auslast_rate as AR
from research_240 import features_neighbour_taxi as NT

BASE = list(H.MERKMALE_RUECK)
WITH_NT = BASE + NT.MERKMALE
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
NAMES = ["v44_fallback", "rate", "anchor", "anchor_rate"]
COMPARISONS = {"rate": "v44_fallback", "anchor_rate": "anchor"}
REFERENCE = {f["fold"]: f["baseline_rmse"] for f in json.loads(
    (project / "research_250/results/fallback_six_pair.json").read_text())["fallback"]["folds"]}
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__).resolve(), Path(AR.__file__).resolve(), Path(NT.__file__).resolve(),
                  project / "main_model.py", project / "features_stand_rate.py")}


def cached_tables():
    found = {}
    for side in sorted(screen.glob("fallback_nt_*.json")):
        months = tuple(sorted(json.loads(side.read_text())["source_months"]))
        if months in found:
            raise SystemExit(f"Two cached tables for months {months}")
        found[months] = side.with_suffix(".parquet")
    return found


def with_rate(frame, months):
    out = frame.copy()
    out["auslast"] = AR.rate(out.auslast.to_numpy(float), out.MVT_TIME_UTC_mvt,
                             AR.month_spans([(2025, m) for m in months]))
    assert np.isfinite(out.auslast).all()
    return out


def prepare(train, valid):
    """Identical to fallback_anchor_screen.prepare."""
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


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05,
                             num_leaves=leaves, min_child_samples=minimum, verbose=-1,
                             n_jobs=args.jobs, random_state=0)


def classifier():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05,
                              num_leaves=31, min_child_samples=20, verbose=-1,
                              n_jobs=args.jobs, random_state=0)


def fit(train, valid):
    """Returns (baseline-style prediction, anchored prediction, prepared valid)."""
    tr, va = prepare(train, valid)
    z0 = tr.z.eq(0).to_numpy()
    y = tr.y.to_numpy(float)
    versp = va.versp_min.to_numpy(float)*60
    q = classifier().fit(tr[BASE], tr.z).predict_proba(va[BASE])[:, 1] ** H.HOCH
    height = (1-H.ANTEIL_HM)*versp + H.ANTEIL_HM*reg(300, 15, 10).fit(tr.loc[~z0, BASE], y[~z0]).predict(va[BASE])
    normal = reg(400, 31, 20).fit(tr.loc[z0, BASE], y[z0]).predict(va[BASE])
    anchor_tr, anchor_va = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
    rows = z0 & np.isfinite(anchor_tr)
    offset = reg(400, 31, 20).fit(tr.loc[rows, WITH_NT], y[rows]-anchor_tr[rows])
    has = np.isfinite(anchor_va)
    anchored = normal.copy()
    anchored[has] = anchor_va[has] + offset.predict(va.loc[has, WITH_NT])
    return (np.maximum((1-q)*normal + q*height, 0), np.maximum((1-q)*anchored + q*height, 0), va)


tables, frames, gates, started = cached_tables(), [], [], time.time()
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    learn_months = tuple(sorted(set(range(1, 13))-set(pair)))
    train = pd.read_parquet(tables[learn_months]).reset_index(drop=True)
    valid = pd.read_parquet(tables[tuple(sorted(pair))]).reset_index(drop=True)
    base, anchor, va = fit(train, valid)
    rate, anchor_rate, va_rate = fit(with_rate(train, learn_months), with_rate(valid, pair))
    assert np.array_equal(va.MVT_ID_mvt.to_numpy(), va_rate.MVT_ID_mvt.to_numpy())
    yv = va.y.to_numpy(float)
    base_rmse = float(np.sqrt(np.mean((base-yv)**2)))
    anchor_gain = float(np.mean((base-yv)**2) - np.mean((anchor-yv)**2))
    expected = json.loads((screen / f"result_{key}.json").read_text())["nt_anchor_normal"]["mse_gain_vs_v44"]
    gate = {"pair": key, "baseline_rmse": base_rmse, "baseline_reference": REFERENCE[key],
            "anchor_gain": anchor_gain, "anchor_reference": expected}
    gate["passed"] = (abs(base_rmse-REFERENCE[key]) <= .01
                      and abs(anchor_gain-expected) <= 1e-6*max(1.0, abs(expected)))
    gates.append(gate)
    print("GATE", json.dumps(gate), flush=True)
    if not gate["passed"]:
        (output / "replication_failed.json").write_text(json.dumps(gates, indent=2))
        sys.exit(f"Replication gate failed on {key}")
    frame = va[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y", "nt_anchor"]].copy()
    frame["ADEP_mvt"] = va.ADEP_mvt.astype(str).to_numpy()
    for name, prediction in zip(NAMES, (base, rate, anchor, anchor_rate)):
        frame[name] = prediction
    frame["fold"] = key
    frame.to_parquet(output / f"predictions_{key}.parquet", index=False)
    print("PAIR", key, json.dumps({n: float(np.sqrt(np.mean((frame[n]-yv)**2))) for n in NAMES}), flush=True)
    frames.append(frame)
    del train, valid, va, va_rate
    gc.collect()

pooled = pd.concat(frames, ignore_index=True)
y = pooled.y.to_numpy(float)
folds = pooled.fold.to_numpy()
stamps = pd.to_datetime(pooled.MVT_TIME_UTC_mvt, utc=True)
raw_month = stamps.dt.month.to_numpy()
month = np.array([m if m in (int(f.split("_")[0]), int(f.split("_")[1])) else m % 12 + 1
                  for m, f in zip(raw_month, folds)])
day = (pooled.ADEP_mvt + "|" + stamps.dt.strftime("%Y-%m-%d")).to_numpy()
_, inverse = np.unique(day, return_inverse=True)
cluster_rows = np.bincount(inverse)
draws = np.random.default_rng(20260912).integers(0, cluster_rows.size, size=(args.bootstrap, cluster_rows.size))
report = {"validation": "fallback-only six-pair screen; not full ensemble; not a leaderboard forecast",
          "replication_gates": gates, "code_sha256": CODE, "rmse": {}, "comparisons": {}}
for name in NAMES:
    report["rmse"][name] = float(np.sqrt(np.mean((pooled[name].to_numpy()-y)**2)))
for candidate, reference in COMPARISONS.items():
    gain = (pooled[reference].to_numpy()-y)**2 - (pooled[candidate].to_numpy()-y)**2
    order = np.argsort(gain)[::-1]
    keep = np.ones(len(gain), bool)
    keep[order[:100]] = False
    fold_gains = {f: float(gain[folds == f].mean()) for f in sorted(set(folds))}
    jan = float(gain[(folds == "1_7") & (month == 1)].mean())
    jul = float(gain[(folds == "1_7") & (month == 7)].mean())
    boot = np.bincount(inverse, weights=gain)[draws].sum(axis=1)/cluster_rows[draws].sum(axis=1)
    checks = {"1_pooled_gain_positive": bool(gain.mean() > 0),
              "2_survives_top100_removal": bool(gain[keep].mean() > 0),
              "3_at_least_4_of_6_pairs": bool(sum(g > 0 for g in fold_gains.values()) >= 4),
              "4_january_and_july_gain_in_1_7": bool(jan > 0 and jul > 0)}
    airports = {a: float(gain[pooled.ADEP_mvt.to_numpy() == a].sum()/len(gain)) for a in sorted(pooled.ADEP_mvt.unique())}
    report["comparisons"][f"{candidate}_vs_{reference}"] = {
        "pooled_mse_gain": float(gain.mean()), "mse_gain_without_top100_rows": float(gain[keep].mean()),
        "positive_sse_mass": float(gain[gain > 0].sum()), "negative_sse_mass": float(gain[gain < 0].sum()),
        "airport_day_bootstrap_mse_gain_95pct": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        "pair_gains": fold_gains, "january_gain_1_7": jan, "july_gain_1_7": jul,
        "airport_contributions_to_pooled_gain": airports, "checks": checks, "advance": bool(all(checks.values()))}
report["seconds"] = time.time()-started
(output / "pooled.json").write_text(json.dumps(report, indent=2))
print("SUMMARY", json.dumps({"rmse": report["rmse"], **{k: {"gain": v["pooled_mse_gain"], "checks": v["checks"],
                                                           "advance": v["advance"]}
                                                       for k, v in report["comparisons"].items()}}, indent=2), flush=True)
