"""Pre-registered January/July diagnostic: a second soft gate for block times copied from EOBT_1.

Pre-specified 13.09.2026, before any fit, after anchor_copy_structure.py found on the v45 candidate-stage holdout
6,058 main rows (1.79 %) whose block time is within 10 s of EOBT_1, LOBT or IOBT while that anchor is more than 60 s
from AOBT_3 and the block time is not the schedule (the existing gate's case): 2.35 % of main squared error, ensemble
bias +74.5 s, oracle bound 1,105 MSE; EOBT_1 is involved in 5,195 of them. Run outside the repository, then moved
into research_240/. No leaderboard feedback.

Label (main rows): eobt_copy = |TAXITIME - (MVT_TIME - EOBT_1)| <= 10 s and |EOBT_1 - AOBT_3| > 60 s and not a
schedule copy (|TAXITIME - delay| <= 10 s).
Classifier: LightGBM binary with the reporting-classifier settings of research_250/release_common.py (700 rounds,
learning rate 0.05, 127 leaves, min_child_samples 40, seed 0) on the 91 v45 main features, research encoding, fitted on
the learning months 2-6 and 8-12 of the v45 release cache.
Gate on the complete holdout ensemble prediction v (schedule gate and floor already applied), January/July main rows:
  v2 = max((1 - p**h) * v + p**h * (MVT_TIME - EOBT_1), 300), unchanged where EOBT_1 is missing; h = 2.5 (ERF_HOCH).
Decision to build a candidate: (1) main RMSE gain >= 0.15 s at h = 2.5, (2) main MSE gain positive without its 100 most
favourable rows, (3) January and July both gain, (4) h chosen on one month from 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 and applied
to the other gains in both months, (5) the real MSE gain exceeds every gain with the training labels permuted within
airport, seeds 101, 202, 303, 404, 505 (run only if 1-4 pass).
Single split on a full-ensemble base, the precedent of ERF_HOCH; not six-pair evidence; not a leaderboard forecast.

Run:
  python3 -B research_240/eobt_copy_gate_diagnostic.py --project <repository> --output <folder>
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
import pyarrow.parquet as pq
from research_250 import candidate_features as B

RELEASE = project / ".cache/clock_queue_release_20260912"
FEATURES = B.main_features(H.MERKMALE_HAUPT)
EXTRA = ["y", "ersatz_s", "ersatz_eobt_s", "versp_min", "hat_aobt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "MVT_ID_mvt"]
H_GRID, H_PRE = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0], 2.5
SEEDS = [101, 202, 303, 404, 505]
HOLDOUT_MAIN_RMSE = 216.7734634758806
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(), project / "main_model.py", Path(B.__file__).resolve()]}
started = time.time()


def load(months):
    path = RELEASE / "data" / ("months_" + "_".join(map(str, months)) + ".parquet")
    wanted = list(dict.fromkeys(FEATURES + EXTRA))
    missing = [c for c in wanted if c not in set(pq.read_schema(path).names)]
    assert not missing, missing
    frame = pd.read_parquet(path, columns=wanted)
    return frame.loc[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)


def label(frame):
    y, implied = frame.y.to_numpy(float), frame.ersatz_eobt_s.to_numpy(float)
    return ((np.abs(y - implied) <= 10) & (np.abs(implied - frame.ersatz_s.to_numpy(float)) > 60)
            & ~(np.abs(y - frame.versp_min.to_numpy(float) * 60) <= 10)).astype(int)


def auc(labels, scores):
    ranks = pd.Series(scores).rank().to_numpy()
    pos = labels.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


train, valid = load([2, 3, 4, 5, 6, 8, 9, 10, 11, 12]), load([1, 7])
assert set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))
for col in H.KAT:
    if col in train.columns:
        cats = pd.unique(train[col].astype(str))
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)
holdout = pd.read_parquet(RELEASE / "candidate/holdout_predictions.parquet", columns=["MVT_ID_mvt", "y", "prediction"]).set_index("MVT_ID_mvt")
holdout = holdout.loc[valid.MVT_ID_mvt]
y = valid.y.to_numpy(float)
assert np.array_equal(holdout.y.to_numpy(float), y)
v = holdout.prediction.to_numpy(float)
assert abs(np.sqrt(np.mean((v - y)**2)) - HOLDOUT_MAIN_RMSE) < 1e-6
target_train, target_valid = label(train), label(valid)
implied = valid.ersatz_eobt_s.to_numpy(float)
usable = np.isfinite(implied)
raw_month = pd.to_datetime(valid.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
month = np.where(np.isin(raw_month, [1, 7]), raw_month, raw_month % 12 + 1)
airports_train = train.ADEP_mvt.astype(str).to_numpy()
print("LOADED", len(train), len(valid), "label rate train", round(float(target_train.mean()), 5), "valid", round(float(target_valid.mean()), 5),
      f"{time.time()-started:.0f}s", flush=True)


def classifier(labels):
    model = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=.05, num_leaves=127, min_child_samples=40,
                               verbose=-1, n_jobs=args.jobs, random_state=0)
    return model.fit(train[FEATURES], labels)


def gated(p, h):
    q = np.where(usable, p ** h, 0.0)
    out = v.copy()
    out[usable] = (1 - q[usable]) * v[usable] + q[usable] * implied[usable]
    return np.maximum(out, 300)


def evaluate(prediction):
    gain = (v - y)**2 - (prediction - y)**2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[::-1][:100]] = False
    return {"rmse": float(np.sqrt(np.mean((prediction - y)**2))), "rmse_gain": float(np.sqrt(np.mean((v - y)**2)) - np.sqrt(np.mean((prediction - y)**2))),
            "mse_gain": float(gain.mean()), "mse_gain_without_top100_rows": float(gain[keep].mean()),
            "january_mse_gain": float(gain[month == 1].mean()), "july_mse_gain": float(gain[month == 7].mean()),
            "rows_changed_over_60s": int(np.sum(np.abs(prediction - v) > 60))}


before = time.time()
p = classifier(target_train).predict_proba(valid[FEATURES])[:, 1]
print("FIT classifier", f"{time.time()-before:.0f}s", flush=True)
by_h = {str(h): evaluate(gated(p, h)) for h in H_GRID}
cross = {}
for mo, other in ((1, 7), (7, 1)):
    key = "january_mse_gain" if other == 1 else "july_mse_gain"
    chosen = max(H_GRID, key=lambda h: by_h[str(h)][key])
    cross[str(mo)] = {"h_chosen_on_other_month": chosen, "mse_gain": by_h[str(chosen)]["january_mse_gain" if mo == 1 else "july_mse_gain"]}
pre = by_h[str(H_PRE)]
report = {"validation": "January/July single split on the complete v45 candidate-stage holdout ensemble; not six-pair; not a leaderboard forecast",
          "rows_train": int(len(train)), "rows_valid": int(len(valid)), "label_rate_train": float(target_train.mean()),
          "label_rate_valid": float(target_valid.mean()), "auc_valid": auc(target_valid, p),
          "precision_recall_at_p_0_5": {"precision": float(target_valid[p > .5].mean()) if (p > .5).any() else None,
                                        "recall": float((p[target_valid == 1] > .5).mean()), "rows_flagged": int((p > .5).sum())},
          "holdout_main_rmse": HOLDOUT_MAIN_RMSE, "by_h": by_h, "cross_fitted_h": cross, "code_sha256": CODE}
checks = {"1_rmse_gain_at_least_0_15_s": pre["rmse_gain"] >= .15, "2_survives_top100_removal": pre["mse_gain_without_top100_rows"] > 0,
          "3_january_and_july_gain": pre["january_mse_gain"] > 0 and pre["july_mse_gain"] > 0,
          "4_cross_fitted_h_gains_both_months": all(c["mse_gain"] > 0 for c in cross.values())}
print("SCREEN", json.dumps({"auc": report["auc_valid"], "pre": pre, "cross": cross, "checks": checks}), flush=True)
if all(checks.values()):
    permuted = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        shuffled = target_train.copy()
        for airport in np.unique(airports_train):
            idx = np.flatnonzero(airports_train == airport)
            shuffled[idx] = shuffled[rng.permutation(idx)]
        permuted[str(seed)] = evaluate(gated(classifier(shuffled).predict_proba(valid[FEATURES])[:, 1], H_PRE))["mse_gain"]
        print("PERMUTATION", seed, permuted[str(seed)], flush=True)
    report["permuted_mse_gains"] = permuted
    checks["5_beats_every_label_permutation"] = all(pre["mse_gain"] > g for g in permuted.values())
report["decision"] = {"checks": checks, "build_candidate": bool(all(checks.values()) and "5_beats_every_label_permutation" in checks)}
report["seconds"] = time.time() - started
(output / "eobt_copy_gate_diagnostic.json").write_text(json.dumps(report, indent=2))
pd.DataFrame({"MVT_ID_mvt": valid.MVT_ID_mvt.to_numpy(), "y": y, "holdout": v, "p_eobt_copy": p, "label": target_valid}).to_parquet(
    output / "eobt_copy_gate_predictions_1_7.parquet", index=False)
print("SUMMARY", json.dumps({k: report[k] for k in ("label_rate_train", "label_rate_valid", "auc_valid", "precision_recall_at_p_0_5", "by_h",
                                                    "cross_fitted_h", "decision")}, indent=2), flush=True)
