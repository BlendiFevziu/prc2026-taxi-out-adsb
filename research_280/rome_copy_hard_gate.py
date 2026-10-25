"""The surgical copy gate: confirmation run with a robustness rule that fits a 38-row change. Fixed 16.09.2026.

Where this stands. research_280/rome_copy_precision_study.py screened four precision targets, each threshold chosen
inside the ten training months of its pair. At the 0.99 target on the plain 35 fallback inputs, 38 rows were flagged
across six pairs, all at Rome, all of them true copies (precision 1.000), no row was made worse anywhere, pooled gain
+12,796 MSE, four pairs gained and two flagged nothing. It failed only "positive without the best 100 rows", which a
38-row change cannot pass: removing 100 rows removes every change and returns exactly 0.

So this run restates that one rule for surgical candidates and leaves the others untouched. The rule must still stop a
candidate whose gain rests on one or two lucky flights, so it removes a quarter of the candidate's OWN changed rows:

  1 pooled MSE gain > 0
  2 the gain survives removing its 10 most favourable FLAGGED rows (out of the roughly 38 it changes)
  3 no pair is negative, and at least 4 of 6 gain
  4 January and July both non-negative inside pair 1_7
  5 precision on held-out flagged rows >= 0.95, with the Wilson lower bound reported
  6 mean gain on ordinary rows (label <= 5,400 s) >= 0

Everything else is unchanged from the study: same tables, same classifier and settings, same label, same
threshold selection inside training months only, same v46 baseline. Only the "without" variant is run, because the
operator rate added nothing at this operating point (AUC 0.9517 against 0.9525).

Per-row output is written to the ignored cache so the build step can reuse it without refitting. Nothing is uploaded.

Run from the repository root:  python3 -u -B research_280/rome_copy_hard_gate.py
"""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, default=Path("."))
parser.add_argument("--jobs", type=int, default=4)
args = parser.parse_args()
project = args.project.resolve()
sys.path.insert(0, str(project))

import main_model as H  # torch before tree libraries on macOS
import features_stand_rate as MSQ
import lightgbm as lgb
import numpy as np
import pandas as pd

BASE = list(H.MERKMALE_RUECK)
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
TARGET = 0.99
TRIM_FLAGGED = 10
TABLES = project / ".cache/research_240/rescreen_production"
SAVED = project / ".cache/research_240/fallback_anchor_production"
CACHE = project / ".cache/research_280"
OUTPUT = project / "research_280/results/rome_copy_hard_gate.json"
ORDINARY = 5400.0
V48, SHARE_2026 = 266.4508, 5290 / 344841
index = {tuple(sorted(json.loads(s.read_text())["source_months"])): s.with_suffix(".parquet")
         for s in TABLES.glob("rescreen_*.json")}


def classifier():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                              min_child_samples=20, verbose=-1, n_jobs=args.jobs, random_state=0)


def pick_threshold(probability, label, target):
    order = np.argsort(probability)[::-1]
    hits = np.cumsum(label[order])
    precision = hits / np.arange(1, len(order) + 1)
    ok = np.flatnonzero((precision >= target) & (hits >= 10))
    return 1.01 if len(ok) == 0 else float(probability[order][ok[-1]])


def wilson(successes, total, z=1.96):
    if total == 0:
        return float("nan")
    p = successes / total
    denominator = 1 + z ** 2 / total
    centre = p + z ** 2 / (2 * total)
    margin = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))
    return float((centre - margin) / denominator)


parts = []
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    learn = [m for m in range(1, 13) if m not in pair]
    tr = pd.read_parquet(index[tuple(learn)]).reset_index(drop=True)
    va = pd.read_parquet(index[tuple(sorted(pair))]).reset_index(drop=True)
    for frame in (tr, va):
        frame["z"] = (frame.y > H.SCHWELLE_EXTREM).astype(int)
        frame["zk"] = (np.abs(frame.y.to_numpy(float) - 60 * frame.versp_min.to_numpy(float)) <= 60).astype(int)
    tr, va = MSQ.ausser_faltung(tr), MSQ.wende_an(MSQ.passe_an(tr), va)
    label = tr.zk.to_numpy()
    inner = np.full(len(tr), np.nan)
    for month in sorted(tr.monat.unique()):
        hold = tr.monat.eq(month).to_numpy()
        inner[hold] = classifier().fit(tr.loc[~hold, BASE], label[~hold]).predict_proba(tr.loc[hold, BASE])[:, 1]
    threshold = pick_threshold(inner, label, TARGET)
    probability = classifier().fit(tr[BASE], label).predict_proba(va[BASE])[:, 1]
    base = pd.read_parquet(SAVED / f"predictions_{key}.parquet").set_index("MVT_ID_mvt").loc[va.MVT_ID_mvt, "prod_anchor"].to_numpy(float)
    delay = 60 * va.versp_min.to_numpy(float)
    part = pd.DataFrame({"fold": key, "MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(),
                         "airport": va.ADEP_mvt.astype(str).to_numpy(), "MVT_TIME_UTC_mvt": va.MVT_TIME_UTC_mvt.to_numpy(),
                         "y": va.y.to_numpy(float), "delay": delay, "prod_anchor": base,
                         "copy": va.zk.to_numpy(bool), "p": probability, "threshold": threshold})
    part["flagged"] = part.p.to_numpy() >= threshold
    part["candidate"] = np.where(part.flagged, np.maximum(np.nan_to_num(delay), 0), base)
    part["gain"] = (base - part.y.to_numpy()) ** 2 - (part.candidate.to_numpy() - part.y.to_numpy()) ** 2
    print(key, "threshold", round(threshold, 5), "flagged", int(part.flagged.sum()),
          "correct", int((part.flagged & part["copy"]).sum()), "gain", round(float(part.gain.mean())), flush=True)
    parts.append(part)

pooled = pd.concat(parts, ignore_index=True)
CACHE.mkdir(parents=True, exist_ok=True)
pooled.to_parquet(CACHE / "rome_copy_hard_gate_rows.parquet", index=False)
gain = pooled.gain.to_numpy()
flagged = pooled.flagged.to_numpy()
y = pooled.y.to_numpy()
stamps = pd.to_datetime(pooled.MVT_TIME_UTC_mvt, utc=True)
folds = pooled.fold.to_numpy()
month = np.array([m if str(m) in f.split("_") else m % 12 + 1 for m, f in zip(stamps.dt.month.to_numpy(), folds)])
flag_gain = np.sort(gain[flagged])[::-1]
trimmed_total = flag_gain[TRIM_FLAGGED:].sum()
pair_gains = {f: float(gain[folds == f].mean()) for f in sorted(set(folds))}
correct = int((flagged & pooled["copy"].to_numpy()).sum())
report = {
    "rows": int(len(pooled)), "flagged": int(flagged.sum()), "flagged_correct": correct,
    "flagged_by_airport": pooled.loc[flagged].airport.value_counts().to_dict(),
    "precision": float(correct / max(flagged.sum(), 1)),
    "precision_wilson_lower_95": wilson(correct, int(flagged.sum())),
    "recall_of_all_copies": float(correct / int(pooled["copy"].sum())),
    "pooled_mse_gain": float(gain.mean()),
    "pooled_mse_gain_without_best_10_flagged": float(trimmed_total / len(gain)),
    "flagged_row_gains_sorted": [float(v) for v in flag_gain],
    "pair_gains": pair_gains, "pairs_positive": int(sum(g > 0 for g in pair_gains.values())),
    "pairs_negative": int(sum(g < 0 for g in pair_gains.values())),
    "january_gain_1_7": float(gain[(folds == "1_7") & (month == 1)].mean()),
    "july_gain_1_7": float(gain[(folds == "1_7") & (month == 7)].mean()),
    "ordinary_rows_mse_gain": float(gain[y <= ORDINARY].mean()),
    "worst_single_row_gain": float(gain.min()),
    "thresholds_by_pair": {f: float(pooled.loc[pooled.fold == f, "threshold"].iloc[0]) for f in sorted(set(folds))},
    "mechanical_leaderboard_rmse_if_full_transfer_not_forecast": float(np.sqrt(max(V48 ** 2 - SHARE_2026 * float(gain.mean()), 0))),
}
report["checks"] = {
    "1_pooled_gain_positive": report["pooled_mse_gain"] > 0,
    "2_survives_removing_best_10_flagged": report["pooled_mse_gain_without_best_10_flagged"] > 0,
    "3_no_pair_negative_and_four_positive": report["pairs_negative"] == 0 and report["pairs_positive"] >= 4,
    "4_january_and_july_not_negative": report["january_gain_1_7"] >= 0 and report["july_gain_1_7"] >= 0,
    "5_precision_at_least_0_95": report["precision"] >= 0.95,
    "6_ordinary_rows_not_worse": report["ordinary_rows_mse_gain"] >= 0,
}
report["advance"] = bool(all(report["checks"].values()))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
print("\nflagged", report["flagged"], "correct", report["flagged_correct"],
      "precision", round(report["precision"], 4), "Wilson lower", round(report["precision_wilson_lower_95"], 4))
print("pooled gain", round(report["pooled_mse_gain"]),
      "| without its best 10 flagged rows", round(report["pooled_mse_gain_without_best_10_flagged"]))
print("pairs", {k: round(v) for k, v in pair_gains.items()})
print("January", round(report["january_gain_1_7"]), "July", round(report["july_gain_1_7"]),
      "| ordinary rows", round(report["ordinary_rows_mse_gain"]), "| worst row", round(report["worst_single_row_gain"]))
print("flagged by airport", report["flagged_by_airport"])
print("checks", json.dumps(report["checks"]), "ADVANCE", report["advance"])
print("mechanical implied RMSE if it transferred in full:", round(report["mechanical_leaderboard_rmse_if_full_transfer_not_forecast"], 4))
