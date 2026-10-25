"""Precision study for the Rome copy mechanism. Rules fixed 16.09.2026 before any fit.

The question, and why it is the only one left. Six wirings of the copy mechanism have now gained in expectation and
failed the same rule: the gain lives in a few dozen flights and is paid for elsewhere. A wiring can only survive that
rule if it changes rows it is nearly certain about. So the decisive quantity is not AUC but precision: is there a
probability threshold at which almost every flagged row really is a copy, and does enough traffic sit above it to
matter?

Design.
  Rows: the six-pair fallback tables, 22,470 held-out rows in total, pooled across pairs.
  Label: copy, |y - schedule delay| <= 60 s.
  Baseline: the scored v46 fallback predictions saved in .cache/research_240/fallback_anchor_production, used as
    given here (the two screens of today already verified that the recipe reproduces them within 1.5e-11 s).
  Classifier: LightGBM binary with the extreme classifier's settings (600 rounds, learning rate 0.05, 31 leaves,
    minimum 20 rows per leaf, seed 0), fitted on the ten training months of each pair.
  Two variants, which also serve as the ablation for the new column:
    WITH    the 35 fallback inputs plus the out-of-fold operator copy rate (features_operator_copy.py)
    WITHOUT the 35 fallback inputs alone
  Candidate action: hard substitution. A flagged row is predicted as its schedule delay, every other row keeps the
    v46 prediction. Hard rather than blended, because the point is to act only where the evidence is decisive.

Threshold selection, and this is what keeps the study honest. Precision targets are fixed here in advance at 0.90,
0.95, 0.97 and 0.99. For each pair and target, the threshold is chosen INSIDE the ten training months, from
month-held-out predictions, as the lowest probability whose training precision reaches the target. It is then applied
unchanged to the held-out pair. No threshold is ever read off the held-out rows.

Reported per target: rows flagged, achieved precision and recall on held-out rows, pooled MSE gain, the gain without
its 100 most favourable rows, pairs positive, January and July inside pair 1_7, the worst single-row loss, and the
same restricted to LIRF. A target passes only with all of: positive pooled gain, positive without the best 100 rows,
at least 4 of 6 pairs, both months positive in pair 1_7, and no worsening on ordinary rows (label <= 5,400 s).

If no target passes, the mechanism is closed for this line of attack and the finding is that a per-row probability
sharp enough to act on does not exist in these inputs. Nothing is built or uploaded here.

Run from the repository root:  python3 -u -B research_280/rome_copy_precision_study.py
"""
import argparse
import glob
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
import features_operator_copy as OPC
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = list(H.MERKMALE_RUECK)
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
TARGETS = [0.90, 0.95, 0.97, 0.99]
TABLES = project / ".cache/research_240/rescreen_production"
SAVED = project / ".cache/research_240/fallback_anchor_production"
OUTPUT = project / "research_280/results/rome_copy_precision_study.json"
ORDINARY = 5400.0
index = {tuple(sorted(json.loads(s.read_text())["source_months"])): s.with_suffix(".parquet")
         for s in TABLES.glob("rescreen_*.json")}


def flight_numbers():
    frames = [pq.ParquetFile(p).read(columns=["MVT_ID_mvt", "FLIGHT_mvt"], use_threads=False).to_pandas()
              for p in sorted(glob.glob(str(project / "data/training_2025-*.parquet")))]
    return pd.concat(frames, ignore_index=True).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").FLIGHT_mvt


def classifier():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                              min_child_samples=20, verbose=-1, n_jobs=args.jobs, random_state=0)


def pick_threshold(probability, label, target):
    """Lowest probability whose precision reaches the target, computed on training rows only."""
    order = np.argsort(probability)[::-1]
    hits = np.cumsum(label[order])
    precision = hits / np.arange(1, len(order) + 1)
    ok = np.flatnonzero((precision >= target) & (hits >= 10))
    if len(ok) == 0:
        return 1.01
    return float(probability[order][ok[-1]])


numbers = flight_numbers()
parts = []
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    learn = [m for m in range(1, 13) if m not in pair]
    tr = pd.read_parquet(index[tuple(learn)]).reset_index(drop=True)
    va = pd.read_parquet(index[tuple(sorted(pair))]).reset_index(drop=True)
    for frame in (tr, va):
        frame["z"] = (frame.y > H.SCHWELLE_EXTREM).astype(int)
        frame["zk"] = (np.abs(frame.y.to_numpy(float) - 60 * frame.versp_min.to_numpy(float)) <= 60).astype(int)
        frame["op_praefix"] = OPC.praefix(frame.MVT_ID_mvt.map(numbers)).to_numpy()
    tr, va = MSQ.ausser_faltung(tr), MSQ.wende_an(MSQ.passe_an(tr), va)
    rate = OPC.passe_an(tr)
    tr_rate, va_rate = OPC.ausser_faltung(tr), OPC.wende_an(rate, va)
    saved = pd.read_parquet(SAVED / f"predictions_{key}.parquet").set_index("MVT_ID_mvt")
    base = saved.loc[va.MVT_ID_mvt, "prod_anchor"].to_numpy(float)
    part = pd.DataFrame({"fold": key, "airport": va.ADEP_mvt.astype(str).to_numpy(),
                         "MVT_TIME_UTC_mvt": va.MVT_TIME_UTC_mvt.to_numpy(), "y": va.y.to_numpy(float),
                         "delay": 60 * va.versp_min.to_numpy(float), "prod_anchor": base,
                         "copy": va.zk.to_numpy(bool)})
    for variant, train_frame, valid_frame in (("with", tr_rate, va_rate), ("without", tr, va)):
        columns = BASE + (OPC.MERKMALE if variant == "with" else [])
        label = train_frame.zk.to_numpy()
        inner = np.full(len(train_frame), np.nan)
        for month in sorted(train_frame.monat.unique()):
            hold = train_frame.monat.eq(month).to_numpy()
            model = classifier().fit(train_frame.loc[~hold, columns], label[~hold])
            inner[hold] = model.predict_proba(train_frame.loc[hold, columns])[:, 1]
        assert np.isfinite(inner).all()
        outer = classifier().fit(train_frame[columns], label).predict_proba(valid_frame[columns])[:, 1]
        part["p_" + variant] = outer
        for target in TARGETS:
            part[f"thr_{variant}_{target}"] = pick_threshold(inner, label, target)
        print(key, variant, "inner base rate", round(float(label.mean()), 4),
              "thresholds", {t: round(part[f'thr_{variant}_{t}'][0], 4) for t in TARGETS}, flush=True)
    parts.append(part)

pooled = pd.concat(parts, ignore_index=True)
y, delay, base = pooled.y.to_numpy(), pooled.delay.to_numpy(), pooled.prod_anchor.to_numpy()
folds, copies = pooled.fold.to_numpy(), pooled["copy"].to_numpy()
stamps = pd.to_datetime(pooled.MVT_TIME_UTC_mvt, utc=True)
month = np.array([m if str(m) in f.split("_") else m % 12 + 1 for m, f in zip(stamps.dt.month.to_numpy(), folds)])
rome = pooled.airport.to_numpy() == "LIRF"
ordinary = y <= ORDINARY
report = {"rows": int(len(pooled)), "copies": int(copies.sum()), "copy_rate": float(copies.mean()),
          "rome_rows": int(rome.sum()), "rome_copies": int(copies[rome].sum()), "targets": {}}


def auc(label, score):
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    positives, negatives = label.sum(), (~label).sum()
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


for variant in ("with", "without"):
    p = pooled["p_" + variant].to_numpy()
    report[f"auc_{variant}"] = {"all": auc(copies, p), "rome": auc(copies[rome], p[rome])}
print("\nAUC", json.dumps({k: {a: round(v, 4) for a, v in report[f'auc_{k}'].items()} for k in ("with", "without")}))

for variant in ("with", "without"):
    for target in TARGETS:
        threshold = pooled[f"thr_{variant}_{target}"].to_numpy()
        flag = pooled["p_" + variant].to_numpy() >= threshold
        candidate = np.where(flag, np.maximum(np.nan_to_num(delay), 0), base)
        gain = (base - y) ** 2 - (candidate - y) ** 2
        keep = np.ones(len(gain), bool)
        keep[np.argsort(gain)[::-1][:100]] = False
        pair_gains = {f: float(gain[folds == f].mean()) for f in sorted(set(folds))}
        block = {"flagged": int(flag.sum()), "flagged_rome": int((flag & rome).sum()),
                 "precision": float(copies[flag].mean()) if flag.any() else float("nan"),
                 "recall": float(copies[flag].sum() / copies.sum()),
                 "precision_rome": float(copies[flag & rome].mean()) if (flag & rome).any() else float("nan"),
                 "pooled_mse_gain": float(gain.mean()),
                 "mse_gain_without_top100_rows": float(gain[keep].mean()),
                 "pairs_positive": int(sum(g > 0 for g in pair_gains.values())), "pair_gains": pair_gains,
                 "january_gain_1_7": float(gain[(folds == "1_7") & (month == 1)].mean()),
                 "july_gain_1_7": float(gain[(folds == "1_7") & (month == 7)].mean()),
                 "ordinary_rows_mse_gain": float(gain[ordinary].mean()),
                 "worst_single_row_loss": float(gain.min()),
                 "rome_mse_gain_share": float(gain[rome].sum() / gain.sum()) if gain.sum() else float("nan")}
        block["checks"] = {"1_pooled_gain_positive": block["pooled_mse_gain"] > 0,
                           "2_survives_top100_removal": block["mse_gain_without_top100_rows"] > 0,
                           "3_at_least_4_of_6_pairs": block["pairs_positive"] >= 4,
                           "4_january_and_july_gain_in_1_7": block["january_gain_1_7"] > 0 and block["july_gain_1_7"] > 0,
                           "5_ordinary_rows_not_worse": block["ordinary_rows_mse_gain"] >= 0}
        block["passes"] = bool(all(block["checks"].values()))
        report["targets"][f"{variant}_{target}"] = block
        print(f"\n{variant:7s} target {target:.2f}: flagged {block['flagged']:4d} "
              f"(Rome {block['flagged_rome']:4d}), precision {block['precision']:.3f}, recall {block['recall']:.3f}")
        print(f"   gain {block['pooled_mse_gain']:>10,.0f} | without best 100 {block['mse_gain_without_top100_rows']:>10,.0f} "
              f"| pairs {block['pairs_positive']}/6 | Jan {block['january_gain_1_7']:>9,.0f} Jul {block['july_gain_1_7']:>9,.0f}")
        print(f"   ordinary rows {block['ordinary_rows_mse_gain']:>10,.0f} | worst row {block['worst_single_row_loss']:>12,.0f} "
              f"| passes {block['passes']}")

passing = [k for k, v in report["targets"].items() if v["passes"]]
report["passing"] = passing
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
print("\nPASSING", passing or "none")
