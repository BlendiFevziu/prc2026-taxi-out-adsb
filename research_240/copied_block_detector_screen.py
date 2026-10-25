"""Pre-registered stage 1: do arrival-side copy signals improve detection of copied block times?

Pre-specified 13.09.2026, before any fit. No leaderboard feedback and no 2026 label are used.

Label (production definition, main lane, hat_aobt = 1): copied = |y - 60 * versp_min| <= 10 s,
which is BLOCK_TIME within 10 s of SCHED_TIME.
Held-out evaluations, both on tables of the v45 release cache, built independently per source-month set:
  A_1_7  learn months 2-6 and 8-12, test January and July (the release's outer split)
  B_3_9  learn months 2, 4-6, 8 and 10-12, test March and September (the release's inner split)
Baseline: the release detector (LightGBM binary, 700 rounds, learning rate 0.05, 127 leaves,
min_child_samples 40, random_state 0, n_jobs 8) on the 91 v45 main features.
Replication gate (split A): the baseline fit must reproduce the saved release detector
(candidate/checkpoints/reporting_classifier.pkl) on the January/July main rows within 1e-6.
Candidate: the same detector with research_240/features_arrival_copy.MERKMALE added, each table's
features built from the arrivals of its own source months.

Metrics per split: ROC AUC, average precision, recall at precision >= 0.95 and >= 0.99, rows and
precision at probability >= 0.9, AUC inside and outside LIRF. Coverage check: ac_inbound_copied
present on at least 80 % of main rows in both test sets and in the 2026 ranking main rows, with
2026 within 5 points of the pooled 2025 test coverage.

Decision rule: stage 2 (RMSE with the release ensemble) is considered only if the AUC gain is at
least 0.005 in both splits and the coverage check passes; then a within-airport permutation of the
four features (seeds 101/102, split A) must bring the AUC gain below 0.005. Otherwise stop, record
the result and build nothing.
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
parser.add_argument("--release", default=Path(".cache/clock_queue_release_20260912"), type=Path)
parser.add_argument("--jobs", type=int, default=8)
args = parser.parse_args()
project, output = args.project.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project))
os.chdir(project)
release = args.release if args.release.is_absolute() else project / args.release

import main_model as H  # torch before tree libraries on macOS
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fid_gap_cv import month_of_file
from research_250 import candidate_features as B
from research_240 import features_arrival_copy as AC

FEATURES = B.main_features(H.MERKMALE_HAUPT)
assert len(FEATURES) == 91, len(FEATURES)
SPLITS = {"A_1_7": ([2, 3, 4, 5, 6, 8, 9, 10, 11, 12], [1, 7]), "B_3_9": ([2, 4, 5, 6, 8, 10, 11, 12], [3, 9])}
DEPARTURE_COLUMNS = ["ADEP_mvt", "STAND_mvt", "AIRCRAFT_OPERATOR_flt", "MVT_TIME_UTC_mvt", "ersatz_s"]
MIN_AUC_GAIN = 0.005
paths = sorted((project / "data").glob("training_2025-*.parquet"))
assert len(paths) == 12 and {month_of_file(p) for p in paths} == set(range(1, 13))
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), Path(AC.__file__).resolve(), project / "main_model.py"]}


def with_arrival_copy(frame, raw_files):
    main = frame.loc[frame.hat_aobt.eq(1)].copy()
    raw = pd.concat([pd.read_parquet(p, columns=AC.ARRIVAL_COLUMNS) for p in raw_files], ignore_index=True)
    extra = AC.features_for(main[DEPARTURE_COLUMNS], AC.arrivals_table(raw))
    for col in AC.MERKMALE:
        main[col] = extra[col].to_numpy()
    return main


def load(months):
    frame = pd.read_parquet(release / "data" / ("months_" + "_".join(map(str, sorted(months))) + ".parquet"))
    main = with_arrival_copy(frame, [p for p in paths if month_of_file(p) in months])
    main["copied"] = ((main.y - main.versp_min * 60).abs() <= 10).astype(int)
    return main


def detector():
    return lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=.05, num_leaves=127,
                              min_child_samples=40, verbose=-1, n_jobs=args.jobs, random_state=0)


def auc(y, score):
    y = np.asarray(y, int)
    ranks = pd.Series(np.asarray(score, float)).rank(method="average").to_numpy()
    positive = y == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def detection(y, score, airports):
    y, score = np.asarray(y, int), np.asarray(score, float)
    order = np.argsort(-score, kind="mergesort")
    ordered = y[order]
    tp, fp = np.cumsum(ordered), np.cumsum(1 - ordered)
    precision, recall = tp / (tp + fp), tp / ordered.sum()
    out = {"auc": auc(y, score), "average_precision": float((precision * ordered).sum() / ordered.sum()),
           "positives": int(y.sum()), "rows": int(len(y))}
    for level in (0.95, 0.99):
        ok = precision >= level
        out[f"recall_at_precision_{level}"] = float(recall[ok].max()) if ok.any() else 0.0
    high = score >= 0.9
    out["rows_p_ge_0.9"] = int(high.sum())
    out["precision_p_ge_0.9"] = float(y[high].mean()) if high.any() else None
    out["recall_p_ge_0.9"] = float(y[high].sum() / y.sum())
    rome = airports == "LIRF"
    out["auc_lirf"], out["auc_outside_lirf"] = auc(y[rome], score[rome]), auc(y[~rome], score[~rome])
    return out


def permute(frame, columns, seed):
    rng = np.random.default_rng(seed)
    out = frame.copy()
    values = out[columns].to_numpy(copy=True)
    airports = out.ADEP_mvt.astype(str).to_numpy()
    for airport in np.unique(airports):
        idx = np.flatnonzero(airports == airport)
        values[idx] = values[rng.permutation(idx)]
    out[columns] = values
    return out


started = time.time()
report = {"validation": "stage 1 detector screen on release-cache tables; main lane only; not an RMSE result",
          "features_added": AC.MERKMALE, "code_sha256": CODE, "splits": {}}
kept = {}
for name, (learn, test) in SPLITS.items():
    train, valid = load(learn), load(test)
    assert set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))
    y = valid.copied.to_numpy()
    airports = valid.ADEP_mvt.astype(str).to_numpy()
    before = time.time()
    p_base = detector().fit(train[FEATURES], train.copied).predict_proba(valid[FEATURES])[:, 1]
    entry = {"learn_months": learn, "test_months": test, "train_rows": int(len(train)), "test_rows": int(len(valid)),
             "copied_share_train": float(train.copied.mean()), "copied_share_test": float(y.mean()),
             "baseline_fit_seconds": time.time() - before}
    if name == "A_1_7":
        saved = joblib.load(release / "candidate/checkpoints/reporting_classifier.pkl")
        saved = saved.get("model", saved.get("value", saved)) if isinstance(saved, dict) else saved
        gap = float(np.abs(p_base - saved.predict_proba(valid[FEATURES])[:, 1]).max())
        entry["replication_max_abs_difference"] = gap
        print("REPLICATION", name, gap, flush=True)
        if gap > 1e-6:
            report["splits"][name] = entry
            (output / "replication_failed.json").write_text(json.dumps(report, indent=2))
            sys.exit("Baseline detector does not reproduce the saved release detector")
    before = time.time()
    p_cand = detector().fit(train[FEATURES + AC.MERKMALE], train.copied).predict_proba(valid[FEATURES + AC.MERKMALE])[:, 1]
    entry["candidate_fit_seconds"] = time.time() - before
    entry["baseline"], entry["candidate"] = detection(y, p_base, airports), detection(y, p_cand, airports)
    entry["auc_gain"] = entry["candidate"]["auc"] - entry["baseline"]["auc"]
    inbound = train.ac_inbound_copied.to_numpy(float)
    rome_train = train.ADEP_mvt.astype(str).to_numpy() == "LIRF"
    entry["train_copied_share_by_inbound_flag"] = {
        region: {str(flag): float(train.copied.to_numpy()[mask & (inbound == flag)].mean()) if (mask & (inbound == flag)).any() else None
                 for flag in (0.0, 1.0)}
        for region, mask in (("LIRF", rome_train), ("outside_LIRF", ~rome_train))}
    entry["coverage_test"] = {c: float(np.isfinite(valid[c].to_numpy(float)).mean()) for c in AC.MERKMALE}
    report["splits"][name] = entry
    pd.DataFrame({"MVT_ID_mvt": valid.MVT_ID_mvt.to_numpy(), "ADEP_mvt": airports, "copied": y,
                  "p_baseline": p_base, "p_candidate": p_cand}).to_parquet(output / f"detector_{name}.parquet", index=False)
    print("SPLIT", name, json.dumps({"auc_base": entry["baseline"]["auc"], "auc_cand": entry["candidate"]["auc"],
                                     "gain": entry["auc_gain"], "coverage": entry["coverage_test"]}), flush=True)
    kept[name] = (learn, test)
    del train, valid
    gc.collect()

ranking = with_arrival_copy(pd.read_parquet(release / "data/ranking.parquet"), [project / "data/ranking.parquet"])
coverage_2026 = float(np.isfinite(ranking.ac_inbound_copied.to_numpy(float)).mean())
rows = [report["splits"][n]["test_rows"] for n in SPLITS]
pooled_2025 = float(np.average([report["splits"][n]["coverage_test"]["ac_inbound_copied"] for n in SPLITS], weights=rows))
report["coverage"] = {"inbound_2025_pooled_test": pooled_2025, "inbound_2026_ranking_main": coverage_2026,
                      "passed": bool(min(report["splits"][n]["coverage_test"]["ac_inbound_copied"] for n in SPLITS) >= .8
                                     and coverage_2026 >= .8 and abs(coverage_2026 - pooled_2025) <= .05)}
gains_ok = all(report["splits"][n]["auc_gain"] >= MIN_AUC_GAIN for n in SPLITS)
decision = {"auc_gain_at_least_0.005_in_both_splits": bool(gains_ok), "coverage_passed": report["coverage"]["passed"]}
if gains_ok and report["coverage"]["passed"]:
    learn, test = SPLITS["A_1_7"]
    train, valid = load(learn), load(test)
    train_p, valid_p = permute(train, AC.MERKMALE, 101), permute(valid, AC.MERKMALE, 102)
    p_perm = detector().fit(train_p[FEATURES + AC.MERKMALE], train_p.copied).predict_proba(valid_p[FEATURES + AC.MERKMALE])[:, 1]
    permuted_gain = auc(valid.copied, p_perm) - report["splits"]["A_1_7"]["baseline"]["auc"]
    decision["permuted_auc_gain_split_A"] = float(permuted_gain)
    decision["permutation_removes_gain"] = bool(permuted_gain < MIN_AUC_GAIN)
decision["proceed_to_stage_2"] = bool(gains_ok and report["coverage"]["passed"] and decision.get("permutation_removes_gain", False))
report["decision"] = decision
report["seconds"] = time.time() - started
(output / "result.json").write_text(json.dumps(report, indent=2))
print("DECISION", json.dumps(decision), flush=True)
