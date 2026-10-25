"""Diagnostic (16.09.2026): is Rome's copy risk a property of the flight identity, and does it transfer in time?

Where this comes from. research_280/rome_record_shape.py found one separating quantity among Rome's fallback rows:
the flight identifier of a copied row occurs a median 44 times in 2025 against 188 for non-copies. Nothing about the
record's shape separates them. That suggests the risk belongs to the flight, not to the moment, which is how the
repository already treats stands (features_stand_rate.py learns an out-of-fold extreme rate per stand instead of
using STAND_mvt directly).

The question this answers is the one that decides whether such a feature is worth building: does a copy rate learned
on early months predict copies in later months, on rows the rate never saw? That is the honest version of the test,
because a rate computed on its own rows reproduces its own answer.

Design, fixed before running. Rows: LIRF fallback departures of 2025 with a label. Identity keys, all derived from
FLIGHT_mvt, which exists for fallback rows where every _flt column is absent: the leading letters (operator prefix)
and the full identifier. Split: months 1-6 learn the rate, months 7-12 are held out, then the reverse. Smoothing as
in features_stand_rate.py, 20 rows towards the base rate. Reported: AUC on the held-out half, the copy rate in the
top and bottom quintile of the predicted rate, and the share of held-out rows whose key was never seen.
Baselines on the same held-out rows: schedule delay alone, hour of day alone, and the identifier's occurrence count.

Diagnostic only. A separating result still needs a pre-registered six-pair screen inside the fallback model before
anything is built. Challenge data only.

Run from the repository root:  python3 -B research_280/rome_identity_copy_rate.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLUMNS = ["MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
           "TAXITIME_SEC_mvt", "AOBT_3_flt"]
OUTPUT = Path("research_280/results/rome_identity_copy_rate.json")
SMOOTHING = 20


def auc(labels, scores):
    labels = np.asarray(labels, bool)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    positives, negatives = labels.sum(), (~labels).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


frames = []
for path in sorted(glob.glob("data/training_2025-*.parquet")):
    table = pq.ParquetFile(path).read(columns=COLUMNS, use_threads=False).to_pandas()
    frames.append(table.loc[table.PHASE_mvt.eq("DEP") & table.ADEP_mvt.eq("LIRF")])
data = pd.concat(frames, ignore_index=True)
data["t"] = pd.to_datetime(data.MVT_TIME_UTC_mvt, utc=True)
data["sched_delay"] = (data.t - pd.to_datetime(data.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
data["y"] = data.TAXITIME_SEC_mvt.astype(float)
data = data.dropna(subset=["y"]).reset_index(drop=True)
data["copy"] = data.sched_delay.notna() & (data.y - data.sched_delay).abs().le(60)
data["month"] = data.t.dt.month
data["hour"] = data.t.dt.hour
flight = data.FLIGHT_mvt.astype(str).str.upper().str.strip()
data["prefix"] = flight.map(lambda v: (re.match(r"^[A-Z]+", v) or [""])[0] if re.match(r"^[A-Z]+", v) else "")
data["identifier"] = flight
data["occurrences"] = data.groupby("identifier", observed=True).MVT_ID_mvt.transform("size")

fallback = data.loc[data.AOBT_3_flt.isna()].reset_index(drop=True)
print(f"LIRF departures {len(data):,}, fallback {len(fallback):,}, copies {int(fallback['copy'].sum()):,} "
      f"({fallback['copy'].mean():.1%})")
report = {"lirf_departures": int(len(data)), "fallback_rows": int(len(fallback)),
          "copies": int(fallback["copy"].sum()), "base_rate": float(fallback["copy"].mean())}

halves = {"learn_1_6_test_7_12": (set(range(1, 7)), set(range(7, 13))),
          "learn_7_12_test_1_6": (set(range(7, 13)), set(range(1, 7)))}
for key in ("prefix", "identifier"):
    report[key] = {}
    print(f"\n=== identity key: {key}")
    for name, (learn_months, test_months) in halves.items():
        learn = fallback.loc[fallback.month.isin(learn_months)]
        test = fallback.loc[fallback.month.isin(test_months)].copy()
        base = learn["copy"].mean()
        grouped = learn.groupby(key, observed=True)["copy"].agg(["sum", "size"])
        rate = ((grouped["sum"] + SMOOTHING * base) / (grouped["size"] + SMOOTHING))
        test["rate"] = test[key].map(rate)
        unseen = test.rate.isna()
        test["rate"] = test.rate.fillna(base)
        quintiles = pd.qcut(test.rate.rank(method="first"), 5, labels=False, duplicates="drop")
        block = {"learn_rows": int(len(learn)), "test_rows": int(len(test)),
                 "unseen_key_share": float(unseen.mean()), "base_rate_learn": float(base),
                 "auc_rate": auc(test["copy"], test.rate.to_numpy()),
                 "auc_schedule_delay": auc(test["copy"], test.sched_delay.fillna(0).to_numpy()),
                 "auc_hour": auc(test["copy"], test.hour.to_numpy()),
                 "auc_occurrences": auc(test["copy"], -test.occurrences.to_numpy()),
                 "copy_rate_top_quintile": float(test.loc[quintiles == 4, "copy"].mean()),
                 "copy_rate_bottom_quintile": float(test.loc[quintiles == 0, "copy"].mean())}
        report[key][name] = block
        print(f"  {name}: test rows {block['test_rows']}, unseen keys {block['unseen_key_share']:.1%}")
        print(f"    AUC  identity rate {block['auc_rate']:.3f} | schedule delay {block['auc_schedule_delay']:.3f} "
              f"| hour {block['auc_hour']:.3f} | rarity {block['auc_occurrences']:.3f}")
        print(f"    copy rate  top quintile {block['copy_rate_top_quintile']:.1%} "
              f"| bottom quintile {block['copy_rate_bottom_quintile']:.1%} | base {base:.1%}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
print("\nwritten", OUTPUT)
