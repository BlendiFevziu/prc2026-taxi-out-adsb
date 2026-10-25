"""Does a smoothed per-stand / per-operator copy rate sharpen the MAIN lane's copy gate, at Rome above all?

Background. `features_stand_rate.py` computes the fabrication rate per (ADEP_mvt, STAND_mvt), out of fold, and
its own docstring records that it beats the raw 1,631-level `STAND_mvt` category on the rows it serves
(1,777.6 s -> 1,752.6 s). It is given to the FALLBACK path only: `MERKMALE_RUECK` swaps it in for `STAND_mvt`,
while the main lane, and therefore the main lane's fabrication classifier, keeps the raw category.

`research_280/gate_calibration_by_airport.py` measured why that might matter. LIRF copies the schedule into the
block time on 18.9 % of main-lane rows against 5.5 to 10.4 % elsewhere, and a perfect gate there would move its
RMSE from 357.6 to 279.9, a 77.7 s gap where every other airport's is 2 to 6 s. The same file rejected a
per-airport exponent: reshaping the existing probability does not help, so the probability itself has to get
sharper.

This measures whether the smoothed rate carries information the current gate does not, before anything is
retrained. No fitting: the rate is a groupby on the ten learning months and is scored on January and July, which
the v47 ensemble held out, so no holdout row contributes to its own rate.

Copy definition follows main_model.py exactly: |BLOCK_TIME - SCHED_TIME| <= 10 s.

  python3 -B research_280/copy_rate_signal.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet"
OUTPUT = PROJECT / "research_280/results/copy_rate_signal.json"
COLUMNS = ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
           "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AIRCRAFT_OPERATOR_flt"]
SMOOTH = 20
ERF_HOCH = 2.5


def auc(score, label):
    """Rank AUC; ties averaged. Returns nan when a class is absent."""
    label = np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan")
    rank = pd.Series(np.asarray(score, float)).rank().to_numpy()
    positives, negatives = label.sum(), (~label).sum()
    return float((rank[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def rate(frame, keys, base):
    group = frame.groupby([frame[k].astype(str) for k in keys]).copy_flag.agg(["size", "sum"])
    return (group["sum"] + SMOOTH * base) / (group["size"] + SMOOTH)


def main():
    parts = []
    for path in sorted(PROJECT.glob("data/training_2025-*.parquet")):
        table = pq.ParquetFile(path).read(columns=COLUMNS, use_threads=False).to_pandas()
        parts.append(table.loc[table.PHASE_mvt.eq("DEP")])
    frame = pd.concat(parts, ignore_index=True)
    del parts
    frame["copy_flag"] = ((pd.to_datetime(frame.BLOCK_TIME_UTC_mvt, utc=True)
                           - pd.to_datetime(frame.SCHED_TIME_UTC_mvt, utc=True))
                          .dt.total_seconds().abs() <= 10).astype(int)
    month = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.month
    learn = frame.loc[~month.isin([1, 7])]
    print(f"{len(frame):,} departures, {len(learn):,} in the ten learning months, "
          f"copy rate learning {learn.copy_flag.mean():.2%} / held out {frame.loc[month.isin([1, 7])].copy_flag.mean():.2%}")

    base = float(learn.copy_flag.mean())
    stand = rate(learn, ["ADEP_mvt", "STAND_mvt"], base)
    operator = rate(learn, ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"], base)

    held = pd.read_parquet(BASELINE, columns=["MVT_ID_mvt", "ADEP_mvt", "y", "schedule", "gate"])
    held = held.merge(frame[["MVT_ID_mvt", "STAND_mvt", "AIRCRAFT_OPERATOR_flt", "copy_flag"]],
                      on="MVT_ID_mvt", how="left", validate="one_to_one")
    missing = held.copy_flag.isna().sum()
    held = held.loc[held.copy_flag.notna()].copy()
    held["q_stand"] = stand.reindex(pd.MultiIndex.from_arrays(
        [held.ADEP_mvt.astype(str), held.STAND_mvt.astype(str)])).to_numpy()
    held["q_op"] = operator.reindex(pd.MultiIndex.from_arrays(
        [held.ADEP_mvt.astype(str), held.AIRCRAFT_OPERATOR_flt.astype(str)])).to_numpy()
    held[["q_stand", "q_op"]] = held[["q_stand", "q_op"]].fillna(base)
    held["p_gate"] = held.gate.to_numpy() ** (1 / ERF_HOCH)
    print(f"held-out main-lane rows joined: {len(held):,} (dropped {missing:,} without a movement record)")

    rows = []
    for code, part in held.groupby("ADEP_mvt", observed=True):
        label = part.copy_flag.to_numpy(bool)
        rows.append({"airport": str(code), "rows": int(len(part)), "copy_rate": float(label.mean()),
                     "auc_current_gate": auc(part.p_gate, label),
                     "auc_stand_rate": auc(part.q_stand, label),
                     "auc_operator_rate": auc(part.q_op, label),
                     "auc_gate_plus_stand": auc(part.p_gate.rank() + part.q_stand.rank(), label),
                     "auc_gate_plus_both": auc(part.p_gate.rank() + part.q_stand.rank() + part.q_op.rank(), label)})
    table = pd.DataFrame(rows).sort_values("copy_rate", ascending=False)
    label = held.copy_flag.to_numpy(bool)
    pooled = {"rows": int(len(held)), "copy_rate": float(label.mean()),
              "auc_current_gate": auc(held.p_gate, label), "auc_stand_rate": auc(held.q_stand, label),
              "auc_operator_rate": auc(held.q_op, label),
              "auc_gate_plus_stand": auc(held.p_gate.rank() + held.q_stand.rank(), label),
              "auc_gate_plus_both": auc(held.p_gate.rank() + held.q_stand.rank() + held.q_op.rank(), label)}
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\npooled " + json.dumps({k: round(v, 4) for k, v in pooled.items()}))

    rome = held.loc[held.ADEP_mvt.eq("LIRF")]
    spread = rome.groupby(rome.STAND_mvt.astype(str)).copy_flag.agg(["size", "mean"])
    spread = spread.loc[spread["size"] >= 30].sort_values("mean")
    print(f"\nLIRF stands with at least 30 held-out rows: {len(spread)}, "
          f"copy rate {spread['mean'].min():.2%} to {spread['mean'].max():.2%}, "
          f"median {spread['mean'].median():.2%}")

    OUTPUT.write_text(json.dumps({"note": "rate built on the ten learning months only, scored on the held-out "
                                          "January and July; no model was fitted",
                                  "smoothing": SMOOTH, "base_rate": base,
                                  "by_airport": rows, "pooled": pooled,
                                  "lirf_stand_spread": {"stands": int(len(spread)),
                                                        "min": float(spread["mean"].min()),
                                                        "max": float(spread["mean"].max()),
                                                        "median": float(spread["mean"].median())}},
                                 indent=2) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
