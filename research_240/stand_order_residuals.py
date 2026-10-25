"""Stand order against NM off-block: does another arrival block in at the departure stand before AOBT_3?

Descriptive, January/July 2025 holdout of the v45 candidate-stage ensemble (main lane, trained without January/July).
Label-free at rank time: arrival in-block times and stands are complete in ranking.parquet; departures carry AOBT_3
and MVT_TIME.
  t1, t2   in-block times of the last and second-to-last arrival at the departure's stand (airport + STAND_mvt) at or
           before the departure's MVT_TIME
  s1 = t1 - AOBT_3 (seconds)
  between  departures from the same stand with MVT_TIME strictly between t2 and the departure's MVT_TIME
Physical reading: one aircraft per stand. An arrival blocking in before our AOBT_3 while no other departure left the
stand since the arrival before it implies our aircraft had already left, or the stand code is shared.
Reports per cell (s1 bin x between): rows, rest = AOBT_3 - BLOCK_TIME (mean, share above 1,800 s), share of
BLOCK_TIME before t1, ensemble residual bias, RMSE and MSE share; enrichment of the not-copied rest > 1,800 s class;
bin-bias bound of the ensemble residual in-sample and cross-fitted between January and July (0.3 s of main RMSE is
about 130 MSE). No model fit, no leaderboard feedback.

Run:  python3 -B research_240/stand_order_residuals.py <output.json>
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
FILES = [P / "data/training_2025-01-01_2025-02-01.parquet", P / "data/training_2025-07-01_2025-08-01.parquet"]
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "AOBT_3_flt", "TAXITIME_SEC_mvt", "SCHED_TIME_UTC_mvt"]
EDGES = [-np.inf, -3600, -1800, -900, -300, 0, 300, 900, np.inf]
ORIGIN = pd.Timestamp("2025-01-01", tz="UTC")


def seconds(values):
    return (pd.to_datetime(values, utc=True) - ORIGIN).dt.total_seconds().to_numpy(float)


raw = pd.concat([pd.read_parquet(f, columns=COLS) for f in FILES], ignore_index=True)
phase = raw.PHASE_mvt.astype(str).str.upper()
arr = raw[phase.str.startswith("ARR").to_numpy()]
dep = raw[phase.str.startswith("DEP").to_numpy()]
arr_key = (arr.ADES_mvt.astype(str) + "|" + arr.STAND_mvt.astype(str)).to_numpy()
dep_key = (dep.ADEP_mvt.astype(str) + "|" + dep.STAND_mvt.astype(str)).to_numpy()
arr_frame = pd.DataFrame({"k": arr_key, "t": seconds(arr.BLOCK_TIME_UTC_mvt)}).dropna()
dep_frame = pd.DataFrame({"k": dep_key, "t": seconds(dep.MVT_TIME_UTC_mvt)}).dropna()
arrivals = {k: np.sort(g.t.to_numpy()) for k, g in arr_frame.groupby("k")}
departures = {k: np.sort(g.t.to_numpy()) for k, g in dep_frame.groupby("k")}

pred = pd.read_parquet(P / ".cache/clock_queue_release_20260912/candidate/holdout_predictions.parquet")
pred = pred[pred.hat_aobt.eq(1)]
d = pd.DataFrame({"MVT_ID_mvt": dep.MVT_ID_mvt.to_numpy(), "k": dep_key, "airport": dep.ADEP_mvt.astype(str).to_numpy(),
                  "mvt": seconds(dep.MVT_TIME_UTC_mvt), "aobt": seconds(dep.AOBT_3_flt), "block": seconds(dep.BLOCK_TIME_UTC_mvt),
                  "sched": seconds(dep.SCHED_TIME_UTC_mvt), "y": pd.to_numeric(dep.TAXITIME_SEC_mvt, errors="coerce").to_numpy(float)})
main = pred[["MVT_ID_mvt", "prediction"]].merge(d, on="MVT_ID_mvt", how="inner", validate="one_to_one").reset_index(drop=True)
n = len(main)
t1, t2, between = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
for key, idx in main.groupby("k").indices.items():
    a = arrivals.get(key)
    if a is None:
        continue
    t = main.mvt.to_numpy()[idx]
    pos = np.searchsorted(a, t, side="right")
    t1[idx] = np.where(pos >= 1, a[np.maximum(pos - 1, 0)], np.nan)
    second = np.where(pos >= 2, a[np.maximum(pos - 2, 0)], np.nan)
    t2[idx] = second
    dd = departures.get(key, np.array([]))
    lo = np.searchsorted(dd, np.nan_to_num(second, nan=np.inf), side="right")
    hi = np.searchsorted(dd, t, side="left")
    between[idx] = np.where(pos >= 2, hi - lo, np.nan)

y = main.y.to_numpy()
rest = main.aobt.to_numpy() - main.block.to_numpy()
residual = y - main.prediction.to_numpy()
s1 = t1 - main.aobt.to_numpy()
copied = np.abs(y - (main.mvt.to_numpy() - main.sched.to_numpy())) <= 10
neither = (~copied) & (rest > 1800)
month = pd.to_datetime(main.mvt, unit="s", origin=ORIGIN.tz_convert(None)).dt.month.to_numpy()
month = np.where(np.isin(month, [1, 7]), month, month % 12 + 1)
s1_bin = pd.cut(pd.Series(s1), EDGES).astype(str).to_numpy()
between_label = np.where(np.isnan(between), "no2arr", np.where(between == 0, "b0", "b1+"))
cell = np.char.add(np.char.add(s1_bin.astype(str), " | "), between_label.astype(str))
base_mse = float(np.mean(residual**2))


def bias_bound(keys):
    frame = pd.DataFrame({"k": keys, "r": residual, "m": month})
    inside = frame.groupby("k").r.transform("mean").to_numpy()
    cross = np.zeros(n)
    for m, other in ((1, 7), (7, 1)):
        means = frame[frame.m == other].groupby("k").r.mean()
        rows = month == m
        cross[rows] = pd.Series(keys[rows]).map(means).fillna(0).to_numpy()
    return {"in_sample": float(base_mse - np.mean((residual - inside)**2)), "cross_fitted": float(base_mse - np.mean((residual - cross)**2))}


rows_out = []
for c in sorted(set(cell)):
    m = cell == c
    rows_out.append({"cell": c, "rows": int(m.sum()), "share": float(m.mean()), "rest_mean_s": float(np.nanmean(rest[m])),
                     "rest_over_1800_share": float(np.mean(rest[m] > 1800)), "block_before_t1_share": float(np.nanmean(main.block.to_numpy()[m] < t1[m])) if np.isfinite(t1[m]).any() else None,
                     "neither_enrichment": float(neither[m].mean() / neither.mean()) if neither.mean() else None,
                     "residual_bias_s": float(residual[m].mean()), "rmse_s": float(np.sqrt(np.mean(residual[m]**2))),
                     "mse_share": float(np.sum(residual[m]**2) / np.sum(residual**2))})
report = {"note": "descriptive; v45 candidate-stage holdout ensemble, January/July 2025 main lane", "rows": n,
          "main_rmse_s": float(np.sqrt(base_mse)), "neither_rows": int(neither.sum()), "neither_mse_share": float(np.sum(residual[neither]**2) / np.sum(residual**2)),
          "cells": rows_out,
          "bias_bound_mse": {"cell": bias_bound(cell), "airport_x_cell": bias_bound(np.char.add(np.char.add(main.airport.to_numpy().astype(str), "|"), cell))}}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ("rows", "main_rmse_s", "neither_rows", "neither_mse_share", "bias_bound_mse")}, indent=1))
for r in rows_out:
    print(f"{r['cell']:>32} rows {r['rows']:>7,} {r['share']:6.1%} rest {r['rest_mean_s']:+7.0f} >30min {r['rest_over_1800_share']:6.1%} "
          f"blk<t1 {r['block_before_t1_share'] if r['block_before_t1_share'] is None else round(r['block_before_t1_share'], 3)} "
          f"neither x{r['neither_enrichment']:.1f} bias {r['residual_bias_s']:+7.1f} rmse {r['rmse_s']:6.0f} mse {r['mse_share']:6.1%}")
