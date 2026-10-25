"""Which event does MVT_ID_mvt follow? Descriptive, January 2025 raw file, no model, no leaderboard feedback.

If the airport assigns the movement ID when it records the off-block event, the ID order of departures carries their
block-time order, and neighbouring movements with known times could be interpolated into a label-free block-time estimate.
Checks: whether IDs form one sequence across airports and phases; per airport and day, Spearman correlation of departure
IDs with BLOCK_TIME, MVT_TIME, SCHED_TIME and AOBT_3; order agreement of consecutive departures by ID; and leave-one-out
interpolation error of each time from the two ID-neighbouring departures at the same airport.

Run:  python3 -B research_240/movement_id_timing_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
ORIGIN = pd.Timestamp("2025-01-01", tz="UTC")
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt"]
raw = pd.read_parquet(P / "data/training_2025-01-01_2025-02-01.parquet", columns=cols)
sec = lambda s: (pd.to_datetime(s, utc=True) - ORIGIN).dt.total_seconds().to_numpy(float)
phase = raw.PHASE_mvt.astype(str).str.upper().str[:3].to_numpy()
airport = np.where(phase == "DEP", raw.ADEP_mvt.astype(str), raw.ADES_mvt.astype(str))
ids = raw.MVT_ID_mvt.to_numpy(float)
print("rows", len(raw), "unique IDs", pd.Series(ids).nunique(), "integer-valued", bool(np.all(np.mod(ids[np.isfinite(ids)], 1) == 0)),
      "min", np.nanmin(ids), "max", np.nanmax(ids))
order = np.argsort(ids, kind="stable")
switch_airport = np.mean(airport[order][1:] != airport[order][:-1])
switch_phase = np.mean(phase[order][1:] != phase[order][:-1])
print(f"consecutive IDs: airport changes {switch_airport:.1%}, phase changes {switch_phase:.1%} (10 airports interleaved would give about 90 %)")
t = {"block": sec(raw.BLOCK_TIME_UTC_mvt), "mvt": sec(raw.MVT_TIME_UTC_mvt), "sched": sec(raw.SCHED_TIME_UTC_mvt), "aobt3": sec(raw.AOBT_3_flt)}
all_mvt_order = pd.Series(ids).rank().corr(pd.Series(t["mvt"]).rank())
print(f"all rows: Spearman(ID, MVT_TIME) {all_mvt_order:.4f}")
for ph in ("DEP", "ARR"):
    m = phase == ph
    frame = pd.DataFrame({"airport": airport[m], "day": np.floor(t["mvt"][m] / 86400), "id": ids[m], **{k: v[m] for k, v in t.items()}})
    stats = {k: [] for k in t}
    weights = []
    for _, g in frame.groupby(["airport", "day"]):
        if len(g) < 50:
            continue
        weights.append(len(g))
        r = g.id.rank()
        for k in t:
            ok = g[k].notna()
            stats[k].append(r[ok].corr(g.loc[ok, k].rank()) if ok.sum() > 30 else np.nan)
    print(ph, "airport-days", len(weights), "weighted mean within-day Spearman(ID, time):",
          {k: round(float(np.nansum(np.array(v) * np.array(weights)) / np.sum(np.array(weights)[np.isfinite(v)])), 4) for k, v in stats.items()})
    # consecutive departures (or arrivals) by ID at the same airport: order agreement and leave-one-out interpolation
    frame = frame.sort_values(["airport", "id"], kind="stable").reset_index(drop=True)
    same_prev = frame.airport.eq(frame.airport.shift(1)).to_numpy()
    same_next = frame.airport.eq(frame.airport.shift(-1)).to_numpy()
    for k in t:
        v = frame[k].to_numpy(float)
        prev, nxt = np.roll(v, 1), np.roll(v, -1)
        agree = same_prev & np.isfinite(v) & np.isfinite(prev)
        interp_ok = same_prev & same_next & np.isfinite(prev) & np.isfinite(nxt) & np.isfinite(v)
        err = np.abs(v[interp_ok] - (prev[interp_ok] + nxt[interp_ok]) / 2)
        print(f"   {k:6s} order agreement with previous ID {np.mean(v[agree] >= prev[agree]):.3f}   leave-one-out neighbour-mean error: median {np.median(err):7.0f} s, p90 {np.quantile(err, .9):7.0f} s")
dep = phase == "DEP"
gap = t["aobt3"][dep] - t["block"][dep]
print("for scale, departures |AOBT_3 - BLOCK|: median", float(np.nanmedian(np.abs(gap))), "s; |MVT - BLOCK| median", float(np.nanmedian(np.abs(t["mvt"][dep] - t["block"][dep]))), "s")
