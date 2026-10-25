"""Surface pressure, apron conflicts and relative delay at the reported off-block time (research_410_surface).

Pre-registered in research_410_surface/PREREG_surface.md before any fit. The anchor is AOBT_3_flt, the reported off-block
time every main-lane row carries; the true off-block (BLOCK_TIME of a departure) is the unknown for scored rows and is
never read for any departure. Arrivals' in-block times are read: they are given in the ranking file for every arrival
(features_arrivals.py uses them the same way).

  gp_dep_surface     other departures of the airport between their AOBT_3 and take-off at this flight's AOBT_3
  gp_arr_surface     arrivals of the airport between landing and in-block at this flight's AOBT_3
  ah_pier_moves_15   other departures (AOBT_3) and arrivals (in-block) at stands with the same letter prefix, within
                     15 minutes of this flight's AOBT_3 (the prefix, e.g. 'A' of 'A12', stands in for the apron or pier)
  sl_rel_delay_min   (AOBT_3 - SCHED) of this flight minus the median of the same quantity over the airport's departures
                     scheduled within 60 minutes either side (the flight included), in minutes

Plausibility windows keep broken records out of the reference sets: departures with 0 <= MVT - AOBT_3 <= 3 h, arrivals
with 0 <= BLOCK - MVT <= 2 h. Rows without AOBT_3 (fallback) get NaN.
"""
import numpy as np
import pandas as pd

MERKMALE = ["gp_dep_surface", "gp_arr_surface", "ah_pier_moves_15", "sl_rel_delay_min"]
RAW_COLUMNS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
               "AOBT_3_flt", "BLOCK_TIME_UTC_mvt"]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def seconds(values):
    t = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    return ((t - EPOCH) / pd.Timedelta(seconds=1)).to_numpy(float)


def prefix(stands):
    s = pd.Series(stands).astype(str).str.strip().str.upper()
    p = s.str.extract(r"^([A-Z]+)")[0]
    return p.fillna(s.str[:1]).to_numpy()


def _count_le(sorted_values, t):
    return np.searchsorted(sorted_values, t, side="right")


def build(raw):
    """The four columns for every departure in raw, indexed by MVT_ID_mvt. Departures' BLOCK_TIME is dropped first."""
    r = raw[RAW_COLUMNS].copy()
    ph = r.PHASE_mvt.astype(str).str.upper()
    dep, arr = r[ph.str.startswith("DEP")].copy(), r[ph.str.startswith("ARR")].copy()
    dep = dep.drop(columns="BLOCK_TIME_UTC_mvt")                       # never read a departure's block time
    dep["a"], dep["m"], dep["s"] = seconds(dep.AOBT_3_flt), seconds(dep.MVT_TIME_UTC_mvt), seconds(dep.SCHED_TIME_UTC_mvt)
    arr["m"], arr["b"] = seconds(arr.MVT_TIME_UTC_mvt), seconds(arr.BLOCK_TIME_UTC_mvt)
    dep["f"], arr["f"] = dep.ADEP_mvt.astype(str).to_numpy(), arr.ADES_mvt.astype(str).to_numpy()
    dep["p"], arr["p"] = prefix(dep.STAND_mvt), prefix(arr.STAND_mvt)
    ok_dep = np.isfinite(dep.a) & np.isfinite(dep.m) & (dep.m - dep.a).between(0, 3 * 3600)
    ok_arr = np.isfinite(arr.m) & np.isfinite(arr.b) & (arr.b - arr.m).between(0, 2 * 3600)
    out = pd.DataFrame(np.nan, index=dep.MVT_ID_mvt.to_numpy(), columns=MERKMALE)
    out.index.name = "MVT_ID_mvt"
    for f, d in dep.groupby("f"):
        q = d[np.isfinite(d.a)]
        if not len(q):
            continue
        t = q.a.to_numpy()
        rd = d[ok_dep.loc[d.index]]
        on = _count_le(np.sort(rd.a.to_numpy()), t) - _count_le(np.sort(rd.m.to_numpy()), t)
        on -= ok_dep.loc[q.index].to_numpy().astype(int)               # the flight itself is on the surface at its AOBT_3
        ra = arr[(arr.f == f) & ok_arr]
        on_a = _count_le(np.sort(ra.m.to_numpy()), t) - _count_le(np.sort(ra.b.to_numpy()), t)
        pier = np.zeros(len(q))
        for p, idx in pd.Series(np.arange(len(q))).groupby(q.p.to_numpy()).indices.items():
            da = np.sort(d.a[(d.p == p) & np.isfinite(d.a)].to_numpy())
            ab = np.sort(ra.b[ra.p == p].to_numpy())
            ti = t[idx]
            pier[idx] = (_count_le(da, ti + 900) - np.searchsorted(da, ti - 900, "left") - 1
                         + _count_le(ab, ti + 900) - np.searchsorted(ab, ti - 900, "left"))
        delay = pd.Series((q.a - q.s).to_numpy() / 60, index=pd.to_datetime(q.s.to_numpy(), unit="s"))
        order = np.argsort(delay.index.values, kind="stable")
        rolled = delay.iloc[order].rolling("7200s", center=True, min_periods=1).median().to_numpy()
        med = np.empty(len(q))
        med[order] = rolled
        ids = q.MVT_ID_mvt.to_numpy()
        out.loc[ids, "gp_dep_surface"] = on
        out.loc[ids, "gp_arr_surface"] = on_a
        out.loc[ids, "ah_pier_moves_15"] = pier
        out.loc[ids, "sl_rel_delay_min"] = delay.to_numpy() - med
    return out


def attach(dep, raw):
    """The four columns for the rows of dep (by MVT_ID_mvt), same index."""
    table = build(raw)
    return table.reindex(dep.MVT_ID_mvt.to_numpy()).set_axis(dep.index)[MERKMALE].astype(float)
