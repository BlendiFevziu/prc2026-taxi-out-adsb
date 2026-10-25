"""Label-free gate-retention columns for the main lane (research_720).

Pre-registered in research_720_retention/PREREG_retention.md before any fit. For a departure i with real push-back clock
t = AOBT_3_flt, the columns count other departures of the same airport (and calendar month) around t. Reads only
ADEP_mvt, RUNWAY_mvt, SCHED_TIME_UTC_mvt, AOBT_3_flt and MVT_TIME_UTC_mvt of other flights, never a label of row i.

  ret_wait_apt / ret_wait_bahn   departures with SCHED <= t and AOBT_3 > t (past schedule, not yet pushed); airport / runway
  ret_roll_apt / ret_roll_bahn   departures with AOBT_3 <= t < MVT (pushed, not yet airborne); airport / runway
  ret_push15 / ret_push60        departures with AOBT_3 in (t - 900 s, t) and (t - 3600 s, t)
  ret_delay15_s                  mean AOBT_3 - SCHED over the ret_push15 departures, missing if none
"""
import numpy as np
import pandas as pd

MERKMALE = ["ret_wait_apt", "ret_wait_bahn", "ret_roll_apt", "ret_roll_bahn", "ret_push15", "ret_push60", "ret_delay15_s"]
RAW_COLUMNS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt"]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def seconds(values):
    """UTC seconds as float, NaN where missing; independent of the datetime unit (ns or us)."""
    t = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    return ((t - EPOCH) / pd.Timedelta(seconds=1)).to_numpy(float)


def _wait(s, a):
    """For each member (t = a_i): #{s <= t} - #{max(s, a) <= t}; the member's own contribution cancels."""
    ss, mx = np.sort(s), np.sort(np.maximum(s, a))
    return np.searchsorted(ss, a, side="right") - np.searchsorted(mx, a, side="right")


def _roll(a, m):
    """For each member (t = a_i): #{a_j <= t < m_j} over the other members."""
    sa, sm = np.sort(a), np.sort(m)
    return (np.searchsorted(sa, a, side="right") - 1) - (np.searchsorted(sm, a, side="right") - (m <= a))


def _push(a, s):
    """For each member: pushes in the open windows (t - 900, t) and (t - 3600, t), and the mean delay in the first."""
    order = np.argsort(a, kind="stable")
    sa, delay = a[order], (a - s)[order]
    cum = np.concatenate([[0.0], np.cumsum(delay)])
    hi = np.searchsorted(sa, a, side="left")                               # strictly before t
    lo15, lo60 = np.searchsorted(sa, a - 900, side="right"), np.searchsorted(sa, a - 3600, side="right")
    n15, n60 = hi - lo15, hi - lo60
    mean = np.where(n15 > 0, (cum[hi] - cum[lo15]) / np.maximum(n15, 1), np.nan)
    return n15, n60, mean


def compute(raw):
    """The seven ret_ columns for every departure of raw that has AOBT_3, as a frame indexed by MVT_ID_mvt.
    raw: movements of whole calendar months (RAW_COLUMNS); computed per airport and per airport and runway."""
    d = raw[RAW_COLUMNS]
    d = d[d.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").copy()
    d["s"], d["a"], d["m"] = seconds(d.SCHED_TIME_UTC_mvt), seconds(d.AOBT_3_flt), seconds(d.MVT_TIME_UTC_mvt)
    d["month"] = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y%m")
    d["rwy"] = d.RUNWAY_mvt.astype(str).str.strip().str.upper().where(d.RUNWAY_mvt.notna(), None)
    d = d[d.a.notna() & d.s.notna() & d.m.notna()].reset_index(drop=True)
    out = pd.DataFrame(np.nan, index=d.index, columns=MERKMALE)
    for _, idx in d.groupby(["ADEP_mvt", "month"]).indices.items():
        s, a, m = d.s.to_numpy()[idx], d.a.to_numpy()[idx], d.m.to_numpy()[idx]
        out.iloc[idx, 0] = _wait(s, a)
        out.iloc[idx, 2] = _roll(a, m)
        out.iloc[idx, 4], out.iloc[idx, 5], out.iloc[idx, 6] = _push(a, s)
    for _, idx in d[d.rwy.notna()].groupby(["ADEP_mvt", "month", "rwy"]).indices.items():
        rows = d[d.rwy.notna()].index.to_numpy()[idx]
        s, a, m = d.s.to_numpy()[rows], d.a.to_numpy()[rows], d.m.to_numpy()[rows]
        out.iloc[rows, 1] = _wait(s, a)
        out.iloc[rows, 3] = _roll(a, m)
    out.index = d.MVT_ID_mvt.to_numpy()
    return out


def attach(ids, raw):
    """The seven ret_ columns for the movement ids in `ids` (aligned, NaN where a row is not a departure with AOBT_3)."""
    return compute(raw).reindex(pd.Index(ids)).reset_index(drop=True)
