"""Label-free proxies for the NM findings, built from the challenge's own NM clocks.

Pre-registered in research_280/NM_PROXY_INVENTORY.md (23.09.2026). The NM add-on's value sits in which off-block
clock types NM received, how they evolved and how much traffic a flight generated. The challenge carries only the
final value of six NM clocks, so these three families ask what those finals still reveal:

  H1  provenance: AOBT_3 exactly equal to a planned clock, i.e. NM carried the plan over as its "actual"
  H2  precision: whether a clock carries a non-zero seconds field (a system stamp does, a planned one does not)
  H3  NM-side reporting regime: the share of OTHER main-lane departures at the same airport, strictly earlier in
      the last 60 minutes and earlier the same UTC day, whose AOBT_3 equals a planned clock

Only the columns in REQUIRED are read. No target, block time or taxi time is used, so the features exist for the
2026 ranking rows exactly as for 2025. Build every source split separately, as the research_250 builders do.

Times are normalised to nanoseconds before any integer arithmetic. The v44 builders treated microsecond storage
as nanoseconds; do not reintroduce that.
"""
import numpy as np
import pandas as pd

PLAN = ("EOBT_1_flt", "IOBT_flt", "LOBT_flt")
REQUIRED = ("ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_1_flt", "ARVT_3_flt") + PLAN
H1 = ["nmp_aobt_eq_eobt1", "nmp_aobt_eq_iobt", "nmp_aobt_eq_lobt"]
H2 = ["nmp_aobt3_sec", "nmp_arvt1_sec", "nmp_arvt3_sec"]
H3 = ["nmp_plan_share_60", "nmp_plan_share_day", "nmp_plan_n_60"]
FAMILIES = {"H1": H1, "H2": H2, "H3": H3}
MERKMALE = H1 + H2 + H3
WINDOW_MIN = 60
_NS_PER_MIN = 60 * 1_000_000_000
_NS_PER_DAY = 24 * 60 * _NS_PER_MIN


def _nanoseconds(values):
    stamps = pd.to_datetime(values, utc=True, errors="coerce")
    naive = stamps.dt.tz_convert(None).dt.as_unit("ns").to_numpy(dtype="datetime64[ns]")
    valid = ~np.isnat(naive)
    return naive.astype(np.int64), valid


def _equal(a, a_ok, b, b_ok):
    """1.0 where both clocks exist and agree to the nanosecond, 0.0 where both exist and differ, else NaN."""
    both = a_ok & b_ok
    return np.where(both, (a == b).astype(float), np.nan)


def _has_seconds(t, ok):
    """1.0 where the clock is not a whole-minute stamp, 0.0 where it is, NaN where it is missing."""
    return np.where(ok, ((t % _NS_PER_MIN) != 0).astype(float), np.nan)


def baue(dep):
    """Return MERKMALE indexed like ``dep``, which must hold every departure of ONE source split."""
    missing = [c for c in REQUIRED if c not in dep.columns]
    if missing:
        raise KeyError(f"missing columns {missing}")
    n = len(dep)
    t, t_ok = _nanoseconds(dep["MVT_TIME_UTC_mvt"])
    aobt, aobt_ok = _nanoseconds(dep["AOBT_3_flt"])
    out = {}

    plans = {}
    for column, name in zip(PLAN, ("nmp_aobt_eq_eobt1", "nmp_aobt_eq_iobt", "nmp_aobt_eq_lobt")):
        clock, ok = _nanoseconds(dep[column])
        out[name] = _equal(aobt, aobt_ok, clock, ok)
        plans[name] = out[name]

    out["nmp_aobt3_sec"] = _has_seconds(aobt, aobt_ok)
    for column, name in (("ARVT_1_flt", "nmp_arvt1_sec"), ("ARVT_3_flt", "nmp_arvt3_sec")):
        clock, ok = _nanoseconds(dep[column])
        out[name] = _has_seconds(clock, ok)

    # H3. A source row is a departure whose provenance flag is defined; any plan match counts.
    stacked = np.vstack([plans[k] for k in H1])
    defined = ~np.all(np.isnan(stacked), axis=0)
    plan_any = np.nansum(np.nan_to_num(stacked, nan=0.0), axis=0) > 0
    source = defined & t_ok
    share60 = np.full(n, np.nan)
    shareday = np.full(n, np.nan)
    count60 = np.full(n, np.nan)
    airports = dep["ADEP_mvt"].astype(str).to_numpy()
    rows = np.arange(n)
    for airport in np.unique(airports):
        at = airports == airport
        src = rows[source & at]
        qry = rows[t_ok & at]
        if not len(qry):
            continue
        src = src[np.argsort(t[src], kind="stable")]
        src_t = t[src]
        cum = np.concatenate([[0], np.cumsum(plan_any[src].astype(np.int64))])
        tq = t[qry]
        # Strictly earlier: the right edge is the first source row AT the query time, so the row itself
        # and anything at the same instant are excluded.
        right = np.searchsorted(src_t, tq, side="left")
        left60 = np.searchsorted(src_t, tq - np.int64(WINDOW_MIN) * _NS_PER_MIN, side="left")
        leftday = np.searchsorted(src_t, tq - (tq % _NS_PER_DAY), side="left")
        n60 = right - left60
        nday = right - leftday
        with np.errstate(invalid="ignore", divide="ignore"):
            share60[qry] = np.where(n60 > 0, (cum[right] - cum[left60]) / n60, np.nan)
            shareday[qry] = np.where(nday > 0, (cum[right] - cum[leftday]) / nday, np.nan)
        count60[qry] = n60
    out["nmp_plan_share_60"] = share60
    out["nmp_plan_share_day"] = shareday
    out["nmp_plan_n_60"] = count60
    return pd.DataFrame(out, index=dep.index)[MERKMALE]
