"""Neighbour linked-departure taxi context for departures without an AOBT anchor.

For a departure, the median network-anchored taxi time (MVT_TIME_UTC_mvt - AOBT_3_flt,
cached as ``ersatz_s``) of the OTHER linked departures at the same airport inside a
centred time window. Only movement time and the AOBT anchor are read. Both are present
in ranking.parquet, so no departure target, block time or arrival label is used.
Build it separately on every source split, as the research_250 builders do.

Motivation, measured 12.09.2026 on all 2025 labels: non-fabricated fallback departures
in hours whose linked neighbours taxied 30-45 minutes took a median 2,753 s against the
neighbours' 2,097 s. 2025 holds only 443 such fallback rows, and 30 above 45 minutes from
six airport-days, so a tree cannot learn that level from features alone. The screen
therefore also tests the context as an ANCHOR (predict y - nt_anchor).

Times are normalised to nanoseconds before any integer arithmetic. The v44 builders
treated microsecond storage as nanoseconds; do not reintroduce that.
"""
import numpy as np
import pandas as pd

REQUIRED = ("ADEP_mvt", "MVT_TIME_UTC_mvt", "ersatz_s", "hat_aobt")
WINDOWS_MIN = (30, 60, 180)
MIN_NEIGHBOURS = 5
MERKMALE = ["nt_med30", "nt_med60", "nt_med180", "nt_n60", "nt_n180", "nt_anchor"]
_NS_PER_MIN = 60 * 1_000_000_000


def _nanoseconds(values):
    stamps = pd.to_datetime(values, utc=True, errors="coerce")
    naive = stamps.dt.tz_convert(None).dt.as_unit("ns").to_numpy(dtype="datetime64[ns]")
    valid = ~np.isnat(naive)
    return naive.astype(np.int64), valid


def baue(dep, query=None):
    """Return MERKMALE indexed like ``dep``; rows outside ``query`` stay NaN.

    ``dep`` must hold every departure of one source split, linked and unlinked.
    ``query`` is an optional boolean mask. The per-row loop is meant for the
    fallback rows (a few thousand per split), not for all 1.7 million departures.
    """
    missing = [c for c in REQUIRED if c not in dep.columns]
    if missing:
        raise KeyError(f"missing columns {missing}")
    n = len(dep)
    t, t_ok = _nanoseconds(dep["MVT_TIME_UTC_mvt"])
    taxi = pd.to_numeric(dep["ersatz_s"], errors="coerce").to_numpy(dtype=float)
    linked = pd.to_numeric(dep["hat_aobt"], errors="coerce").to_numpy() == 1
    source = linked & t_ok & np.isfinite(taxi)
    wanted = np.ones(n, dtype=bool) if query is None else np.asarray(query, dtype=bool).copy()
    if wanted.shape != (n,):
        raise ValueError("query mask must match dep")
    wanted &= t_ok
    airports = dep["ADEP_mvt"].astype(str).to_numpy()
    out = {name: np.full(n, np.nan) for name in MERKMALE}
    rows = np.arange(n)
    for airport in np.unique(airports[wanted]):
        at_airport = airports == airport
        src = rows[source & at_airport]
        qry = rows[wanted & at_airport]
        src = src[np.argsort(t[src], kind="stable")]
        src_t, src_v = t[src], taxi[src]
        for minutes in WINDOWS_MIN:
            radius = np.int64(minutes) * _NS_PER_MIN
            left = np.searchsorted(src_t, t[qry] - radius, side="left")
            right = np.searchsorted(src_t, t[qry] + radius, side="right")
            med = np.full(len(qry), np.nan)
            cnt = np.zeros(len(qry))
            for k, row in enumerate(qry):
                values = src_v[left[k]:right[k]][src[left[k]:right[k]] != row]  # never itself
                cnt[k] = values.size
                if values.size:
                    med[k] = np.median(values)
            out[f"nt_med{minutes}"][qry] = med
            if minutes in (60, 180):
                out[f"nt_n{minutes}"][qry] = cnt
        out["nt_anchor"][qry] = np.where(
            out["nt_n60"][qry] >= MIN_NEIGHBOURS, out["nt_med60"][qry],
            np.where(out["nt_n180"][qry] >= MIN_NEIGHBOURS, out["nt_med180"][qry], np.nan))
    return pd.DataFrame(out, index=dep.index)[MERKMALE]
