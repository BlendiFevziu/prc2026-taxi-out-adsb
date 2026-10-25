"""Scale-free rate of the legacy fallback count `auslast`.

main_model.auslastung counts the other departures at the same airport within +/-FENSTER_S
on a clock built as int64(microseconds) // 1e9, that is in units of 1,000 s. The intended
+/-30 min window is therefore +/-1,800,000 s = +/-20.83 days, and the count grows with the
days of data the table holds. Measured 12.09.2026 on January/July fallback rows: 0.61 of the
twelve-month training scale in the 2026 ranking file, 0.65 in two-month 2025 tables.

rate() divides the legacy count by the days of that window covered by the table's calendar
months and rescales to the full window, so a value means the same in a twelve-month table,
a two-month table and the ranking file. Nothing in main_model, research_250 or any scored
bundle changes.
"""
import numpy as np
import pandas as pd

LEGACY_HALF_WINDOW_S = 1800 * 1000
FULL_WINDOW_DAYS = 2 * LEGACY_HALF_WINDOW_S / 86400
_NS = 1_000_000_000


def _ns(times):
    stamps = pd.to_datetime(pd.Series(times), utc=True, errors="coerce")
    arr = stamps.dt.tz_convert(None).dt.as_unit("ns").to_numpy(dtype="datetime64[ns]")
    ok = ~np.isnat(arr)
    values = arr.astype(np.int64)
    values[~ok] = 0
    return values, ok


def month_spans(year_months):
    """Merged [start_ns, end_ns) spans of the given (year, month) calendar months."""
    spans = []
    for year, month in sorted(set((int(y), int(m)) for y, m in year_months)):
        start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        if spans and start.value <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end.value)
        else:
            spans.append([start.value, end.value])
    return [tuple(s) for s in spans]


def spans_from_times(times, min_rows=100, min_fraction=.95):
    """Calendar months present in a table.

    Months with fewer than ``min_rows`` departures are ignored: two source files hold records
    stamped seconds before their month. A kept month must be essentially fully observed,
    otherwise coverage would be overstated, so a partial month raises.
    """
    t = pd.to_datetime(pd.Series(times), utc=True, errors="coerce").dropna()
    counts = pd.Series(list(zip(t.dt.year, t.dt.month))).value_counts()
    kept = [key for key, n in counts.items() if n >= min_rows]
    for year, month in kept:
        start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        inside = t[(t >= start) & (t < end)]
        if (inside.max() - inside.min()) / (end - start) < min_fraction:
            raise ValueError(f"{year}-{month:02d} is only partly observed")
    return month_spans(kept)


def available_days(times, spans, half_window_s=LEGACY_HALF_WINDOW_S):
    """Days of the legacy window [t - half, t + half] covered by ``spans``; NaN for NaT."""
    t, ok = _ns(times)
    lo, hi = t - half_window_s*_NS, t + half_window_s*_NS
    covered = np.zeros(len(t))
    for start, end in spans:
        covered += np.clip(np.minimum(hi, end) - np.maximum(lo, start), 0, None)
    days = covered / (86400*_NS)
    days[~ok] = np.nan
    return days


def rate(auslast, times, spans):
    """Legacy count per covered day, on the full-window count scale."""
    days = available_days(times, spans)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(auslast, dtype=float) / days * FULL_WINDOW_DAYS
    out[~(days > 0)] = np.nan
    return out
