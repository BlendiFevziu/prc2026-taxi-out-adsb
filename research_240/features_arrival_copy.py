"""Arrival-side signals for copied departure block times, label-free for departures.

Airports sometimes write the scheduled time into BLOCK_TIME when no actual time was recorded.
Arrival rows keep BLOCK_TIME and SCHED_TIME in the ranking file, so whether an arrival's
in-block equals its schedule (within 10 s) is observable at prediction time. For each departure,
relative to its off-block reference time (AOBT_3 = MVT_TIME - ersatz_s, main-lane rows only):

  ac_inbound_copied     the last arrival at the same airport and stand that landed at least 10 min
                        and at most 24 h before the reference time: 1 if its block time equals its
                        schedule, else 0; missing if there is none
  ac_inbound_gap_min    minutes from that landing to the reference time
  ac_stand_share_7d     share of copied arrivals at the same airport and stand among arrivals that
                        landed in the 7 days up to 10 min before the reference time (at least 3)
  ac_operator_share_7d  the same for the departure's aircraft operator at that airport (at least 5)

Build training and held-out or ranking references independently, each from the arrivals of its
own source table.
"""
import numpy as np
import pandas as pd

MERKMALE = ["ac_inbound_copied", "ac_inbound_gap_min", "ac_stand_share_7d", "ac_operator_share_7d"]
ARRIVAL_COLUMNS = ["PHASE_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                   "SCHED_TIME_UTC_mvt", "AIRCRAFT_OPERATOR_flt"]
COPY_TOLERANCE_S = 10
LEAD = pd.Timedelta(minutes=10)
INBOUND_WINDOW = pd.Timedelta(hours=24)
SHARE_WINDOW = pd.Timedelta(days=7)
MIN_STAND, MIN_OPERATOR = 3, 5


def _utc(values):
    return pd.to_datetime(pd.Series(values).reset_index(drop=True), utc=True, errors="coerce").dt.as_unit("ns")


def _key(airport, other):
    """'airport|value', missing where the value is missing."""
    airport = pd.Series(airport).reset_index(drop=True).astype(str)
    other = pd.Series(other).reset_index(drop=True)
    text = other.astype(str)
    present = other.notna().to_numpy() & ~text.isin(["nan", "None", "", "<NA>"]).to_numpy()
    return (airport + "|" + text).where(present)


def arrivals_table(raw):
    """Arrival rows of a raw movement table: landing time, copied flag and grouping keys."""
    phase = raw.PHASE_mvt.astype(str).str.upper().str[:3].to_numpy()
    arrivals = raw[phase == "ARR"].reset_index(drop=True)
    out = pd.DataFrame({"t": _utc(arrivals.MVT_TIME_UTC_mvt)})
    gap = (_utc(arrivals.BLOCK_TIME_UTC_mvt) - _utc(arrivals.SCHED_TIME_UTC_mvt)).dt.total_seconds().abs()
    out["copied"] = np.where(gap.notna(), (gap <= COPY_TOLERANCE_S).astype(float), np.nan)
    out["stand_key"] = _key(arrivals.ADES_mvt, arrivals.STAND_mvt)
    out["operator_key"] = _key(arrivals.ADES_mvt, arrivals.AIRCRAFT_OPERATOR_flt)
    return out.dropna(subset=["t", "copied"]).reset_index(drop=True)


def _cumulative(arrivals, key_col):
    right = arrivals.dropna(subset=[key_col]).sort_values([key_col, "t"], kind="mergesort").reset_index(drop=True)
    right["n"] = right.groupby(key_col).cumcount() + 1
    right["k"] = right.groupby(key_col).copied.cumsum()
    return right[[key_col, "t", "n", "k"]].rename(columns={key_col: "key"}).sort_values("t", kind="mergesort")


def _asof_counts(queries, times, right):
    """Cumulative arrival and copied counts at or before each query time, per key."""
    frame = pd.DataFrame({"key": queries.key.to_numpy(), "_row": queries._row.to_numpy()})
    frame["at"] = times.to_numpy() if not hasattr(times, "dt") else times.reset_index(drop=True)
    frame = frame.sort_values("at", kind="mergesort")
    joined = pd.merge_asof(frame, right, left_on="at", right_on="t", by="key", direction="backward")
    n = pd.Series(joined.n.to_numpy(float), index=joined._row.to_numpy()).fillna(0.0)
    k = pd.Series(joined.k.to_numpy(float), index=joined._row.to_numpy()).fillna(0.0)
    return n, k


def _window_share(arrivals, key_col, keys, end, minimum):
    share = np.full(len(end), np.nan)
    queries = pd.DataFrame({"key": keys.to_numpy(), "_row": np.arange(len(end))})
    queries["end"] = end
    queries["start"] = end - SHARE_WINDOW
    queries = queries[queries.key.notna().to_numpy() & queries.end.notna().to_numpy()].reset_index(drop=True)
    if queries.empty:
        return share
    right = _cumulative(arrivals, key_col)
    n_end, k_end = _asof_counts(queries, queries.end, right)
    n_start, k_start = _asof_counts(queries, queries.start, right)
    rows = n_end.index.to_numpy()
    count = n_end.to_numpy() - n_start.reindex(rows).to_numpy()
    copied = k_end.to_numpy() - k_start.reindex(rows).to_numpy()
    valid = count >= minimum
    share[rows[valid]] = copied[valid] / count[valid]
    return share


def features_for(departures, arrivals):
    """Arrival-copy features for departure rows (ADEP_mvt, STAND_mvt, AIRCRAFT_OPERATOR_flt,
    MVT_TIME_UTC_mvt, ersatz_s), in their order."""
    n = len(departures)
    ersatz = pd.to_numeric(pd.Series(departures.ersatz_s).reset_index(drop=True), errors="coerce").to_numpy(float)
    reference = _utc(departures.MVT_TIME_UTC_mvt) - pd.to_timedelta(ersatz, unit="s")
    end = reference - LEAD
    stand_keys = _key(departures.ADEP_mvt, departures.STAND_mvt)
    operator_keys = _key(departures.ADEP_mvt, departures.AIRCRAFT_OPERATOR_flt)
    out = pd.DataFrame(np.nan, index=np.arange(n), columns=MERKMALE)

    left = pd.DataFrame({"key": stand_keys.to_numpy(), "_row": np.arange(n)})
    left["end"] = end
    left["reference"] = reference
    left = left[left.key.notna().to_numpy() & left.end.notna().to_numpy()]
    if len(left):
        right = (arrivals.dropna(subset=["stand_key"])[["stand_key", "t", "copied"]]
                 .rename(columns={"stand_key": "key"}).sort_values("t", kind="mergesort"))
        joined = pd.merge_asof(left.sort_values("end", kind="mergesort"), right, left_on="end", right_on="t",
                               by="key", direction="backward", tolerance=INBOUND_WINDOW - LEAD)
        rows = joined._row.to_numpy()
        out.loc[rows, "ac_inbound_copied"] = joined.copied.to_numpy(float)
        out.loc[rows, "ac_inbound_gap_min"] = (joined.reference - joined.t).dt.total_seconds().to_numpy() / 60
    out["ac_stand_share_7d"] = _window_share(arrivals, "stand_key", stand_keys, end, MIN_STAND)
    out["ac_operator_share_7d"] = _window_share(arrivals, "operator_key", operator_keys, end, MIN_OPERATOR)
    return out
