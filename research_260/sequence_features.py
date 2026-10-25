"""Versioned neighbour tensors; the team's origin grouping retained as a control.

local: departure movements use ADEP, arrival movements use ADES. origin: reproduce
7095116's ADEP grouping exactly, including remote arrivals. Both use 12 preceding
and 12 following rows within 90 minutes, not the 24 smallest absolute time gaps.
No departure label or departure block time contributes to the tensor. References
must be built independently for each source split. Future movements are supplied
by this retrospective challenge; this is not a causal live-operations feature.
"""
import numpy as np
import pandas as pd

SCHEMA = "sequence_airport_v1"
RAW = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
       "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "STAND_mvt",
       "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "AOBT_3_flt", "ARVT_3_flt"]
FIELDS = ["relative_time", "departure", "arrival", "nm_taxi", "has_aobt", "arrival_taxi",
          "nm_arrival_gap", "has_arvt", "arrival_copy", "same_stand", "same_runway",
          "same_operator", "schedule_delay", "flight_id_offset", "has_flight_id", "same_type"]


def secs(values):
    return (pd.to_datetime(values, utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()


def neighbours(raw, mvt_ids, mode="local", k=24, window_min=90.):
    if mode not in ("origin", "local"):
        raise ValueError("mode must be origin or local")
    if k <= 0 or k % 2 or window_min <= 0:
        raise ValueError("k must be positive/even and window positive")
    if not raw.MVT_ID_mvt.is_unique:
        raise ValueError("duplicate movement IDs")
    raw = raw[RAW].copy()
    if not raw.PHASE_mvt.isin(["ARR", "DEP"]).all():
        raise ValueError("unknown movement phase")
    # Preserve stable equal-time ordering for an exact original-mode control.
    raw["_airport"] = raw.ADEP_mvt.astype(object)
    if mode == "local":
        arrivals = raw.PHASE_mvt.eq("ARR")
        raw.loc[arrivals, "_airport"] = raw.loc[arrivals, "ADES_mvt"].astype(object)
    raw = raw.sort_values(["_airport", "MVT_TIME_UTC_mvt"], kind="mergesort").reset_index(drop=True)
    t = secs(raw.MVT_TIME_UTC_mvt)
    is_dep, is_arr = raw.PHASE_mvt.eq("DEP").to_numpy(), raw.PHASE_mvt.eq("ARR").to_numpy()
    aobt, arvt, sched = secs(raw.AOBT_3_flt), secs(raw.ARVT_3_flt), secs(raw.SCHED_TIME_UTC_mvt)
    block = secs(raw.BLOCK_TIME_UTC_mvt.where(is_arr))
    nm_taxi = np.where(is_dep & np.isfinite(aobt), np.clip((t-aobt)/60, -30, 120)/60, 0.)
    has_aobt = (is_dep & np.isfinite(aobt)).astype(np.float32)
    taxi_in = np.where(is_arr & np.isfinite(block), np.clip((block-t)/60, -30, 120)/60, 0.)
    nm_arr = np.where(is_arr & np.isfinite(arvt), np.clip((arvt-t)/60, -60, 60)/60, 0.)
    has_arvt = (is_arr & np.isfinite(arvt)).astype(np.float32)
    copied = (is_arr & np.isfinite(block) & np.isfinite(sched) & (np.abs(block-sched) <= 10)).astype(np.float32)
    sdel = np.where(np.isfinite(sched), np.clip((t-sched)/60, -180, 180)/180, 0.)
    fid = raw.FLIGHT_ID_mvt.to_numpy(float)
    stand, rwy = raw.STAND_mvt.astype(str).to_numpy(), raw.RUNWAY_mvt.astype(str).to_numpy()
    oper = raw.AIRCRAFT_OPERATOR_flt.astype(str).to_numpy()
    typ, airport = raw.AIRCRAFT_TYPE_mvt.astype(str).to_numpy(), raw._airport.to_numpy()
    own = pd.Series(np.arange(len(raw)), index=raw.MVT_ID_mvt.to_numpy()).loc[mvt_ids].to_numpy()
    if not is_dep[own].all():
        raise ValueError("queries must be departures")
    n = len(own)
    X, M = np.zeros((n, k, len(FIELDS)), np.float32), np.zeros((n, k), bool)
    for j, offset in enumerate([o for o in range(-k//2, k//2+1) if o]):
        nb = own + offset
        ok = (nb >= 0) & (nb < len(raw))
        nb = np.clip(nb, 0, len(raw)-1)
        ok &= (airport[nb] == airport[own]) & pd.notna(airport[own])
        dt = (t[nb]-t[own])/60
        ok &= np.abs(dt) <= window_min
        M[:, j] = ok
        X[:, j, 0] = np.clip(dt, -window_min, window_min)/window_min
        X[:, j, 1], X[:, j, 2] = is_dep[nb], is_arr[nb]
        X[:, j, 3], X[:, j, 4] = nm_taxi[nb], has_aobt[nb]
        X[:, j, 5], X[:, j, 6] = taxi_in[nb], nm_arr[nb]
        X[:, j, 7], X[:, j, 8] = has_arvt[nb], copied[nb]
        X[:, j, 9], X[:, j, 10] = stand[nb] == stand[own], rwy[nb] == rwy[own]
        X[:, j, 11] = (oper[nb] == oper[own]) & (oper[own] != "None") & (oper[own] != "nan")
        X[:, j, 12] = sdel[nb]
        fd = (fid[nb]-fid[own])/1000
        okf = np.isfinite(fd)
        X[:, j, 13], X[:, j, 14] = np.where(okf, np.clip(fd, -10, 10)/10, 0.), okf
        X[:, j, 15] = typ[nb] == typ[own]
        X[:, j][~ok] = 0.
    if not np.isfinite(X).all():
        raise ValueError("nonfinite sequence field")
    return X, M
