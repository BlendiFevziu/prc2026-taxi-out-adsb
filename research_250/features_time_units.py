"""Correct variants of the six active features affected by datetime storage units.

The legacy v44 columns are retained for paired ablations. New columns are prefixed
tu_ so a saved v44 model can never silently receive a different feature definition.
"""
import numpy as np
import pandas as pd

RAW_COLUMNS = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt"]
DEP_COLUMNS = ["ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt"]
CORRECTED_COLUMNS = ["auslast", "um_dichte_stand_3h", "fid_zeit_vor", "fid_zeit_nach",
                     "fid_rate_vor", "fid_rate_nach"]
MERKMALE = ["tu_" + c for c in CORRECTED_COLUMNS]

def normalize_datetime_columns(frame):
    out = frame.copy()
    for col in out:
        if pd.api.types.is_datetime64_any_dtype(out[col].dtype):
            out[col] = out[col].dt.as_unit("ns")
    return out

def _epoch_seconds(values):
    stamps = pd.to_datetime(values, utc=True, errors="coerce").dt.as_unit("ns")
    seconds = stamps.astype("int64").to_numpy() / 1e9
    seconds[stamps.isna().to_numpy()] = np.nan
    return seconds

def _counts(source_keys, source_times, query_keys, query_times, radius):
    source = pd.DataFrame({"key": np.asarray(source_keys), "t": source_times})
    lookup = {k: np.sort(g.t.to_numpy()) for k, g in
              source[source.t.notna()].groupby("key", sort=False, observed=True)}
    query = pd.DataFrame({"key": np.asarray(query_keys), "t": query_times})
    values = np.zeros(len(query), dtype=np.float64)
    values[~np.isfinite(query_times)] = np.nan
    for k, indices in query.groupby("key", sort=False, observed=True).groups.items():
        indices = np.asarray(indices)
        t = query_times[indices]
        reference = lookup.get(k, np.array([], dtype=float))
        values[indices] = (np.searchsorted(reference, t + radius, side="right")
                           - np.searchsorted(reference, t - radius, side="left"))
        values[indices[~np.isfinite(t)]] = np.nan
    return values

def corrected_values(raw, dep):
    import features_flight_sequence as F
    d = normalize_datetime_columns(dep[DEP_COLUMNS])
    t = _epoch_seconds(d.MVT_TIME_UTC_mvt)
    airport = d.ADEP_mvt.astype(str).to_numpy()
    result = pd.DataFrame(index=dep.index)
    result["auslast"] = _counts(airport, t, airport, t, 1800) - 1

    phase = raw.PHASE_mvt.astype(str).str.upper()
    movement = phase.str.startswith("ARR") | phase.str.startswith("DEP")
    r = raw.loc[movement, RAW_COLUMNS]
    arrival = phase.loc[movement].str.startswith("ARR").to_numpy()
    reporting_airport = np.where(arrival, r.ADES_mvt.astype(str), r.ADEP_mvt.astype(str))
    source_keys = pd.Series(reporting_airport).astype(str).to_numpy() + "|" + r.STAND_mvt.astype(str).to_numpy()
    query_keys = airport + "|" + d.STAND_mvt.astype(str).to_numpy()
    result["um_dichte_stand_3h"] = _counts(
        source_keys, _epoch_seconds(r.MVT_TIME_UTC_mvt),
        query_keys, t, 5400)
    fixed = F.baue(d)
    assert fixed.index.equals(dep.index)
    for col in CORRECTED_COLUMNS[2:]:
        result[col] = fixed[col].to_numpy()
    return result

def _self_check():
    base = pd.Timestamp("2025-01-01", tz="UTC")
    dep = pd.DataFrame({"ADEP_mvt":["EDDF"]*5, "STAND_mvt":["A1"]*5,
                        "MVT_TIME_UTC_mvt": [base + pd.Timedelta(minutes=m) for m in [0,10,30,31,200]],
                        "FLIGHT_ID_mvt":[10.,20.,30.,40.,50.]})
    raw = dep.assign(PHASE_mvt="DEP", ADES_mvt="EGLL")[RAW_COLUMNS]
    fixed = corrected_values(raw, dep)
    assert fixed.auslast.to_list() == [2,3,3,2,0], fixed.auslast.to_list()
    assert fixed.um_dichte_stand_3h.to_list() == [4,4,4,4,1]
    dep_us, raw_us = dep.copy(), raw.copy()
    dep_us["MVT_TIME_UTC_mvt"] = dep_us.MVT_TIME_UTC_mvt.dt.as_unit("us")
    raw_us["MVT_TIME_UTC_mvt"] = raw_us.MVT_TIME_UTC_mvt.dt.as_unit("us")
    pd.testing.assert_frame_equal(fixed, corrected_values(raw_us, dep_us))
    shifted = dep.copy()
    shifted["FLIGHT_ID_mvt"] += 10000
    pd.testing.assert_frame_equal(fixed, corrected_values(raw, shifted))
    assert normalize_datetime_columns(dep_us).MVT_TIME_UTC_mvt.dtype.unit == "ns"
    print("Timestamp feature checks passed: actual clock windows, ns/us invariance, absolute-ID invariance.")

if __name__ == "__main__":
    _self_check()
