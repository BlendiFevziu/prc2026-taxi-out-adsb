"""Label-free per-departure ADS-B ground-track features (research_730 pilot, PREREG_pilot.md).

A departure is matched to the track points of the same callsign at the same airport in the window (take-off - 90 min,
take-off + 2 min). Points are split into contiguous segments at gaps above 20 minutes; the last segment is used. If
several aircraft (hex) share the callsign in the window, the one with most points is used. Speeds are in knots
(readsb), a stop is a ground point below 2 kt. No label is read.

  ad_matched, ad_pts, ad_first_rel_s, ad_span_s, ad_gs_first, ad_gs_max, ad_last_stop_rel_s, ad_move_rel_s,
  ad_dist_m, ad_gap_max_s, ad_stops, ad_airborne_first
"""
import numpy as np
import pandas as pd

MERKMALE = ["ad_matched", "ad_pts", "ad_first_rel_s", "ad_span_s", "ad_gs_first", "ad_gs_max", "ad_last_stop_rel_s",
            "ad_move_rel_s", "ad_dist_m", "ad_gap_max_s", "ad_stops", "ad_airborne_first"]
WINDOW_BEFORE, WINDOW_AFTER, SPLIT_GAP, STOP_KT, STOP_MIN_S = 5400.0, 120.0, 1200.0, 2.0, 30.0
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
# Stand readout (PREREG_stand.md): research_290's detector run on the matched segment against the assigned stand.
STAND_MERKMALE = ["sd_status", "sd_onset_rel_s", "sd_gap_s", "sd_parked_m", "sd_run_pts", "sd_first_dist_m"]
STATUS_CODE = {"ok": 1, "gap_too_large": 2, "ambiguous_onsets": 3, "arrival_transition": 4, "no_parked_onset": 5,
               "no_occupancy": 6, "no_points": 7, "unmapped": 8, "ambiguous": 9}


def _stand_module():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from research_290 import stand_detector
    return stand_detector


def stand_features(SD, table, stand, ts, lat, lon, gs, takeoff):
    """The six sd_ columns for one matched segment; the detector abstains rather than guesses (status codes above)."""
    out = dict.fromkeys(STAND_MERKMALE, np.nan)
    stand = stand if isinstance(stand, str) else ""                  # a missing stand (NaN) is "no stand", code 8
    vertices, status = SD.lookup(table, stand) if table is not None else (None, "unmapped")
    if vertices is None:
        out["sd_status"] = float(STATUS_CODE[status])
        return out
    result = SD.detect_onset_v2(ts, lat, lon, gs, vertices, takeoff, SD.R_PARK_M)
    out["sd_status"] = float(STATUS_CODE.get(result.get("status"), 0))
    if "t_before" in result:
        out["sd_onset_rel_s"] = takeoff - (result["t_before"] + result["t_after"]) / 2
        out["sd_gap_s"], out["sd_parked_m"] = result["gap_s"], result["parked_distance_m"]
    out["sd_run_pts"] = float(result.get("run_points", result.get("points_at_stand", np.nan)))
    out["sd_first_dist_m"] = float(SD._metres(lat[0], lon[0], vertices)[0])
    return out


def seconds(values):
    t = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    return ((t - EPOCH) / pd.Timedelta(seconds=1)).to_numpy(float)


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def segment_features(ts, lat, lon, gs, ground, takeoff):
    """Features of one matched segment (arrays sorted by ts); NaN-filled dict entries where undefined."""
    out = dict.fromkeys(MERKMALE, np.nan)
    out["ad_matched"], out["ad_pts"] = 1.0, float(len(ts))
    out["ad_first_rel_s"], out["ad_span_s"] = takeoff - ts[0], ts[-1] - ts[0]
    out["ad_gs_first"], out["ad_gs_max"] = float(gs[0]), float(np.nanmax(gs)) if np.isfinite(gs).any() else np.nan
    out["ad_airborne_first"] = float(not ground[0])
    out["ad_gap_max_s"] = float(np.max(np.diff(ts))) if len(ts) > 1 else 0.0
    stop = ground & (gs < STOP_KT)
    if stop.any():
        last_stop = int(np.flatnonzero(stop)[-1])
        out["ad_last_stop_rel_s"] = takeoff - ts[last_stop]
        after = np.flatnonzero((np.arange(len(ts)) > last_stop) & (gs >= STOP_KT))
        if len(after):
            out["ad_move_rel_s"] = takeoff - ts[after[0]]
    both = ground[1:] & ground[:-1]
    out["ad_dist_m"] = float(_haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])[both].sum()) if len(ts) > 1 else 0.0
    runs, start = 0, None                                        # stop episodes lasting at least STOP_MIN_S
    for i, s in enumerate(stop):
        if s and start is None:
            start = i
        if (not s or i == len(stop) - 1) and start is not None:
            end = i if s else i - 1
            runs += int(ts[end] - ts[start] >= STOP_MIN_S)
            start = None
    out["ad_stops"] = float(runs)
    return out


def compute(deps, tracks, stands=False, stand_tables=None):
    """deps: frame with MVT_ID_mvt, ADEP_mvt, CALLSIGN_flt, MVT_TIME_UTC_mvt (and STAND_mvt when stands=True). tracks: frame
    with hex, callsign, ts, lat, lon, on_ground, gs, airport. Returns the feature frame indexed by MVT_ID_mvt (unmatched
    rows: ad_matched 0, rest NaN). stands=True adds the STAND_MERKMALE columns (sd_status 0 for unmatched rows).
    stand_tables: optional {airport: stand table} used instead of the OSM table (airports missing from it count as unmapped)."""
    t = tracks.dropna(subset=["callsign", "ts", "lat", "lon"]).copy()
    t["callsign"] = t.callsign.astype(str).str.strip().str.upper()
    t = t.sort_values(["airport", "callsign", "ts"], kind="stable").reset_index(drop=True)
    key = (t.airport.astype(str) + "|" + t.callsign).to_numpy()
    first = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]
    last = np.r_[first[1:], len(key)]
    where = {key[a]: (a, b) for a, b in zip(first, last)}
    ts, lat, lon = t.ts.to_numpy(float), t.lat.to_numpy(float), t.lon.to_numpy(float)
    gs, ground, hexa = t.gs.to_numpy(float), t.on_ground.to_numpy(bool), t.hex.astype(str).to_numpy()
    cols = ["MVT_ID_mvt", "ADEP_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt"] + (["STAND_mvt"] if stands else [])
    d = deps[cols].reset_index(drop=True)
    SD, tables = (_stand_module(), dict(stand_tables) if stand_tables is not None else {}) if stands else (None, None)
    take = seconds(d.MVT_TIME_UTC_mvt)
    keys = (d.ADEP_mvt.astype(str) + "|" + d.CALLSIGN_flt.astype(str).str.strip().str.upper()).to_numpy()
    rows = []
    for i in range(len(d)):
        row = dict.fromkeys(MERKMALE + (STAND_MERKMALE if stands else []), np.nan)
        row["ad_matched"] = 0.0
        if stands:
            row["sd_status"] = 0.0
        ab = where.get(keys[i])
        if ab is not None and np.isfinite(take[i]):
            a, b = ab
            lo = a + np.searchsorted(ts[a:b], take[i] - WINDOW_BEFORE, side="left")
            hi = a + np.searchsorted(ts[a:b], take[i] + WINDOW_AFTER, side="right")
            if hi > lo:
                idx = np.arange(lo, hi)
                if len(set(hexa[idx])) > 1:
                    vals, counts = np.unique(hexa[idx], return_counts=True)
                    idx = idx[hexa[idx] == vals[counts.argmax()]]
                cut = np.flatnonzero(np.diff(ts[idx]) > SPLIT_GAP)
                if len(cut):
                    idx = idx[cut[-1] + 1:]
                row = segment_features(ts[idx], lat[idx], lon[idx], gs[idx], ground[idx], take[i])
                if stands:
                    airport = str(d.ADEP_mvt.iat[i])
                    if airport not in tables and stand_tables is not None:
                        tables[airport] = None
                    if airport not in tables:
                        try:
                            tables[airport] = SD.stand_table(airport)
                        except FileNotFoundError:
                            tables[airport] = None
                    row.update(stand_features(SD, tables[airport], d.STAND_mvt.iat[i], ts[idx], lat[idx], lon[idx],
                                              gs[idx], take[i]))
        rows.append(row)
    out = pd.DataFrame(rows, columns=MERKMALE + (STAND_MERKMALE if stands else []))
    out.index = d.MVT_ID_mvt.to_numpy()
    return out
