"""Stand positions learned from arriving aircraft (research_730 pilot 3, PREREG_arrival_stands.md).

An arriving aircraft ends its roll-in at its assigned stand. For each arrival, the longest stationary episode (ground, below 2 kt, at least
60 s, starting after landing) of the matched track within 45 minutes of landing is its parked position. Per airport and normalised stand key a stand
is learned when at least MIN_N parked positions exist and all lie within MAX_SPREAD_M of their median; the table maps key -> [one-vertex array], the
shape that research_290.stand_detector.lookup expects. No departure label of any kind is read.
"""
import numpy as np
import pandas as pd

from research_730_adsb_learner import features_adsb as F

MIN_N, MAX_SPREAD_M, REST_KT, MIN_REST_S, MAX_GAP_S = 3, 30.0, 2.0, 60.0, 120.0
WINDOW_BEFORE, WINDOW_AFTER = 60.0, 2700.0


def parked_position(ts, lat, lon, gs, ground, landing):
    """Median (lat, lon) of the longest stationary episode that starts after landing, or None."""
    still = ground & (gs < REST_KT) & (ts >= landing)
    best, start = None, None
    for i in range(len(ts) + 1):
        inside = i < len(ts) and still[i] and (start is None or ts[i] - ts[i - 1] <= MAX_GAP_S)
        if inside and start is None:
            start = i
        if not inside and start is not None:
            if ts[i - 1] - ts[start] >= MIN_REST_S and (best is None or ts[i - 1] - ts[start] > best[0]):
                best = (ts[i - 1] - ts[start], start, i)
            start = i if (i < len(ts) and still[i]) else None
    if best is None:
        return None
    _, a, b = best
    return float(np.median(lat[a:b])), float(np.median(lon[a:b]))


def parked_positions(arrivals, tracks):
    """Frame (airport, stand, lat, lon) with one row per arrival that has a matched track and a parked position.
    arrivals: ADES_mvt, CALLSIGN_flt, MVT_TIME_UTC_mvt (landing), STAND_mvt; tracks as in features_adsb.compute."""
    t = tracks.dropna(subset=["callsign", "ts", "lat", "lon"]).copy()
    t["callsign"] = t.callsign.astype(str).str.strip().str.upper()
    t = t.sort_values(["airport", "callsign", "ts"], kind="stable").reset_index(drop=True)
    key = (t.airport.astype(str) + "|" + t.callsign).to_numpy()
    first = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]
    where = {key[a]: (a, b) for a, b in zip(first, np.r_[first[1:], len(key)])}
    ts, lat, lon, gs = t.ts.to_numpy(float), t.lat.to_numpy(float), t.lon.to_numpy(float), t.gs.to_numpy(float)
    ground, hexa = t.on_ground.to_numpy(bool), t.hex.astype(str).to_numpy()
    d = arrivals.reset_index(drop=True)
    land = F.seconds(d.MVT_TIME_UTC_mvt)
    keys = (d.ADES_mvt.astype(str) + "|" + d.CALLSIGN_flt.astype(str).str.strip().str.upper()).to_numpy()
    rows = []
    for i in range(len(d)):
        ab = where.get(keys[i])
        if ab is None or not np.isfinite(land[i]) or not isinstance(d.STAND_mvt.iat[i], str):
            continue
        a, b = ab
        lo = a + np.searchsorted(ts[a:b], land[i] - WINDOW_BEFORE, side="left")
        hi = a + np.searchsorted(ts[a:b], land[i] + WINDOW_AFTER, side="right")
        if hi <= lo:
            continue
        idx = np.arange(lo, hi)
        if len(set(hexa[idx])) > 1:
            vals, counts = np.unique(hexa[idx], return_counts=True)
            idx = idx[hexa[idx] == vals[counts.argmax()]]
        pos = parked_position(ts[idx], lat[idx], lon[idx], gs[idx], ground[idx], land[i])
        if pos is not None:
            rows.append((str(d.ADES_mvt.iat[i]), d.STAND_mvt.iat[i], pos[0], pos[1]))
    return pd.DataFrame(rows, columns=["airport", "stand", "lat", "lon"])


def _metres(lat, lon, lat0, lon0):
    return np.sqrt(((lat - lat0) * 110540.0) ** 2 + ((lon - lon0) * 111320.0 * np.cos(np.radians(lat0))) ** 2)


def learn_table(points, key_function, min_n=MIN_N, max_spread_m=MAX_SPREAD_M):
    """{airport: {stand key: [array([[lat, lon]])]}}; a stand needs min_n positions, all within max_spread_m of their median."""
    out = {}
    p = points.assign(key=[key_function(s) for s in points.stand])
    for (airport, k), g in p.groupby(["airport", "key"]):
        if not k or len(g) < min_n:
            continue
        la, lo = float(g.lat.median()), float(g.lon.median())
        if _metres(g.lat.to_numpy(), g.lon.to_numpy(), la, lo).max() > max_spread_m:
            continue
        out.setdefault(airport, {})[k] = [np.array([[la, lo]])]
    return out
