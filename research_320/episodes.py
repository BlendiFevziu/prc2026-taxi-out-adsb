"""Ordered stop/move episodes of a partial ground track. Label-free; never asserts an off-block time.

The points are exactly those behind v56's 23 trajectory summaries (research_310/partial_track_features.summarise,
commit 9884150): ground points with finite coordinates in [takeoff - 7200 s, takeoff], the same ground mask, sorted by
time. A STOP is a maximal run of consecutive points that all lie within STOP_RADIUS_M of the run's first point, lasts
at least MIN_STOP_S and has a median finite speed below STILL_KT (or no finite speed); everything else is movement.

What the summaries cannot say is how the stops are arranged between the stand and the runway: they give the still
fraction and the last still point near the stand, not the order, length and place of the individual stops. Read
backwards from take-off, the stops say that directly, for example a short push-back stop and then a long hold near
the runway, against a long wait at the stand and then a quick taxi.
"""
import numpy as np
import pandas as pd

LOOKBACK_S = 7200.0            # v56's window
STOP_RADIUS_M = 30.0
MIN_STOP_S = 30.0
STILL_KT = 2.0                 # v56's still threshold
NEAR_STAND_M = 80.0            # v56's near-stand radius
EPISODE_FEATURES = ("stop_count", "last_stop_s", "last_stop_end_age_s", "last_stop_stand_m",
                    "prev_stop_s", "prev_stop_end_age_s", "prev_stop_stand_m",
                    "longest_stop_s", "longest_stop_end_age_s", "stand_stop_s", "stopped_after_stand_s")


def _metres(lat, lon, points):
    """Equirectangular distance to the nearest of points (n, 2), as research_290/stand_detector._metres."""
    lat = np.atleast_1d(np.asarray(lat, float))[:, None]
    lon = np.atleast_1d(np.asarray(lon, float))[:, None]
    dy = (points[None, :, 0] - lat) * 110540.0
    dx = (points[None, :, 1] - lon) * 111320.0 * np.cos(np.radians(lat))
    return np.sqrt((dx * dx + dy * dy).min(axis=1))


def ground_mask(values):
    """research_310/partial_track_audit.ground_mask: True/1 is ground, missing is unknown, strings are refused."""
    values = pd.Series(values)
    if not values.dropna().isin([True, False, 0, 1]).all():
        raise ValueError("unexpected on_ground representation")
    return values.eq(True).fillna(False).to_numpy(dtype=bool), values.isna().to_numpy()   # verbatim, 9884150


def window_ground(points, takeoff):
    p = points[["ts", "lat", "lon", "gs", "on_ground"]].copy()
    p = p[np.isfinite(p.ts) & p.ts.between(takeoff - LOOKBACK_S, takeoff)].sort_values("ts", kind="stable")
    if not len(p):
        return p
    ground, _ = ground_mask(p.on_ground)
    return p.loc[ground & np.isfinite(p.lat) & np.isfinite(p.lon)]


def stops(t, lat, lon, speed):
    """[(start_index, end_index)] of the stops, in time order."""
    out, anchor, n = [], 0, len(t)
    for i in range(1, n + 1):
        if i < n and _metres(lat[i], lon[i], np.array([[lat[anchor], lon[anchor]]]))[0] <= STOP_RADIUS_M:
            continue
        end = i - 1
        s = speed[anchor:end + 1]
        s = s[np.isfinite(s)]
        if end > anchor and t[end] - t[anchor] >= MIN_STOP_S and (not len(s) or np.median(s) < STILL_KT):
            out.append((anchor, end))
        anchor = i
    return out


def episodes(points, stand, takeoff):
    out = {k: np.nan for k in EPISODE_FEATURES}
    g = window_ground(points, takeoff)
    if not len(g):
        return out
    t, lat, lon = g.ts.to_numpy(float), g.lat.to_numpy(float), g.lon.to_numpy(float)
    speed = pd.to_numeric(g.gs, errors="coerce").to_numpy(float)
    found = stops(t, lat, lon, speed)
    out["stop_count"] = float(len(found))
    if not found:
        return out
    dur = np.array([t[e] - t[s] for s, e in found])
    end_age = np.array([takeoff - t[e] for s, e in found])
    where = [(np.median(lat[s:e + 1]), np.median(lon[s:e + 1])) for s, e in found]
    dist = np.array([_metres(a, b, stand)[0] for a, b in where]) if stand is not None else np.full(len(found), np.nan)
    out.update(last_stop_s=dur[-1], last_stop_end_age_s=end_age[-1], last_stop_stand_m=dist[-1])
    if len(found) > 1:
        out.update(prev_stop_s=dur[-2], prev_stop_end_age_s=end_age[-2], prev_stop_stand_m=dist[-2])
    k = int(np.argmax(dur))
    out.update(longest_stop_s=dur[k], longest_stop_end_age_s=end_age[k])
    if stand is not None:
        near = np.flatnonzero(dist <= NEAR_STAND_M)
        if len(near):
            out["stand_stop_s"] = dur[near[-1]]
        beyond = np.flatnonzero(_metres(lat, lon, stand) > NEAR_STAND_M)
        if len(beyond):
            out["stopped_after_stand_s"] = float(sum(d for (s, e), d in zip(found, dur) if t[s] >= t[beyond[0]]))
    return {k: float(v) for k, v in out.items()}
