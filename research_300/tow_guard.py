"""Tow and false-onset guard for the stand detector (research only; rules in research_300/PREREG_towguard.md).

Applied after stand_detector.detect_onset_v2 and v54's arrival-and-stay guard (arm_test.guarded) returned "ok".
It withdraws the estimate in two situations that the development tracks showed (research_300/tow_guard_dev.py,
used dates only):
  false_onset   after the onset the aircraft is seen resting again at its own parked spot (within PARKED_SPOT_M of
                the point at t_before, below REST_KT) for REST_AT_SPOT_S or longer: the "onset" was a speed blip and
                the push-back came later (QY1239 on 16 January 2025 rested there 46 more minutes)
  long_stop     after the aircraft first leaves R_STAND_M it stands still somewhere for LONG_STOP_S or longer before
                take-off (ground points within STOP_RADIUS_M of the stop's first point): a tow or reposition, after
                which the airport's block time belongs to the second position (UA021 on 16 July 2025, 51 minutes)
Only points after t_after and up to take-off are used, exactly the points the detector itself may use.
"""
import numpy as np

from research_290 import stand_detector as SD

PARKED_SPOT_M = 20.0
REST_AT_SPOT_S = 300.0
STOP_RADIUS_M = 30.0
LONG_STOP_S = 1800.0


def post_onset_features(ts, lat, lon, gs, ground, vertices, t_before, t_after, t_takeoff):
    """(rest_at_spot_s, longest_stop_s) after the detected onset; see the module docstring."""
    ts, lat, lon, gs = (np.asarray(a, float) for a in (ts, lat, lon, gs))
    ground = np.asarray(ground, bool)
    order = np.argsort(ts, kind="stable")
    ts, lat, lon, gs, ground = (a[order] for a in (ts, lat, lon, gs, ground))
    i_park = np.flatnonzero(ts == t_before)
    if not len(i_park):
        return 0.0, 0.0
    parked = np.array([[lat[i_park[-1]], lon[i_park[-1]]]])
    keep = (ts > t_after) & (ts <= t_takeoff) & np.isfinite(lat) & np.isfinite(lon)
    ts, lat, lon, gs, ground = (a[keep] for a in (ts, lat, lon, gs, ground))
    if not len(ts):
        return 0.0, 0.0
    spot = (SD._metres(lat, lon, parked) <= PARKED_SPOT_M) & np.isfinite(gs) & (gs < SD.REST_KT)
    rest_at_spot = float(ts[spot].max() - ts[spot].min()) if spot.sum() > 1 else 0.0
    longest = 0.0
    out = np.flatnonzero(SD._metres(lat, lon, vertices) > SD.R_STAND_M)
    if len(out):
        anchor, first = None, None
        for i in np.flatnonzero(ground[out[0]:]) + out[0]:
            if anchor is None or SD._metres(lat[i], lon[i], np.array([[lat[anchor], lon[anchor]]]))[0] > STOP_RADIUS_M:
                anchor, first = i, ts[i]
            longest = max(longest, float(ts[i] - first))
    return rest_at_spot, longest


def check(ts, lat, lon, gs, ground, vertices, onset, t_takeoff):
    """The detector's result with status "false_onset" or "long_stop" when the guard withdraws it."""
    if onset.get("status") != "ok":
        return onset
    rest, longest = post_onset_features(ts, lat, lon, gs, ground, vertices, onset["t_before"], onset["t_after"], t_takeoff)
    out = dict(onset, rest_at_spot_s=rest, longest_stop_s=longest)
    if rest >= REST_AT_SPOT_S:
        out["status"] = "false_onset"
    elif longest >= LONG_STOP_S:
        out["status"] = "long_stop"
    return out
