"""Assigned-stand departure detector for ADS-B ground tracks. Rules: research_290/PREREG_stand_detector.md.

Why a new detector. The old one (adsb-lol/offblock_from_tracks.py) takes the FIRST point at MOVE_KT or faster in a
visit and calls the point before it the off-block, provided the rest stretch lies near ANY stand or apron vertex.
When reporting starts after the push-back, the first rest stretch it sees is a later stop on an apron taxilane, and
apron vertices are everywhere on the apron, so that stop passes. Codex measured the result on 13 December 2025 at
Munich: all 55 detections late, median +634.9 s, while 36 of the 55 had a ground point within 60 s of the block.

This detector anchors on the ASSIGNED stand (the challenge's STAND_mvt, mapped to OpenStreetMap parking positions)
and never uses a stop elsewhere. It abstains rather than guess: an unmapped or ambiguous stand, no stationary point
at the stand, no observed departure from it before take-off, or an exit whose timing is too uncertain.

Pure functions only, no I/O beyond reading the repository's OSM geometry. Times are seconds (float epoch).
"""
import json
from pathlib import Path

import numpy as np

import features_geometry as FG

GEOMETRY = Path(__file__).resolve().parents[1] / "geometry"
R_STAND_M = 80.0      # a point is AT the stand within this distance of any vertex of the assigned stand
REST_KT = 2.0         # stationary
MOVE_KT = 4.0         # moving under its own power or under tow
AMBIGUOUS_M = 100.0   # a stand key whose candidate elements lie further apart than this is ambiguous
MAX_GAP_S = 120.0     # abstain when the departure moment is bracketed only this loosely
R_PARK_M = 30.0       # the parked position lies this close to a stand vertex (chosen on the development day)
MAX_ONSET_SPREAD_S = 600.0  # parked-to-motion transitions further apart than this mean a tow or reposition
TAKEOFF_MARGIN_S = 60.0
# v2 only (PREREG addendum B): a transition this soon after motion this fast is docking after the arrival
FAST_KT = 30.0
ARRIVAL_WINDOW_S = 600.0


def _vertices(element):
    geometry = element.get("geometry")
    if geometry:
        return np.array([(p["lat"], p["lon"]) for p in geometry if p], dtype=float)
    if "lat" in element and "lon" in element:
        return np.array([(element["lat"], element["lon"])], dtype=float)
    return np.empty((0, 2))


def stand_table(icao):
    """Normalised stand key -> list of vertex arrays, one per OSM parking_position carrying that key."""
    path = GEOMETRY / f"{icao.lower()}_osm.json"
    table = {}
    for element in json.loads(path.read_text()).get("elements", []):
        tags = element.get("tags", {})
        if tags.get("aeroway") != "parking_position" or not tags.get("ref"):
            continue
        points = _vertices(element)
        if not len(points):
            continue
        for key in FG.stand_kandidaten(tags["ref"]):
            table.setdefault(key, []).append(points)
    return table


def _metres(lat, lon, points):
    """Distance in metres from (lat, lon) arrays to the nearest of `points`, equirectangular, as the old detector."""
    lat = np.atleast_1d(np.asarray(lat, float))[:, None]
    lon = np.atleast_1d(np.asarray(lon, float))[:, None]
    dy = (points[None, :, 0] - lat) * 110540.0
    dx = (points[None, :, 1] - lon) * 111320.0 * np.cos(np.radians(lat))
    return np.sqrt((dx * dx + dy * dy).min(axis=1))


def _spread_m(points):
    """Largest pairwise distance in metres among `points` (n, 2)."""
    lat, lon = points[:, 0], points[:, 1]
    dy = (lat[:, None] - lat[None, :]) * 110540.0
    dx = (lon[:, None] - lon[None, :]) * 111320.0 * np.cos(np.radians(lat[:, None]))
    return float(np.sqrt(dx * dx + dy * dy).max())


def lookup(table, stand):
    """(vertices, status) for a challenge stand id; status is ok, unmapped or ambiguous.

    features_geometry.lade keeps the FIRST element when several OSM stands normalise to one key and so hides the
    ambiguity; here every candidate is kept, and candidates further apart than AMBIGUOUS_M make the stand ambiguous.
    """
    key = FG.stand_schluessel(stand or "")
    elements, status = table.get(key), "ok"
    if not elements and key:
        # The challenge writes a stand as a bare number (111) where OSM splits it into lettered sub-positions
        # (111A, 111B), as on stands that take one wide-body or two narrow-bodies. Measured on December 2025 at
        # Munich: 4,982 of 12,035 departures sat on such ids. Their union is the stand; the spread check still applies.
        elements = [e for k, v in table.items() if len(k) == len(key) + 1 and k.startswith(key) and k[-1].isalpha()
                    for e in v]
        status = "ok_subdivided"
    if not elements:
        return None, "unmapped"
    centres = np.array([e.mean(axis=0) for e in elements])
    if len(centres) > 1 and _spread_m(centres) > AMBIGUOUS_M:
        return None, "ambiguous"
    return np.vstack(elements), status


def detect(ts, lat, lon, gs, vertices, t_takeoff):
    """Departure from the assigned stand, or an abstention with its reason.

    Uses only points before take-off. The LAST stationary point at the stand is t_rest; the first later point that
    is moving or outside the stand radius is t_leave. The push-back began somewhere in (t_rest, t_leave]; if that
    interval exceeds MAX_GAP_S the detector abstains. Returns both brackets so the estimate definition chosen on the
    development day can be applied without re-running this function.
    """
    ts, lat, lon, gs = (np.asarray(a, float) for a in (ts, lat, lon, gs))
    keep = np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lon) & (ts <= t_takeoff + TAKEOFF_MARGIN_S)
    order = np.argsort(ts[keep], kind="stable")
    ts, lat, lon, gs = (a[keep][order] for a in (ts, lat, lon, gs))
    if not len(ts):
        return {"status": "no_points"}
    at_stand = _metres(lat, lon, vertices) <= R_STAND_M
    resting = at_stand & np.isfinite(gs) & (gs < REST_KT)
    if not resting.any():
        return {"status": "no_occupancy", "points_at_stand": int(at_stand.sum())}
    i_rest = int(np.flatnonzero(resting)[-1])
    after = np.arange(len(ts)) > i_rest
    leaving = after & ((~at_stand) | (np.isfinite(gs) & (gs >= MOVE_KT)))
    if not leaving.any():
        return {"status": "no_exit", "t_rest": float(ts[i_rest])}
    i_leave = int(np.flatnonzero(leaving)[0])
    gap = float(ts[i_leave] - ts[i_rest])
    out = {"t_rest": float(ts[i_rest]), "t_leave": float(ts[i_leave]), "gap_s": gap,
           "points_at_stand": int(at_stand.sum()), "rest_points": int(resting.sum())}
    out["status"] = "ok" if gap <= MAX_GAP_S else "gap_too_large"
    return out


def detect_onset(ts, lat, lon, gs, vertices, t_takeoff, r_park_m):
    """First motion out of the PARKED position at the assigned stand: the push-back, or the power-out taxi start.

    Why not `detect`. On the development day (13 December 2025, Munich) every estimate of `detect` ran a median
    +257 to +285 s late. The tracks show why: push-back runs at 2 to 4 kt, below MOVE_KT, and ends in a stop 30 to
    70 m from the stand, still inside R_STAND_M; engine start and tug disconnect happen there, and taxi follows
    minutes later. The LAST rest at the stand is that post-push-back stop, and leaving the radius is the taxi start.

    Rule. Take the final run of consecutive points inside R_STAND_M before take-off, plus the first point after it.
    Scan it in time order for the first point j whose predecessor is stationary (below REST_KT) AND within r_park_m
    of the stand, and which itself is in motion (REST_KT or faster) or outside R_STAND_M. The parked position lies
    close to the stand vertex (development median 15 m) and the post-push-back stop does not (median 41 m), so
    requiring the parked distance keeps a later stop from standing in for the parked state. Abstain when no such j
    exists, or when the bracket (t_before, t_after] is wider than MAX_GAP_S.
    """
    ts, lat, lon, gs = (np.asarray(a, float) for a in (ts, lat, lon, gs))
    keep = np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lon) & (ts <= t_takeoff + TAKEOFF_MARGIN_S)
    order = np.argsort(ts[keep], kind="stable")
    ts, lat, lon, gs = (a[keep][order] for a in (ts, lat, lon, gs))
    if not len(ts):
        return {"status": "no_points"}
    d = _metres(lat, lon, vertices)
    at = d <= R_STAND_M
    if not at.any():
        return {"status": "no_occupancy", "points_at_stand": 0}
    end = int(np.flatnonzero(at)[-1])
    start = end
    while start > 0 and at[start - 1]:
        start -= 1
    stop = min(end + 1, len(ts) - 1)          # include the exit point when there is one
    still = np.isfinite(gs) & (gs < REST_KT)
    moving = (np.isfinite(gs) & (gs >= REST_KT)) | ~at
    onsets = [j for j in range(start + 1, stop + 1) if still[j - 1] and d[j - 1] <= r_park_m and moving[j]]
    if not onsets:
        return {"status": "no_parked_onset", "run_points": int(end - start + 1)}
    mid = [(ts[j - 1] + ts[j]) / 2 for j in onsets]
    spread = float(mid[-1] - mid[0])
    j = onsets[0]
    gap = float(ts[j] - ts[j - 1])
    out = {"t_before": float(ts[j - 1]), "t_after": float(ts[j]), "gap_s": gap, "onsets": len(onsets),
           "onset_spread_s": spread, "parked_distance_m": float(d[j - 1]), "run_points": int(end - start + 1)}
    # Several parked-to-motion transitions far apart mean a tow or a reposition happened inside the stand area,
    # and the first one is then not the push-back (development day: two such rows ran 3,410 and 3,206 s early,
    # spreads 3,399 and 3,514 s, while every ordinary row spread at most 363 s). Abstain rather than guess.
    if spread > MAX_ONSET_SPREAD_S:
        out["status"] = "ambiguous_onsets"
    else:
        out["status"] = "ok" if gap <= MAX_GAP_S else "gap_too_large"
    return out


def detect_onset_v2(ts, lat, lon, gs, vertices, t_takeoff, r_park_m):
    """detect_onset unchanged, plus one rule added after its validation failed (PREREG addendum B).

    Abstain (status "arrival_transition") when any observation in [t_before - ARRIVAL_WINDOW_S, t_after) is faster
    than FAST_KT. A push-back follows a parked turnaround, and nothing parked was at landing or fast-taxi speed ten
    minutes earlier, so such a transition is a stop-and-creep step while docking. On 3 December 2025 at stand 215
    that step made v1 3,179 s early and failed its validation; on the four used dates the rule removes that one
    estimate and none within 60 s of the recorded block. Observations are filtered exactly as in detect_onset.
    """
    out = detect_onset(ts, lat, lon, gs, vertices, t_takeoff, r_park_m)
    if out.get("status") != "ok":
        return out
    ts, lat, lon, gs = (np.asarray(a, float) for a in (ts, lat, lon, gs))
    window = (np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(gs)
              & (ts >= out["t_before"] - ARRIVAL_WINDOW_S) & (ts < out["t_after"]))
    if window.any() and gs[window].max() > FAST_KT:
        out["status"] = "arrival_transition"
    return out
