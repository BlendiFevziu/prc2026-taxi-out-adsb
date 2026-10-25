"""Derive an off-block time per departure from ADS-B tracks, with an explicit usability gate.

What the first day measured (results/offblock_2025-07-13.json, adsb.lol prod-0 only):
  the first point near an airport is a whole taxi late in the median (+840 s), because at most airports this feed
  sees aircraft only once they are near the runway. EDDM is the extreme: 17,380 points, zero on the ground, median
  ground speed 151 kt. EHAM is the opposite: 74 % of its points are on the ground, median speed 21 kt, and where a
  departure matched, 48 % of off-block estimates landed within 60 s of the airport's own block time.

So the quantity is only available where the track starts at rest. This module therefore returns an estimate ONLY
when the visit shows the aircraft stationary before it starts moving, and says so per row:
  rest_points   at least three points below 5 kt before the first sustained movement
  t_offblock    the last stationary point before that movement, which is the push-back moment
  usable        rest_points satisfied and the stationary stretch covers at least 60 s
Rows without that evidence get no estimate rather than a bad one, which is what separates this from the naive
"first observation" rule.

Matching. Movement callsigns are the commercial number at some airports (KL1234) while ADS-B sends the ICAO
callsign (KLM1234). The mapping is learned from our own data: on main-lane rows the challenge carries both
FLIGHT_mvt and CALLSIGN_flt, so prefix pairs with an identical numeric suffix give 130 mappings (LH->DLH, KL->KLM,
DE->CFG, HV->TRA, OS->AUA and so on). A departure matches a visit when any candidate callsign agrees and the
visit's last point lies within 5 minutes of the given take-off time; no label is used.

Source: adsb.lol daily releases, ODbL 1.0, attribution required.
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT = Path(".")
REST_KT = 2.0      # standing at the stand
MOVE_KT = 4.0      # under tow or taxiing; a push-back runs at two to five knots, hence the hysteresis
MIN_REST_POINTS = 3
MIN_REST_SECONDS = 60.0
STAND_METRES = 80.0
PREFIX = re.compile(r"^([A-Z]+)")


def stand_vertices():
    """Latitude and longitude of every parking position and apron vertex, per airport, from the repo geometry.

    A post-push-back stop on a taxiway looks exactly like a stand in speed alone, which is why the first day's
    estimate ran 242 s late in the median. Requiring the standing stretch to sit at a stand is what separates them.
    """
    out = {}
    for path in sorted((PROJECT / "geometry").glob("*_osm.json")):
        data = json.loads(path.read_text())
        points = [(node["lat"], node["lon"])
                  for element in data.get("elements", [])
                  if element.get("tags", {}).get("aeroway") in ("parking_position", "apron")
                  for node in element.get("geometry", []) or []]
        if points:
            out[data["icao"].upper()] = np.asarray(points, dtype=float)
    return out


def near_stand(latitude, longitude, vertices):
    if vertices is None or not len(vertices) or not np.isfinite(latitude):
        return False
    dy = (vertices[:, 0] - latitude) * 110540.0
    dx = (vertices[:, 1] - longitude) * 111320.0 * np.cos(np.radians(latitude))
    return bool(np.min(dx * dx + dy * dy) <= STAND_METRES ** 2)


def callsign_mapping():
    """IATA-style movement prefix -> ICAO prefix, learned from rows that carry both identifiers."""
    frames = []
    for path in sorted(glob.glob(str(PROJECT / "data/training_2025-*.parquet"))):
        frames.append(pq.ParquetFile(path).read(columns=["FLIGHT_mvt", "CALLSIGN_flt", "PHASE_mvt"],
                                                use_threads=False).to_pandas())
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame.PHASE_mvt.eq("DEP") & frame.CALLSIGN_flt.notna()]
    movement = frame.FLIGHT_mvt.astype(str).str.upper().str.strip()
    network = frame.CALLSIGN_flt.astype(str).str.upper().str.strip()
    table = pd.DataFrame({
        "mvt_prefix": movement.str.extract(r"^([A-Z]+)")[0],
        "flt_prefix": network.str.extract(r"^([A-Z]+)")[0],
        "mvt_number": movement.str.extract(r"([0-9].*)$")[0],
        "flt_number": network.str.extract(r"([0-9].*)$")[0]})
    table = table.loc[table.mvt_prefix.notna() & table.flt_prefix.notna() & table.mvt_number.eq(table.flt_number)]
    counts = table.groupby(["mvt_prefix", "flt_prefix"]).size().reset_index(name="n")
    best = counts.sort_values("n", ascending=False).drop_duplicates("mvt_prefix")
    return dict(zip(best.mvt_prefix, best.flt_prefix))


def candidates(callsigns, mapping):
    """Every spelling a movement callsign might take in an ADS-B feed."""
    out = []
    for value in callsigns:
        text = str(value).upper().strip()
        options = {text}
        match = PREFIX.match(text)
        if match:
            prefix = match.group(1)
            rest = text[len(prefix):]
            target = mapping.get(prefix)
            if target:
                options.add(target + rest)
                options.add(target + rest.lstrip("0"))
        out.append(options)
    return out


def visits(tracks, stands=None):
    """One row per aircraft visit to an airport, with the push-back evidence and where it stood."""
    stands = stand_vertices() if stands is None else stands
    tracks = tracks.copy()
    tracks["callsign"] = tracks.callsign.astype(str).str.upper().str.strip()
    tracks = tracks.sort_values(["hex", "airport", "ts"])
    rows = []
    for (hexid, airport), group in tracks.groupby(["hex", "airport"], observed=True):
        ts = group.ts.to_numpy(float)
        gs = pd.to_numeric(group.gs, errors="coerce").to_numpy(float)
        lat = group.lat.to_numpy(float)
        lon = group.lon.to_numpy(float)
        names = group.callsign.to_numpy()
        vertices = stands.get(airport)
        breaks = np.flatnonzero(np.diff(ts) > 3600)
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks + 1, [len(ts)]])
        for start, end in zip(starts, ends):
            piece_ts, piece_gs = ts[start:end], gs[start:end]
            piece_lat, piece_lon = lat[start:end], lon[start:end]
            if len(piece_ts) < 3:
                continue
            moving = np.isfinite(piece_gs) & (piece_gs >= MOVE_KT)
            first_move = int(np.argmax(moving)) if moving.any() else None
            resting = np.zeros(len(piece_ts), bool)
            if first_move:
                resting[:first_move] = np.isfinite(piece_gs[:first_move]) & (piece_gs[:first_move] < REST_KT)
            rest_points = int(resting.sum())
            rest_seconds = float(piece_ts[first_move] - piece_ts[0]) if first_move else 0.0
            at_stand = False
            if rest_points:
                at_stand = near_stand(float(np.median(piece_lat[resting])),
                                      float(np.median(piece_lon[resting])), vertices)
            usable = (first_move is not None and rest_points >= MIN_REST_POINTS
                      and rest_seconds >= MIN_REST_SECONDS and at_stand)
            label = pd.Series(names[start:end])
            label = label[label.ne("") & label.ne("NONE")]
            rows.append({"hex": hexid, "airport": airport,
                         "callsign_adsb": label.mode().iloc[0] if len(label) else "",
                         "t_first": piece_ts[0], "t_last": piece_ts[-1], "points": len(piece_ts),
                         "t_offblock": piece_ts[first_move - 1] if usable else np.nan,
                         "rest_points": rest_points, "rest_seconds": rest_seconds,
                         "at_stand": at_stand, "usable": usable})
    return pd.DataFrame(rows)


def match(departures, visit_table, mapping):
    """Match on any candidate spelling of the callsign plus a take-off time within five minutes."""
    options = candidates(departures.callsign, mapping)
    expanded = pd.DataFrame({"MVT_ID_mvt": np.repeat(departures.MVT_ID_mvt.to_numpy(), [len(o) for o in options]),
                             "airport": np.repeat(departures.airport.to_numpy(), [len(o) for o in options]),
                             "mvt_epoch": np.repeat(departures.mvt_epoch.to_numpy(), [len(o) for o in options]),
                             "callsign_adsb": [c for o in options for c in o]})
    joined = expanded.merge(visit_table, on=["callsign_adsb", "airport"], how="inner")
    joined = joined.loc[(joined.t_last - joined.mvt_epoch).abs().le(300)]
    joined["closeness"] = (joined.t_last - joined.mvt_epoch).abs()
    joined = joined.sort_values(["closeness", "points"], ascending=[True, False]).drop_duplicates("MVT_ID_mvt")
    return joined.drop(columns=["closeness"])
