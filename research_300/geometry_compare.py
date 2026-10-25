"""Stand geometry: X-Plane Gateway against OpenStreetMap against where aircraft actually park. Label-free.

Everything here uses the 62 SCORED days (no block times exist for them), so no labelled date is touched and the
labelled days stay fresh for the geometry test that follows.

Per airport:
  names      share of scored departures whose STAND_mvt maps, in OSM (stand_detector.lookup, the validated rule),
             in X-Plane (the same rule applied to the X-Plane table), and in either
  agreement  for stand names in both maps, distance from the X-Plane point to the nearest OSM vertex
  empirical  where aircraft park: for each matched scored departure, the longest run of consecutive stationary
             points (below REST_KT, at least 2 points over at least 600 s) within WINDOW_M of either map's position
             for its stand; its median position is that flight's parked position, and the median over flights is
             the stand's empirical position. The 600 s floor matters: with 60 s the longest stillness was often
             the engine-start stop 30 to 70 m after the push-back, which favoured OSM's long lead-in lines
             (smoke run, 25 September: Munich 32 m to OSM against 90 m to X-Plane while the maps agree to 5.7 m). Distances from it to each map are compared ONLY on stands present in
             both maps, so neither map is favoured by where the window was centred.
  heading    for X-Plane stands, the bearing of the push-back (from the parked position to the first later point
             more than 20 m away) against the stand heading + 180 degrees

Per-stand tables go to research_300/geometry/empirical_{ICAO}.csv; aggregates to research_300/results/.
Run from the repository root:  python3 -u -B research_300/geometry_compare.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADSB = Path("adsb-lol")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ADSB))
import features_geometry as FG  # noqa: E402
import offblock_from_tracks as OB  # noqa: E402
from research_290 import munich_dev as MD  # noqa: E402
from research_290 import scored_coverage_2026 as SC  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402

AIRPORTS = ("EDDM", "EHAM", "EDDF")
WINDOW_M, MIN_STILL_S, PUSH_M = 150.0, 600.0, 20.0   # 600 s: a turnaround, never the stop after a push-back
OUT = ROOT / "research_300/results/geometry_compare.json"


def xplane_table(icao):
    """Same shape as stand_detector.stand_table: normalised key -> list of (1, 2) vertex arrays."""
    table, headings = {}, {}
    for s in json.loads((ROOT / f"research_300/geometry/xplane_{icao}.json").read_text())["stands"]:
        key = FG.stand_schluessel(s["name"])
        if not key:
            continue
        table.setdefault(key, []).append(np.array([[s["lat"], s["lon"]]]))
        headings.setdefault(key, []).append(s["heading"])
    return table, headings


def bearing(lat1, lon1, lat2, lon2):
    y = np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2))
    x = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - \
        np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1))
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def parked(ts, lat, lon, gs, anchors):
    """Median position of the longest stationary run near the anchors, and the push-back bearing after it."""
    near = SD._metres(lat, lon, anchors) <= WINDOW_M
    still = np.isfinite(gs) & (gs < SD.REST_KT) & near
    best, i = None, 0
    while i < len(ts):
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ts) and still[j + 1]:
            j += 1
        if j > i and ts[j] - ts[i] >= MIN_STILL_S and (best is None or ts[j] - ts[i] > ts[best[1]] - ts[best[0]]):
            best = (i, j)
        i = j + 1
    if best is None:
        return None
    a, b = best
    plat, plon = float(np.median(lat[a:b + 1])), float(np.median(lon[a:b + 1]))
    push = None
    for k in range(b + 1, len(ts)):
        if SD._metres(lat[k:k + 1], lon[k:k + 1], np.array([[plat, plon]]))[0] > PUSH_M:
            push = float(bearing(plat, plon, lat[k], lon[k]))
            break
    return plat, plon, push


def angle(a, b):
    return float(abs((a - b + 180) % 360 - 180))


def main():
    frame = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["PHASE_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"])
    frame = frame[frame.PHASE_mvt.eq("DEP") & frame.TAXITIME_SEC_mvt.isna()]
    days = sorted(pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").unique())
    if len(sys.argv) > 1:                          # smoke runs: the first N scored days only
        days = days[:int(sys.argv[1])]
    mapping, report = OB.callsign_mapping(), {}
    maps = {ap: (SD.stand_table(ap), *xplane_table(ap)) for ap in AIRPORTS}
    flights = {ap: [] for ap in AIRPORTS}
    names = {ap: [] for ap in AIRPORTS}
    for day in days:
        tracks = pd.read_parquet(ADSB / f"tracks/{day}.parquet")
        dep_all = SC.scored_departures(day)
        for ap in AIRPORTS:
            osm, xp, xh = maps[ap]
            dep = dep_all[dep_all.airport.eq(ap)]
            names[ap] += list(dep.STAND_mvt.astype(str))
            t = tracks[tracks.airport.astype(str).eq(ap)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
            if not len(t) or not len(dep):
                continue
            j = dep.merge(OB.match(dep, OB.visits(t), mapping), on=["MVT_ID_mvt", "airport"], how="inner",
                          suffixes=("", "_v"))
            j = j[j.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
            for row in j.itertuples(index=False):
                vo, _ = SD.lookup(osm, row.STAND_mvt)
                vx, _ = SD.lookup(xp, row.STAND_mvt)
                if vo is None and vx is None:
                    continue
                anchors = np.vstack([v for v in (vo, vx) if v is not None])
                ts, lat, lon, gs, _ = MD.points_of(t, row)
                keep = np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lon) & (ts <= row.mvt_epoch)
                order = np.argsort(ts[keep], kind="stable")
                p = parked(*(a[keep][order] for a in (ts, lat, lon, gs)), anchors)
                if p is not None:
                    flights[ap].append({"stand": FG.stand_schluessel(str(row.STAND_mvt)), "lat": p[0], "lon": p[1],
                                        "push_bearing": p[2], "in_osm": vo is not None, "in_xplane": vx is not None})
        print(day, {ap: len(flights[ap]) for ap in AIRPORTS}, flush=True)

    for ap in AIRPORTS:
        osm, xp, xh = maps[ap]
        n = pd.Series(names[ap])
        st_osm = n.map(lambda s: SD.lookup(osm, s)[1])
        st_xp = n.map(lambda s: SD.lookup(xp, s)[1])
        ok = ("ok", "ok_subdivided")
        in_o, in_x = st_osm.isin(ok), st_xp.isin(ok)
        both_keys = [k for k in xp if k in osm]
        agree = [float(SD._metres(xp[k][0][:, 0], xp[k][0][:, 1], np.vstack(osm[k]))[0]) for k in both_keys]
        f = pd.DataFrame(flights[ap])
        emp = f.groupby("stand").agg(n=("lat", "size"), lat=("lat", "median"), lon=("lon", "median")).reset_index()
        rows = []
        for r in emp.itertuples(index=False):
            vo, _ = SD.lookup(osm, r.stand)
            vx, _ = SD.lookup(xp, r.stand)
            here = np.array([r.lat]), np.array([r.lon])
            rows.append({"stand": r.stand, "flights": int(r.n), "lat": r.lat, "lon": r.lon,
                         "d_osm_m": float(SD._metres(*here, vo)[0]) if vo is not None else np.nan,
                         "d_xplane_m": float(SD._metres(*here, vx)[0]) if vx is not None else np.nan})
        per = pd.DataFrame(rows)
        per.to_csv(ROOT / f"research_300/geometry/empirical_{ap}.csv", index=False)
        both = per.dropna(subset=["d_osm_m", "d_xplane_m"])
        w = both.flights.to_numpy(float)
        wq = lambda v, q: float(np.interp(q, np.cumsum(w[np.argsort(v)]) / w.sum(), np.sort(v))) if len(v) else None
        heads = []
        for r in f[f.in_xplane & f.push_bearing.notna()].itertuples(index=False):
            hs = xh.get(r.stand) or []
            if len(hs) == 1:
                heads.append(angle(r.push_bearing, (hs[0] + 180) % 360))
        report[ap] = {
            "scored_departures": int(len(n)),
            "names_mapped_share": {"osm": float(in_o.mean()), "xplane": float(in_x.mean()), "either": float((in_o | in_x).mean()),
                                   "xplane_only": float((in_x & ~in_o).mean()), "osm_only": float((in_o & ~in_x).mean())},
            "map_agreement_m": {"names_in_both": len(both_keys),
                                "median": float(np.median(agree)) if agree else None,
                                "p90": float(np.percentile(agree, 90)) if agree else None,
                                "share_within_30_m": float(np.mean(np.array(agree) <= 30)) if agree else None},
            "empirical": {"flights_with_parked_position": int(len(f)), "stands": int(len(per)),
                          "stands_in_both_maps": int(len(both)), "flights_on_those_stands": int(w.sum()),
                          "flight_weighted_median_m": {"osm": wq(both.d_osm_m.to_numpy(), .5), "xplane": wq(both.d_xplane_m.to_numpy(), .5)},
                          "flight_weighted_p90_m": {"osm": wq(both.d_osm_m.to_numpy(), .9), "xplane": wq(both.d_xplane_m.to_numpy(), .9)},
                          "flight_share_within_30_m": {"osm": float(w[both.d_osm_m.to_numpy() <= 30].sum() / w.sum()) if len(w) else None,
                                                       "xplane": float(w[both.d_xplane_m.to_numpy() <= 30].sum() / w.sum()) if len(w) else None},
                          "stands_where_xplane_is_closer": int((both.d_xplane_m < both.d_osm_m).sum())},
            "heading": {"pushbacks": len(heads), "median_diff_deg": float(np.median(heads)) if heads else None,
                        "share_within_30_deg": float(np.mean(np.array(heads) <= 30)) if heads else None}}
        print(ap, json.dumps(report[ap]), flush=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
