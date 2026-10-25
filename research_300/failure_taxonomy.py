"""Where the stand detector loses departures, at EDDM, EHAM and EDDF: matching, stand mapping, or never seen parked.

The detector is fixed (stand_detector.detect_onset_v2, OSM geometry, as validated). Every departure gets ONE
first-failing category, in pipeline order:

  matching     no_callsign_in_feed       no candidate spelling of its callsign appears anywhere in the day's tracks
               callsign_not_at_airport   it appears, but never at this airport
               no_visit_near_takeoff     it appears here, but no visit ends within 300 s of the movement time
               visit_claimed_twice       its visit is claimed by more than one departure
  mapping      stand_unmapped, stand_ambiguous   (stand_detector.lookup)
  observation  never_seen_parked         no stationary stretch of 600 s anywhere before take-off, and no point at
                                         the stand: the aircraft was only ever seen moving
               parked_elsewhere          parked (>= 600 s still) but more than 150 m from the recorded stand
               parked_near_stand         parked 80 to 150 m from it: a map offset or a neighbouring stand
               only_moving_at_stand      points within 80 m of the stand, but none stationary there
               parked_beyond_30m         stationary within 80 m of the stand but never within 30 m (R_PARK)
  detection    still_no_transition       stationary within 30 m, but no move out was observed after it
               gap_too_large, ambiguous_onsets, arrival_transition   (the detector's own abstentions)
  estimate     ok

For mapping and geometry-sensitive categories it also reports, label-free, whether the X-Plane geometry
(research_300/geometry/xplane_{ICAO}.json) would give an estimate instead: the upper bound of what a map can fix.

Categories use no label and run on the 62 scored days. Accuracy against the recorded block is computed only on
labelled days already spent: the nine December 2025 days and 13 July 2025 at all three airports, and the four
January 2025 days at EDDM. The January 2025 days at EHAM and EDDF are NOT read, so they stay fresh.
Run from the repository root:  python3 -u -B research_300/failure_taxonomy.py [N scored days for a smoke run]
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
import offblock_from_tracks as OB  # noqa: E402
from research_290 import munich_dev as MD  # noqa: E402
from research_290 import scored_coverage_2026 as SC  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402
from research_300 import geometry_compare as GC  # noqa: E402

AIRPORTS = ("EDDM", "EHAM", "EDDF")
PARKED_S, ELSEWHERE_M = 600.0, 150.0
LABELLED = {ap: ["2025-12-03", "2025-12-05", "2025-12-08", "2025-12-10", "2025-12-11", "2025-12-13", "2025-12-14",
                 "2025-12-17", "2025-12-20", "2025-07-13"] for ap in AIRPORTS}
LABELLED["EDDM"] += ["2025-01-08", "2025-01-16", "2025-01-24", "2025-01-26"]
GROUP = {"no_callsign_in_feed": "matching", "callsign_not_at_airport": "matching",
         "no_visit_near_takeoff": "matching", "visit_claimed_twice": "matching",
         "stand_unmapped": "mapping", "stand_ambiguous": "mapping",
         "never_seen_parked": "observation", "parked_elsewhere": "observation", "parked_near_stand": "observation",
         "only_moving_at_stand": "observation", "parked_beyond_30m": "observation",
         "still_no_transition": "detection", "gap_too_large": "detection", "ambiguous_onsets": "detection",
         "arrival_transition": "detection", "no_points": "detection", "ok": "estimate"}
OUT = ROOT / "research_300/results/failure_taxonomy.json"


def still_runs(ts, lat, lon, gs, min_s):
    """(start, end) index pairs of consecutive stationary points lasting at least min_s seconds."""
    still = np.isfinite(gs) & (gs < SD.REST_KT)
    runs, i = [], 0
    while i < len(ts):
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ts) and still[j + 1]:
            j += 1
        if ts[j] - ts[i] >= min_s:
            runs.append((i, j))
        i = j + 1
    return runs


def observation_category(ts, lat, lon, gs, vertices, t_takeoff):
    """Why a mapped, matched departure gave no parked transition. Uses the same filtered points as detect_onset."""
    keep = np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lon) & (ts <= t_takeoff + SD.TAKEOFF_MARGIN_S)
    order = np.argsort(ts[keep], kind="stable")
    ts, lat, lon, gs = (a[keep][order] for a in (ts, lat, lon, gs))
    if not len(ts):
        return "never_seen_parked"
    d = SD._metres(lat, lon, vertices)
    still = np.isfinite(gs) & (gs < SD.REST_KT)
    if (d <= SD.R_STAND_M).any():
        if (still & (d <= SD.R_PARK_M)).any():
            return "still_no_transition"
        return "parked_beyond_30m" if (still & (d <= SD.R_STAND_M)).any() else "only_moving_at_stand"
    runs = still_runs(ts, lat, lon, gs, PARKED_S)
    if not runs:
        return "never_seen_parked"
    gap = min(float(np.median(d[a:b + 1])) for a, b in runs)
    return "parked_near_stand" if gap <= ELSEWHERE_M else "parked_elsewhere"


def classify(day, airport, tracks_all, dep, mapping, osm, xp):
    t = tracks_all[tracks_all.airport.astype(str).eq(airport)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
    feed = set(tracks_all.callsign.dropna().astype(str).str.strip().str.upper())
    here = set(t.callsign.dropna().astype(str).str.strip().str.upper())
    options = OB.candidates(dep.callsign, mapping)
    visits = OB.visits(t) if len(t) else None
    joined = dep.merge(OB.match(dep, visits, mapping), on=["MVT_ID_mvt", "airport"], how="inner",
                       suffixes=("", "_v")) if visits is not None and len(visits) else dep.iloc[:0]
    claims = joined.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size") if len(joined) else pd.Series(dtype=int)
    joined = joined.assign(unique=claims.eq(1).to_numpy() if len(joined) else [])
    by_id = joined.set_index("MVT_ID_mvt")
    rows = []
    for k, d in enumerate(dep.itertuples(index=False)):
        rec = {"day": day, "airport": airport, "MVT_ID_mvt": d.MVT_ID_mvt, "lane": d.lane,
               "block_epoch": getattr(d, "block_epoch", np.nan), "xplane_ok": None}
        if d.MVT_ID_mvt not in by_id.index:
            rec["category"] = ("no_callsign_in_feed" if not (options[k] & feed) else
                               "callsign_not_at_airport" if not (options[k] & here) else "no_visit_near_takeoff")
            rows.append(rec)
            continue
        row = by_id.loc[d.MVT_ID_mvt]
        if not row.unique:
            rec["category"] = "visit_claimed_twice"
            rows.append(rec)
            continue
        ts, lat, lon, gs, _ = MD.points_of(t, row)
        vx, _ = SD.lookup(xp, d.STAND_mvt)
        if vx is not None:
            rec["xplane_ok"] = SD.detect_onset_v2(ts, lat, lon, gs, vx, d.mvt_epoch, SD.R_PARK_M)["status"] == "ok"
        vo, status = SD.lookup(osm, d.STAND_mvt)
        if vo is None:
            rec["category"] = "stand_unmapped" if status == "unmapped" else "stand_ambiguous"
            rows.append(rec)
            continue
        out = SD.detect_onset_v2(ts, lat, lon, gs, vo, d.mvt_epoch, SD.R_PARK_M)
        if out["status"] == "ok":
            rec["category"], rec["estimate"] = "ok", (out["t_before"] + out["t_after"]) / 2
        elif out["status"] in ("no_occupancy", "no_parked_onset"):
            rec["category"] = observation_category(ts, lat, lon, gs, vo, d.mvt_epoch)
        else:
            rec["category"] = out["status"]
        rows.append(rec)
    return rows


def summary(frame):
    n = len(frame)
    cat = frame.category.value_counts()
    grp = frame.category.map(GROUP).value_counts()
    fixable = frame[frame.category.isin(["stand_unmapped", "stand_ambiguous", "parked_near_stand",
                                         "parked_beyond_30m", "only_moving_at_stand"])]
    return {"departures": n, "groups_share": {g: round(float(v / n), 4) for g, v in grp.items()},
            "categories": {c: int(v) for c, v in cat.items()},
            "geometry_sensitive_rows": int(len(fixable)),
            "of_which_xplane_gives_an_estimate": int(fixable.xplane_ok.eq(True).sum()),
            "xplane_estimates_where_osm_gives_none": int(((frame.category != "ok") & frame.xplane_ok.eq(True)).sum()),
            "osm_estimates": int((frame.category == "ok").sum()),
            "osm_estimates_main_lane_per_departure": round(float(((frame.category == "ok") & (frame.lane == "main")).sum() / max(n, 1)), 4)}


def main():
    mapping = OB.callsign_mapping()
    maps = {ap: (SD.stand_table(ap), GC.xplane_table(ap)[0]) for ap in AIRPORTS}
    frame = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["PHASE_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"])
    frame = frame[frame.PHASE_mvt.eq("DEP") & frame.TAXITIME_SEC_mvt.isna()]
    scored_days = sorted(pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").unique())
    if len(sys.argv) > 1:
        scored_days = scored_days[:int(sys.argv[1])]
    scored, labelled = [], []
    for day in scored_days:
        tracks = pd.read_parquet(ADSB / f"tracks/{day}.parquet")
        dep_all = SC.scored_departures(day)
        for ap in AIRPORTS:
            scored += classify(day, ap, tracks, dep_all[dep_all.airport.eq(ap)], mapping, *maps[ap])
        print("scored", day, len(scored), flush=True)
    for ap in AIRPORTS:
        for day in (LABELLED[ap] if len(sys.argv) == 1 else LABELLED[ap][:1]):
            tracks = pd.read_parquet(ADSB / f"tracks/{day}.parquet")
            labelled += classify(day, ap, tracks, MD.departures(day, ap), mapping, *maps[ap])
        print("labelled", ap, len(labelled), flush=True)
    scored, labelled = pd.DataFrame(scored), pd.DataFrame(labelled)
    report = {"note": "detector v2 with OSM geometry, fixed; categories label-free", "scored_days": len(scored_days)}
    for ap in AIRPORTS:
        s, l = scored[scored.airport.eq(ap)], labelled[labelled.airport.eq(ap)]
        ok = l[l.category.eq("ok")]
        err = (ok.estimate - ok.block_epoch).to_numpy(float)
        report[ap] = {"scored": summary(s), "labelled": summary(l),
                      "labelled_accuracy": {"days": LABELLED[ap], "estimates": int(len(err)),
                                            "within_60": int((np.abs(err) <= 60).sum()),
                                            "late_over_120": int((err > 120).sum()), "early_over_120": int((err < -120).sum()),
                                            "median_s": float(np.median(err)) if len(err) else None}}
        print(ap, json.dumps(report[ap]["scored"]["groups_share"]), flush=True)
    (ROOT / ".cache/research_300").mkdir(parents=True, exist_ok=True)
    scored.to_parquet(ROOT / ".cache/research_300/taxonomy_scored.parquet", index=False)
    labelled.to_parquet(ROOT / ".cache/research_300/taxonomy_labelled.parquet", index=False)
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
