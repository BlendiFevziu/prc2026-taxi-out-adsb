"""Report-only ("dry") evaluation of ONE labelled day with the detector and thresholds frozen. No decision, no tuning.

Configuration, frozen as of 25 September 2026: stand_detector.detect_onset_v2 with its constants plus v54's
arrival-and-stay guard (arm_test.guarded); geometry as adopted by the geometry test (X-Plane-first at EDDM and EHAM,
OSM at EDDF); matching on the FLIGHT_mvt spellings (the filed-callsign rule was rejected). The v54 production
geometry (OSM everywhere) is reported alongside as a reference.

Per airport and date: departures; matching failures by cause; stand-mapping failures; every abstention; estimates
and coverage; onset error against the recorded block; held-out RMSE on the main lane against the v45 full ensemble's
held-out prediction (policy = detector where it estimates, ensemble elsewhere). Completeness: every departure is
counted exactly once, and every main-lane departure is found in the comparator.

Run from the repository root:  python3 -u -B research_300/dry_day_report.py 2025-01-28 [more dates]
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "adsb-lol")
import offblock_from_tracks as OB  # noqa: E402
from research_290 import munich_dev as MD  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402
from research_300 import arm_test as AT  # noqa: E402
from research_300 import failure_taxonomy as FT  # noqa: E402
from research_300 import geometry_compare as GC  # noqa: E402

AIRPORTS = ("EDDM", "EHAM", "EDDF")
CURRENT = {"EDDM": "xplane_first", "EHAM": "xplane_first", "EDDF": "osm"}
OUT = ROOT / "research_300/results"


def classify(date, airport, geometry, mapping, osm, xp, tracks_all):
    t = tracks_all[tracks_all.airport.astype(str).eq(airport)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
    feed = set(tracks_all.callsign.dropna().astype(str).str.strip().str.upper())
    here = set(t.callsign.dropna().astype(str).str.strip().str.upper())
    dep = MD.departures(date, airport)
    options = AT.options_for(dep, "flight_mvt", mapping)
    visits = OB.visits(t) if len(t) else None
    j = AT.match_candidates(dep, visits, options) if visits is not None and len(visits) else pd.DataFrame(columns=["MVT_ID_mvt"])
    j = dep.merge(j, on=["MVT_ID_mvt", "airport"], how="inner", suffixes=("", "_v")) if len(j) else dep.iloc[:0]
    unique = j.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1) if len(j) else pd.Series(dtype=bool)
    by_id = j.assign(unique=unique.to_numpy() if len(j) else []).set_index("MVT_ID_mvt")
    rows = []
    for k, d in enumerate(dep.itertuples(index=False)):
        rec = {"date": date, "airport": airport, "MVT_ID_mvt": d.MVT_ID_mvt, "lane": d.lane,
               "block_epoch": d.block_epoch, "mvt_epoch": d.mvt_epoch, "taxi": np.nan}
        if d.MVT_ID_mvt not in by_id.index:
            rec["category"] = ("no_callsign_in_feed" if not (options[k] & feed) else
                               "callsign_not_at_airport" if not (options[k] & here) else "no_visit_near_takeoff")
        elif not by_id.loc[d.MVT_ID_mvt, "unique"]:
            rec["category"] = "visit_claimed_twice"
        else:
            row = by_id.loc[d.MVT_ID_mvt]
            vertices, status = AT.lookup(geometry, osm, xp, d.STAND_mvt)
            if vertices is None:
                rec["category"] = "stand_unmapped" if status == "unmapped" else "stand_ambiguous"
            else:
                ts, lat, lon, gs, _ = MD.points_of(t, row)
                o = AT.guarded(ts, lat, lon, gs, vertices, d.mvt_epoch)
                if o["status"] == "ok":
                    rec["category"], rec["taxi"] = "estimate", d.mvt_epoch - (o["t_before"] + o["t_after"]) / 2
                elif o["status"] in ("no_occupancy", "no_parked_onset"):
                    rec["category"] = "abstain:" + FT.observation_category(ts, lat, lon, gs, vertices, d.mvt_epoch)
                else:
                    rec["category"] = "abstain:" + o["status"]
        rows.append(rec)
    return pd.DataFrame(rows)


def summarise(frame, model):
    n = len(frame)
    est = frame[frame.category.eq("estimate")]
    err = (est.taxi - (est.mvt_epoch - est.block_epoch)).to_numpy(float)
    main = frame[frame.lane.eq("main")].merge(model, on="MVT_ID_mvt", how="left")
    missing = int(main.y.isna().sum())
    main = main[main.y.notna()]
    y, comp = main.y.to_numpy(float), main.baseline.to_numpy(float)
    pol = np.where(np.isfinite(main.taxi.to_numpy(float)), main.taxi.to_numpy(float), comp)
    cov = np.isfinite(main.taxi.to_numpy(float))
    rmse = lambda p, t: float(np.sqrt(np.mean((p - t) ** 2))) if len(t) else None
    return {"departures": n, "main_lane": int(frame.lane.eq("main").sum()),
            "completeness": {"categorised": int(frame.category.notna().sum()) == n,
                             "main_lane_missing_from_comparator": missing},
            "matching_failures": {c: int((frame.category == c).sum()) for c in
                                  ("no_callsign_in_feed", "callsign_not_at_airport", "no_visit_near_takeoff", "visit_claimed_twice")},
            "mapping_failures": {c: int((frame.category == c).sum()) for c in ("stand_unmapped", "stand_ambiguous")},
            "abstentions": frame.loc[frame.category.str.startswith("abstain:"), "category"].str[8:].value_counts().to_dict(),
            "estimates": int(len(est)), "coverage_all": round(len(est) / max(n, 1), 4),
            "coverage_main": round(float(cov.mean()), 4) if len(cov) else None,
            "onset_error_s": {"median": float(np.median(err)) if len(err) else None,
                              "mae": float(np.abs(err).mean()) if len(err) else None,
                              "rmse": float(np.sqrt((err ** 2).mean())) if len(err) else None,
                              "within_60": int((np.abs(err) <= 60).sum()), "within_120": int((np.abs(err) <= 120).sum()),
                              "late_over_120": int((err > 120).sum()), "early_over_120": int((err < -120).sum()),
                              "worst": [float(err.min()), float(err.max())] if len(err) else None},
            "held_out_rmse_main_lane": {"rows": int(len(y)), "comparator_v45": rmse(comp, y), "policy": rmse(pol, y),
                                        "covered_rows": int(cov.sum()), "detector_on_covered": rmse(pol[cov], y[cov]),
                                        "comparator_on_covered": rmse(comp[cov], y[cov])}}


def main():
    dates = sys.argv[1:]
    mapping = OB.callsign_mapping()
    maps = {ap: (SD.stand_table(ap), GC.xplane_table(ap)[0]) for ap in AIRPORTS}
    model = AT.comparator()
    report = {"note": "dry report, frozen detector and thresholds; no decision", "current_geometry": CURRENT, "days": {}}
    for date in dates:
        tracks = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet")
        day = {}
        for ap in AIRPORTS:
            osm, xp = maps[ap]
            day[ap] = {"current": summarise(classify(date, ap, CURRENT[ap], mapping, osm, xp, tracks), model),
                       "v54_osm_reference": summarise(classify(date, ap, "osm", mapping, osm, xp, tracks), model)}
            c = day[ap]["current"]
            print(date, ap, "estimates", c["estimates"], "of", c["departures"], "| held-out policy",
                  c["held_out_rmse_main_lane"]["policy"], "vs v45", c["held_out_rmse_main_lane"]["comparator_v45"], flush=True)
        report["days"][date] = day
    name = "dry_day_report_" + "_".join(dates) + ".json"
    (OUT / name).write_text(json.dumps(report, indent=2) + "\n")
    print("written", OUT / name)


if __name__ == "__main__":
    main()
