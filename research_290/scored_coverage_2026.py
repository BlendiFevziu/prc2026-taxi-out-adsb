"""How far the FROZEN v2 detector reaches on the scored set, per airport. Label-free: scored departures carry no
block time, and nothing here compares an estimate with a label.

Two things per airport. (1) ADS-B ground coverage: aircraft seen at the airport and the share with at least one
ground point, on every downloaded coverage day (scored 2026 days and labelled 2025 days alike, since this uses no
labels). (2) On the scored 2026 days only: scored departures, how many match an ADS-B visit uniquely, how their
stands map, and how many get a v2 estimate. Estimates per scored departure is the quantity that multiplies any
per-row gain on the official scale.

Complete daily archives only (fetch_days.py writes tracks only for complete downloads since 24 September).
Run from the repository root:  python3 -u -B research_290/scored_coverage_2026.py
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
from research_290 import stand_detector as SD  # noqa: E402

SCORED_DAYS = ("2026-01-14", "2026-07-15")
GROUND_ONLY_DAYS = ("2025-01-16",)   # 2025-01-15 stalled three times on 24 September; the next day stands in
OUTPUT = ROOT / "research_290/results/scored_coverage_2026.json"


def scored_departures(date):
    frame = pd.read_parquet(ROOT / "data/ranking.parquet",
                            columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "STAND_mvt",
                                     "MVT_TIME_UTC_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt"])
    frame = frame[frame.PHASE_mvt.eq("DEP") & frame.TAXITIME_SEC_mvt.isna()]
    t = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
    frame = frame[t.dt.strftime("%Y-%m-%d").eq(date)].reset_index(drop=True)
    frame["mvt_epoch"] = (pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True) - MD.EPOCH0).dt.total_seconds()
    frame["callsign"] = frame.FLIGHT_mvt.astype(str).str.strip().str.upper()
    frame["airport"] = frame.ADEP_mvt.astype(str)
    frame["lane"] = np.where(frame.AOBT_3_flt.notna(), "main", "fallback")
    return frame


def ground_coverage(tracks):
    out = {}
    for airport, g in tracks.groupby(tracks.airport.astype(str)):
        ground = g[g.on_ground.astype(bool)]
        out[airport] = {"aircraft": int(g.hex.nunique()), "with_ground_points": int(ground.hex.nunique()),
                        "share": round(float(ground.hex.nunique() / max(g.hex.nunique(), 1)), 3)}
    return out


def detector_reach(date, tracks, mapping):
    dep, out = scored_departures(date), {}
    for airport, d in dep.groupby("airport"):
        t = tracks[tracks.airport.astype(str).eq(airport)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
        table = SD.stand_table(airport)
        joined = d.merge(OB.match(d, OB.visits(t), mapping), on=["MVT_ID_mvt", "airport"], how="inner",
                         suffixes=("", "_v")) if len(t) else d.iloc[:0]
        if len(joined):
            joined = joined[joined.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
        statuses = []
        for row in joined.itertuples(index=False):
            vertices, stand_status = SD.lookup(table, row.STAND_mvt)
            if vertices is None:
                statuses.append((row.lane, stand_status))
                continue
            ts, lat, lon, gs, _ = MD.points_of(t, row)
            statuses.append((row.lane, SD.detect_onset_v2(ts, lat, lon, gs, vertices, row.mvt_epoch,
                                                          SD.R_PARK_M)["status"]))
        s = pd.DataFrame(statuses, columns=["lane", "status"]) if statuses else pd.DataFrame(columns=["lane", "status"])
        ok_main = int((s.status.eq("ok") & s.lane.eq("main")).sum())
        out[airport] = {"scored_departures": int(len(d)), "matched_unique": int(len(joined)),
                        "status": s.status.value_counts().to_dict(), "estimates_main": ok_main,
                        "estimates_per_departure": round(ok_main / max(len(d), 1), 4)}
    return out


def main():
    mapping, report = OB.callsign_mapping(), {"note": "label-free; frozen v2 detector", "days": {}}
    for date in SCORED_DAYS + GROUND_ONLY_DAYS:
        path = ADSB / f"tracks/{date}.parquet"
        if not path.exists():
            report["days"][date] = {"missing": True}
            print(date, "no complete tracks", flush=True)
            continue
        tracks = pd.read_parquet(path)
        entry = {"ground_coverage": ground_coverage(tracks)}
        if date in SCORED_DAYS:
            entry["detector_reach"] = detector_reach(date, tracks, mapping)
        report["days"][date] = entry
        print(date, json.dumps(entry)[:600], flush=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"written {OUTPUT}")


if __name__ == "__main__":
    main()
