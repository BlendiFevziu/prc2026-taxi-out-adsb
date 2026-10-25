"""Does the FROZEN stand detector fire on the scored July 2026 dates? Label-free: scored departures carry no block time.

Gate 0 (adsb_coverage_gate.py) showed Munich has ADS-B ground points on 6 and 13 July 2026. That is necessary, not
sufficient: the detector also needs a mapped stand and an observed parked-to-motion transition. This runs the frozen
detector, unchanged, on scored 2026 Munich departures matched to those two days' tracks.

The tracks are truncated archives holding about a fifth of the day's aircraft, so the number of matches is limited by
sampling. The informative quantity is estimates PER MATCHED departure, which is comparable with the December
development and validation days, since sampling by aircraft is unrelated to how a flight's stand visit was reported.

Run from the repository root:  python3 -u -B research_290/july2026_detector_coverage.py
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

AIRPORT = "EDDM"
DATES = ("2026-07-06", "2026-07-13")
OUTPUT = ROOT / "research_290/results/july2026_detector_coverage.json"


def scored_departures(date):
    frame = pd.read_parquet(ROOT / "data/ranking.parquet",
                            columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "STAND_mvt",
                                     "MVT_TIME_UTC_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt"])
    frame = frame[frame.PHASE_mvt.eq("DEP") & frame.ADEP_mvt.astype(str).eq(AIRPORT) & frame.TAXITIME_SEC_mvt.isna()]
    t = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
    frame = frame[t.dt.strftime("%Y-%m-%d").eq(date)].reset_index(drop=True)
    frame["mvt_epoch"] = (pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True) - MD.EPOCH0).dt.total_seconds()
    frame["callsign"] = frame.FLIGHT_mvt.astype(str).str.strip().str.upper()
    frame["airport"] = frame.ADEP_mvt.astype(str)
    frame["lane"] = np.where(frame.AOBT_3_flt.notna(), "main", "fallback")
    return frame


def main():
    mapping, table, result = OB.callsign_mapping(), SD.stand_table(AIRPORT), {}
    for date in DATES:
        tracks = pd.read_parquet(ADSB / f"tracks/{date}-partial.parquet")
        tracks = tracks[tracks.airport.astype(str).eq(AIRPORT)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
        dep = scored_departures(date)
        joined = dep.merge(OB.match(dep, OB.visits(tracks), mapping), on=["MVT_ID_mvt", "airport"], how="inner",
                           suffixes=("", "_v"))
        joined = joined[joined.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
        statuses, stands = [], []
        for row in joined.itertuples(index=False):
            vertices, stand_status = SD.lookup(table, row.STAND_mvt)
            stands.append(stand_status)
            if vertices is None:
                statuses.append(stand_status)
                continue
            ts, lat, lon, gs, _ = MD.points_of(tracks, row)
            statuses.append(SD.detect_onset(ts, lat, lon, gs, vertices, row.mvt_epoch, SD.R_PARK_M)["status"])
        s = pd.Series(statuses, dtype=object)
        result[date] = {"scored_departures": int(len(dep)), "matched_in_sample": int(len(joined)),
                        "stand_status": pd.Series(stands, dtype=object).value_counts().to_dict(),
                        "detector_status": s.value_counts().to_dict(),
                        "estimates": int(s.eq("ok").sum()),
                        "estimates_per_matched": float(s.eq("ok").mean()) if len(s) else None}
        print(date, json.dumps(result[date]), flush=True)
    OUTPUT.write_text(json.dumps({"note": "label-free; frozen detector on scored July 2026 samples", "dates": result},
                                 indent=2) + "\n")
    print(f"written {OUTPUT}")


if __name__ == "__main__":
    main()
