"""Frozen v56 extraction (full_coverage_2026.extract_day, OSM stands) with the stand model's 36th input, detector_taxi_s,
for Munich and Madrid: 2025 test days and 2026 scored days. One cached file per airport and day. No label is used.
Same patch as research_450_stand_january/build_v67.py (summarise extended by stand_experiment.detector_taxi).
Run from the repository root:  python3 -u -B research_590_stand_new/extract.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_400_cells"))
from research_320 import stand_experiment as S  # noqa: E402
import cells_confirm as CC  # noqa: E402

MS, R, F = S.MS, S.R, CC.F
ID = "MVT_ID_mvt"
CACHE = ROOT / ".cache/research_590/days"
DAYS_2025 = {"EDDM": CC.JAN, "LEMD": CC.JUL}


def departures_2026(airport):
    raw = pd.read_parquet(ROOT / "data/ranking.parquet", columns=[ID, "PHASE_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt",
                                                                "ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "STAND_mvt"])
    dep = raw[raw.PHASE_mvt.eq("DEP") & raw.TAXITIME_SEC_mvt.isna() & raw.AOBT_3_flt.notna() & raw.ADEP_mvt.eq(airport)].copy()
    dep["date"] = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    dep["airport"] = airport
    dep["callsign"] = dep.FLIGHT_mvt.astype(str).str.strip().str.upper()
    dep["mvt_epoch"] = (pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True) - R.MD.EPOCH0).dt.total_seconds()
    return dep


def extract(airport, dep, days):
    CACHE.mkdir(parents=True, exist_ok=True)
    mapping, maps = R.AT.OB.callsign_mapping(), {airport: (R.SD.stand_table(airport), {})}
    original, saved = MS.summarise, MS.AIRPORTS
    MS.summarise = lambda p, stand, takeoff: {**original(p, stand, takeoff), "detector_taxi_s": S.detector_taxi(p, stand, takeoff)}
    try:
        MS.AIRPORTS = (airport,)
        with R.heavy_work(f"research_590: extraction {airport}", wait=True):
            for d in days:
                path = CACHE / f"{airport}_{d}.parquet"
                if path.exists():
                    continue
                F.extract_day(dep[dep.date.eq(d)], pd.read_parquet(R.MD.ADSB / f"tracks/{d}.parquet"), mapping, maps).to_parquet(path, index=False)
                print(airport, d, flush=True)
    finally:
        MS.summarise, MS.AIRPORTS = original, saved


def load(airport, days):
    return pd.concat([pd.read_parquet(CACHE / f"{airport}_{d}.parquet") for d in days], ignore_index=True)


def main():
    for ap, days in DAYS_2025.items():
        extract(ap, CC.departures(ap, days), days)
    for ap in ("EDDM", "LEMD"):
        dep = departures_2026(ap)
        extract(ap, dep, sorted(dep.date.unique()))


if __name__ == "__main__":
    main()
