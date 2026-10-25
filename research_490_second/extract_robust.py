"""Sampling-aware track features for the partial rows of the five v56 airports (research_490_second/PREREG_second.md).
Codex's eham_recovery.extract (checkout 8d98953) generalised only in the airport: same matching, same stand maps as v56
(X-Plane first at EHAM, OSM elsewhere), the same assertion that every partial row replays its category and 23 summaries
against the frozen v56 extraction, then robust_track_features.summarise. No label is read. One file per airport and day.
Run from the repository root:  python3 -u -B research_490_second/extract_robust.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
EH = HERE.parent / "skenderbeu-eham-readonly"
STAND = HERE / ".cache/research_320/stand"
OUT = HERE / ".cache/research_490/features"
AIRPORTS = ("EDDF", "EGLL", "EHAM", "LEBL", "LSZH")
TAGS = ("train", "volume_extra", "development", "confirm", "fresh_july", "eddf_july_confirm", "fresh_january")
ID = "MVT_ID_mvt"


def main():
    head = subprocess.run(["git", "-C", str(EH), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    if not head.startswith("8d98953"):
        raise SystemExit("the EHAM checkout is not 8d98953")
    sys.path.insert(0, str(EH))
    import main_model  # noqa: F401  (torch before tree libraries)
    from research_300 import arm_test as AT, geometry_compare as GC
    from research_290 import munich_dev as MD, stand_detector as SD
    from research_300.heavy_lock import heavy_work
    from research_310 import robust_track_features as F, partial_track_features as OLD
    from research_310.partial_track_audit import evidence, unique_visit
    ref = pd.concat([pd.read_parquet(STAND / f"{t}.parquet", columns=[ID, "airport", "date", "category"] + list(OLD.FEATURES)) for t in TAGS],
                    ignore_index=True).drop_duplicates(ID)
    ref = ref[ref.airport.isin(AIRPORTS) & ref.category.eq("partial_ground")]
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = AT.OB.callsign_mapping()
    maps = {ap: (SD.stand_table(ap), GC.xplane_table(ap)[0] if ap == "EHAM" else {}) for ap in AIRPORTS}
    for date in sorted(ref.date.unique()):
        todo = [ap for ap in AIRPORTS if not (OUT / f"{ap}_{date}.parquet").exists() and ref[ref.date.eq(date) & ref.airport.eq(ap)].shape[0]]
        if not todo:
            continue
        with heavy_work(f"research_490 robust features {date}", wait=True):
            raw = pd.read_parquet(Path(MD.month_file(date)), columns=[ID, "STAND_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt"])
            for ap in todo:
                old = ref[ref.date.eq(date) & ref.airport.eq(ap)]
                dep = old[[ID, "airport", "date"]].merge(raw, on=ID, validate="one_to_one")
                dep["callsign"] = dep.FLIGHT_mvt.astype(str).str.strip().str.upper()
                dep["mvt_epoch"] = (pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True) - MD.EPOCH0).dt.total_seconds()
                tracks = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet", filters=[("airport", "=", ap)])
                tracks = tracks.drop_duplicates(["hex", "ts"]).reset_index(drop=True)
                visits = AT.OB.visits(tracks)
                joined = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "plus_callsign_flt", mapping)),
                                   on=[ID, "airport"], suffixes=("", "_v"), validate="one_to_one")
                joined["unique"] = joined.groupby(["hex", "t_first"])[ID].transform("size").eq(1)   # as in eham_recovery.extract
                joined = joined.set_index(ID, verify_integrity=True)
                by_hex = {k: g for k, g in tracks.groupby("hex", sort=False)}
                rows = []
                for r in old.itertuples(index=False):
                    mid = getattr(r, ID)
                    j = joined.loc[mid]
                    assert unique_visit(j)
                    h = by_hex[j.hex]
                    points = h[h.ts.between(j.t_first, j.t_last)]
                    stand, _ = AT.lookup("xplane_first", *maps[ap], j.STAND_mvt)
                    assert evidence(points, stand, j.mvt_epoch)["category"] == r.category
                    replay = OLD.summarise(points, stand, j.mvt_epoch)
                    np.testing.assert_array_equal([replay[f] for f in OLD.FEATURES], [getattr(r, f) for f in OLD.FEATURES])
                    rows.append({ID: mid, "airport": ap, "date": date, **F.summarise(points, j.mvt_epoch)})
                pd.DataFrame(rows).to_parquet(OUT / f"{ap}_{date}.parquet", index=False)
                print(date, ap, len(rows), "exact replays", flush=True)


if __name__ == "__main__":
    main()
