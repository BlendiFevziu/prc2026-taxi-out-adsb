"""LABEL-FREE pilot: does adsb.lol's preferred release add detector estimates on the scored days? No candidate.

All 62 scored days were extracted from prod-0; on 24 of them adsb.lol prefers staging-0. Listed sizes
(research_300/results/preferred_release_sizes.json) differ by at most 0.05 % on 22 of those days, by 0.34 % on
24 July 2026 and by 3.9 % on 11 July 2026, so only those two were fetched from staging-0, into
tracks/{date}-preferred.parquet (research_300/fetch_preferred.py --suffix=-preferred).

Per day and airport it runs the exact v55 configuration (research_300/build_combined_candidate.estimates: EDDM
X-Plane-first, EHAM X-Plane-first plus the tow guard) on both track files and compares which scored main-lane rows
get an estimate, and whether shared rows get the same value.
Run from the repository root:  python3 -u -B research_300/preferred_pilot_compare.py
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
from research_300 import build_combined_candidate as BC  # noqa: E402
from research_300 import geometry_compare as GC  # noqa: E402

DAYS = ("2026-07-11", "2026-07-24")
OUT = ROOT / "research_300/results/preferred_pilot.json"


def ground_aircraft(tracks, airport):
    t = tracks[tracks.airport.astype(str).eq(airport)]
    return int(t[t.on_ground.astype(bool)].hex.nunique())


def main():
    mapping = OB.callsign_mapping()
    maps = {ap: (SD.stand_table(ap), GC.xplane_table(ap)[0]) for ap in BC.CONFIG}
    report = {}
    for day in DAYS:
        prod = pd.read_parquet(MD.ADSB / f"tracks/{day}.parquet")
        pref = pd.read_parquet(MD.ADSB / f"tracks/{day}-preferred.parquet")
        rec = {"points": {"prod0": int(len(prod)), "preferred": int(len(pref))}, "airports": {}}
        for ap, (geometry, tow_guard) in BC.CONFIG.items():
            a, _ = BC.estimates(prod, day, ap, geometry, tow_guard, mapping, maps[ap])
            b, _ = BC.estimates(pref, day, ap, geometry, tow_guard, mapping, maps[ap])
            both = sorted(set(a) & set(b))
            diff = np.array([b[k] - a[k] for k in both]) if both else np.array([])
            rec["airports"][ap] = {
                "ground_aircraft": {"prod0": ground_aircraft(prod, ap), "preferred": ground_aircraft(pref, ap)},
                "estimates": {"prod0": len(a), "preferred": len(b)},
                "only_prod0": len(set(a) - set(b)), "only_preferred": len(set(b) - set(a)), "both": len(both),
                "both_identical": int((np.abs(diff) < 1e-6).sum()),
                "both_abs_diff_s": {"median": float(np.median(np.abs(diff))) if len(diff) else None,
                                    "max": float(np.abs(diff).max()) if len(diff) else None}}
        report[day] = rec
        print(day, json.dumps(rec), flush=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
