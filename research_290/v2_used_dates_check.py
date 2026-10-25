"""What detect_onset_v2 changes on the dates already used: the development day and the four v1 validation dates.

Supports one sentence of PREREG addendum B: the arrival rule removes exactly the one estimate that failed v1 and
nothing else. These dates are spent, so this is development evidence, not validation. Aggregates only.
Run from the repository root:  python3 -u -B research_290/v2_used_dates_check.py
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_290 import munich_dev as MD  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402

AIRPORT = "EDDM"
DATES = ("2025-12-13", "2025-12-03", "2025-12-10", "2025-12-17", "2025-12-20")
OUTPUT = ROOT / "research_290/results/v2_used_dates_check.json"


def main():
    table, report = SD.stand_table(AIRPORT), {}
    for date in DATES:
        tracks, _, joined = MD.matched_set(date, AIRPORT)
        changed = []
        for row in joined[joined.unique].itertuples(index=False):
            vertices, _ = SD.lookup(table, row.STAND_mvt)
            if vertices is None:
                continue
            ts, lat, lon, gs, _ = MD.points_of(tracks, row)
            v1 = SD.detect_onset(ts, lat, lon, gs, vertices, row.mvt_epoch, SD.R_PARK_M)
            v2 = SD.detect_onset_v2(ts, lat, lon, gs, vertices, row.mvt_epoch, SD.R_PARK_M)
            if v1["status"] != v2["status"]:
                changed.append(float((v1["t_before"] + v1["t_after"]) / 2 - row.block_epoch))
        report[date] = {"role": "development" if date == "2025-12-13" else "v1 validation (spent)",
                        "estimates_removed": len(changed),
                        "their_v1_errors_s": [round(e, 1) for e in changed],
                        "removed_within_60_s": int(np.sum(np.abs(changed) <= 60)) if changed else 0}
        print(date, report[date], flush=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
