"""POST-HOC, after the v2 verdict: what v1 would have estimated on the rows v2 withdrew as arrival transitions.

It changes nothing in results/munich_validate_v2.json. It shows whether the added rule removed real failures on
the fresh dates or good estimates. Aggregates only; row-level data stays in .cache.
Run from the repository root after munich_validate_v2.py:  python3 -u -B research_290/v2_withdrawn_rows.py
"""
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_290 import munich_dev as MD  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402

AIRPORT = "EDDM"
ROWS = ROOT / ".cache/munich_stand_detector/validation_rows_v2.parquet"
OUTPUT = ROOT / "research_290/results/v2_withdrawn_rows.json"


def main():
    rows = pd.read_parquet(ROWS)
    withdrawn = rows[rows.status.eq("arrival_transition")]
    table, report = SD.stand_table(AIRPORT), []
    for date, group in withdrawn.groupby("date"):
        tracks, _, joined = MD.matched_set(date, AIRPORT)
        joined = joined[joined.unique].set_index("MVT_ID_mvt")
        for r in group.itertuples(index=False):
            row = joined.loc[r.MVT_ID_mvt]
            vertices, _ = SD.lookup(table, row.STAND_mvt)
            ts, lat, lon, gs, _ = MD.points_of(tracks, row)
            v1 = SD.detect_onset(ts, lat, lon, gs, vertices, row.mvt_epoch, SD.R_PARK_M)
            report.append({"date": date, "v1_status": v1["status"],
                           "v1_error_s": round((v1["t_before"] + v1["t_after"]) / 2 - row.block_epoch, 1)})
    OUTPUT.write_text(json.dumps({"note": "POST-HOC after the v2 verdict; description only",
                                  "withdrawn": report}, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
