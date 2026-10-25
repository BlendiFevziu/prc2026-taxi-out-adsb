"""Direct-detector values for the 2025 evaluation days, with v55's production configuration. No fit.

Munich: X-Plane-first stands, detect_onset_v2 with v54's arrival-and-stay guard, no tow guard (value = `unguarded`).
Amsterdam: the same plus the tow and false-onset guard (value = `guarded`). Both from research_300/tow_guard_test's
estimates(), which build_combined_candidate.py mirrors for 2026; main-lane rows only. The rows found here are the
detector owners in the 2025 policy (base-free, identical in every arm).
Consistency check: B1's Amsterdam July rows found here must be exactly the 584 detector rows the volume/capacity
studies excluded.
Run from the repository root:  python3 -u -B research_360_integration/prepare_detectors_2025.py
"""
import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_300 import tow_guard_test as TG  # noqa: E402
from research_300 import geometry_compare as GC  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402
from research_300.heavy_lock import heavy_work  # noqa: E402

ID = "MVT_ID_mvt"
B1 = ("2025-01-04", "2025-01-13", "2025-01-14", "2025-01-20", "2025-01-21", "2025-01-26", "2025-01-28",
      "2025-07-09", "2025-07-11", "2025-07-13", "2025-07-14", "2025-07-16", "2025-07-25", "2025-07-27")
B2 = ("2025-01-01", "2025-01-02", "2025-01-06", "2025-01-18", "2025-07-07", "2025-07-12", "2025-07-26", "2025-07-31")
EXTRA = ("2025-01-05", "2025-01-07", "2025-01-19", "2025-07-15", "2025-07-17")   # held out for every layer
DAYS = B1 + B2 + EXTRA
VALUE = {"EDDM": "unguarded", "EHAM": "guarded"}
OUT = HERE / ".cache/research_360/detector_2025.parquet"
REPORT = HERE / "research_360_integration/results/detectors_2025.json"


def main():
    if OUT.exists():                       # the replay is deterministic; reuse a completed one
        det = pd.read_parquet(OUT)
        return report(det)
    mapping = TG.OB.callsign_mapping()
    maps = {ap: (SD.stand_table(ap), GC.xplane_table(ap)[0]) for ap in VALUE}
    frames = []
    with heavy_work("research_360: 2025 detector replay (EDDM, EHAM)", poll_s=10.0):
        for day in DAYS:
            for ap, col in VALUE.items():
                est = TG.estimates(day, ap, mapping, maps[ap])
                est = est[est.lane.eq("main") & est[col].notna()]
                frames.append(pd.DataFrame({ID: est[ID].to_numpy(), "date": day, "airport": ap, "detector_taxi": est[col].to_numpy(float)}))
            print(day, flush=True)
    det = pd.concat(frames, ignore_index=True)
    if not det[ID].is_unique:
        raise SystemExit("duplicate detector rows")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    det.to_parquet(OUT, index=False)
    return report(det)


def report(det):
    b1 = pd.read_parquet(HERE / ".cache/research_320/stand/development.parquet")
    b1 = b1[b1.airport.eq("EHAM") & b1.date.str[5:7].eq("07")]      # all categories, as volume_capacity counted
    excluded = json.loads((HERE / "research_320/results/volume_capacity.json").read_text())["b1_detector_rows_excluded"]
    rep = {"days": list(DAYS), "rows_by_airport_month": {k: int(v) for k, v in (det.airport + "/" + det.date.str[5:7]).value_counts().sort_index().items()},
              "b1_eham_july_detector_rows": int(b1[ID].isin(det[ID]).sum()), "recorded_b1_exclusion": excluded}
    rep["consistent_with_recorded_exclusion"] = rep["b1_eham_july_detector_rows"] == excluded
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
