"""LABEL-FREE: how many scored EHAM main-lane rows would the validated guarded detector replace? No candidate is built.

The detector exactly as validated in research_300/PREREG_towguard.md (addendum A): X-Plane-first stands, FLIGHT_mvt
spellings, detect_onset_v2 plus v54's arrival-and-stay guard, plus the tow and false-onset guard. Run on the scored
ranking departures (no labels exist there) of all 62 scored days, the same rows research_290/build_munich_candidate.py
used for Munich (research_290/scored_coverage_2026.scored_departures). The tracks are those fetched overnight on 24/25
September (24 of the 62 days from prod-0 where adsb.lol prefers staging-0; not re-fetched).

What it gives: rows replaced, their share of the scored set, and a NAIVE official-scale value using the fresh-date
gain per replaced row (17,044.9 MSE, comparison 2), bracketed by Munich's realised transfer (v54 realised 1.52 times
its validated MSE per row; the lower bracket assumes 0.74, the transfer the NM add-on showed). Not a forecast.
Run from the repository root:  python3 -u -B research_300/eham_scored_reach.py
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
from research_290 import scored_coverage_2026 as SC  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402
from research_300 import arm_test as AT  # noqa: E402
from research_300 import geometry_compare as GC  # noqa: E402
from research_300 import tow_guard as TG  # noqa: E402

AIRPORT = "EHAM"
GAIN_PER_ROW = 17044.857407687694      # results/tow_guard_test.json, comparison_2_decision.gain_per_changed_row
TRANSFER = {"low_0.74": 0.74, "validated_1.0": 1.0, "munich_realised_1.52": 1.52}
OUT = ROOT / "research_300/results/eham_scored_reach.json"


def scored_days():
    frame = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["PHASE_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"])
    frame = frame[frame.PHASE_mvt.eq("DEP") & frame.TAXITIME_SEC_mvt.isna()]
    return sorted(pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").unique()), len(frame)


def main():
    days, n_scored = scored_days()
    missing = [d for d in days if not (MD.ADSB / f"tracks/{d}.parquet").exists()]
    if missing:
        raise SystemExit(f"tracks missing for {missing}")
    mapping = OB.callsign_mapping()
    maps = (SD.stand_table(AIRPORT), GC.xplane_table(AIRPORT)[0])
    per_day, statuses = {}, {}
    for date in days:
        tracks = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet")
        t = tracks[tracks.airport.astype(str).eq(AIRPORT)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
        dep = SC.scored_departures(date)
        dep = dep[dep.airport.eq(AIRPORT)].reset_index(drop=True)
        main_rows = int(dep.lane.eq("main").sum())
        replaced = unguarded = 0
        visits = OB.visits(t) if len(t) else pd.DataFrame()
        if len(visits):
            j = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "flight_mvt", mapping)),
                          on=["MVT_ID_mvt", "airport"], how="inner", suffixes=("", "_v"))
            j = j[j.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
            for row in j[j.lane.eq("main")].itertuples(index=False):
                vertices, _ = AT.lookup("xplane_first", *maps, row.STAND_mvt)
                if vertices is None:
                    continue
                ts, lat, lon, gs, ground = MD.points_of(t, row)
                o = AT.guarded(ts, lat, lon, gs, vertices, row.mvt_epoch)
                g = TG.check(ts, lat, lon, gs, ground, vertices, o, row.mvt_epoch)
                statuses[g["status"]] = statuses.get(g["status"], 0) + 1
                unguarded += o["status"] == "ok"
                replaced += g["status"] == "ok"
        per_day[date] = {"eham_main_scored": main_rows, "unguarded": unguarded, "replaced": replaced}
        print(date, per_day[date], flush=True)
    frame = pd.DataFrame(per_day).T
    replaced, eham_main = int(frame.replaced.sum()), int(frame.eham_main_scored.sum())
    official = 264.2328
    value = {k: float(official - np.sqrt(official ** 2 - replaced / n_scored * GAIN_PER_ROW * f)) for k, f in TRANSFER.items()}
    report = {"scored_days": len(days), "scored_rows": n_scored, "eham_main_scored_rows": eham_main,
              "rows_replaced": replaced, "share_of_eham_main": replaced / eham_main,
              "share_of_scored_rows": replaced / n_scored, "unguarded_estimates": int(frame.unguarded.sum()),
              "withdrawn_by_tow_guard": int(frame.unguarded.sum() - replaced),
              "january_2026_replaced": int(frame[frame.index.str[5:7] == "01"].replaced.sum()),
              "july_2026_replaced": int(frame[frame.index.str[5:7] == "07"].replaced.sum()),
              "detector_statuses": statuses, "gain_per_replaced_row_fresh_dates": GAIN_PER_ROW,
              "naive_official_gain_s": value, "per_day": per_day}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_day"}, indent=1))


if __name__ == "__main__":
    main()
