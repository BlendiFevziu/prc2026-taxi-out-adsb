"""POST-HOC check of a stay guard for detector v2, found while inspecting the built candidate. Label-free on scored rows.

Found 25 September 2026: of the nine candidate rows whose prediction moved by more than 20 minutes, one (29 July 2026,
+3,264 s) was an arrival transition the 600 s rule missed: the aircraft had been at 148 kt within the previous 30
minutes and then stayed at the stand for 61 minutes. A push-back is followed by leaving the stand within minutes.

Guard: withdraw an estimate when the aircraft is still seen within R_STAND of its stand more than STAY_S after the
chosen transition. It only ever withdraws, so a withdrawn row keeps the v52 prediction.

This script measures, on every labelled Munich day already used (development 13 December, v1 dates, v2 dates, and the
January 2025 production dates), how many estimates the guard would withdraw and how wrong those were; and on the 62
scored days, how many candidate rows it would withdraw. Aggregates only.
Run from the repository root:  python3 -u -B research_290/stay_guard_check.py
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

STAY_S = 1200.0
ARRIVAL_WINDOW_S = 1800.0   # the combined guard: fast motion (> SD.FAST_KT) this long before, AND a stay over STAY_S
LABELLED = ("2025-12-13", "2025-12-03", "2025-12-10", "2025-12-17", "2025-12-20", "2025-12-05", "2025-12-08",
            "2025-12-11", "2025-12-14", "2025-01-08", "2025-01-16", "2025-01-24", "2025-01-26")
OUTPUT = ROOT / "research_290/results/stay_guard_check.json"


def stays(date, scored, mapping, table):
    tracks = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet")
    t = tracks[tracks.airport.astype(str).eq("EDDM")].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
    dep = SC.scored_departures(date) if scored else MD.departures(date, "EDDM")
    dep = dep[dep.airport.eq("EDDM")]
    j = dep.merge(OB.match(dep, OB.visits(t), mapping), on=["MVT_ID_mvt", "airport"], how="inner", suffixes=("", "_v"))
    j = j[j.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
    out = []
    for row in j[j.lane.eq("main")].itertuples(index=False):
        vertices, _ = SD.lookup(table, row.STAND_mvt)
        if vertices is None:
            continue
        ts, lat, lon, gs, _ = MD.points_of(t, row)
        o = SD.detect_onset_v2(ts, lat, lon, gs, vertices, row.mvt_epoch, SD.R_PARK_M)
        if o["status"] != "ok":
            continue
        at = (ts >= o["t_after"]) & (SD._metres(lat, lon, vertices) <= SD.R_STAND_M)
        before = (ts >= o["t_before"] - ARRIVAL_WINDOW_S) & (ts < o["t_after"]) & np.isfinite(gs)
        out.append({"stay_s": float(ts[at].max() - o["t_after"]) if at.any() else 0.0,
                    "vmax_before_kt": float(gs[before].max()) if before.any() else 0.0,
                    "taxi_s": float(row.mvt_epoch - (o["t_before"] + o["t_after"]) / 2),
                    "err_s": None if scored else float((o["t_before"] + o["t_after"]) / 2 - row.block_epoch)})
    return out


def main():
    mapping, table, report = OB.callsign_mapping(), SD.stand_table("EDDM"), {"stay_s": STAY_S}
    lab = pd.DataFrame([r for d in LABELLED for r in stays(d, False, mapping, table)])
    hit = lab.stay_s > STAY_S
    report["labelled"] = {"days": len(LABELLED), "estimates": int(len(lab)),
                          "stay_percentiles_s": dict(zip(["p50", "p90", "p99", "max"],
                                                         np.percentile(lab.stay_s, [50, 90, 99, 100]).round(0).tolist())),
                          "withdrawn": int(hit.sum()), "withdrawn_errors_s": lab.loc[hit, "err_s"].round(0).tolist(),
                          "withdrawn_within_60_s": int((lab.loc[hit, "err_s"].abs() <= 60).sum()),
                          "long_stays": lab.loc[hit, ["stay_s", "vmax_before_kt", "err_s"]].round(0).to_dict("records"),
                          "combined_guard_withdraws": int((hit & (lab.vmax_before_kt > SD.FAST_KT)).sum())}
    print("labelled", json.dumps(report["labelled"]), flush=True)
    days = SC.scored_days() if hasattr(SC, "scored_days") else None
    if days is None:
        frame = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["PHASE_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"])
        frame = frame[frame.PHASE_mvt.eq("DEP") & frame.TAXITIME_SEC_mvt.isna()]
        days = sorted(pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").unique())
    sc = pd.DataFrame([r for d in days for r in stays(d, True, mapping, table)])
    report["scored"] = {"days": len(days), "estimates": int(len(sc)),
                        "stay_percentiles_s": dict(zip(["p50", "p90", "p99", "max"],
                                                       np.percentile(sc.stay_s, [50, 90, 99, 100]).round(0).tolist())),
                        "withdrawn": int((sc.stay_s > STAY_S).sum()),
                        "long_stays": sc.loc[sc.stay_s > STAY_S, ["stay_s", "vmax_before_kt", "taxi_s"]].round(0)
                        .to_dict("records"),
                        "combined_guard_withdraws": int(((sc.stay_s > STAY_S) & (sc.vmax_before_kt > SD.FAST_KT)).sum())}
    print("scored", json.dumps(report["scored"]), flush=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
