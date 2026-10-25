"""Build the Munich-only stand-detector candidate on top of v52. Nothing is uploaded.

Recipe. v52 everywhere, except scored MAIN-lane departures at Munich where the frozen v2 detector
(stand_detector.detect_onset_v2) gives an estimate: there the prediction becomes the detector's taxi time,
MVT_TIME - estimate, unfloored, exactly as validated. Fallback rows are never touched (not validated).

One post-hoc guard, disclosed (research_290/stay_guard_check.py, 25 September 2026): an estimate is withdrawn, and the
row keeps v52's prediction, when the aircraft moved faster than FAST_KT within 30 minutes before the chosen transition
AND was still at its stand more than 20 minutes after it. That is docking after a landing, not a push-back. It was
found by inspecting the first build (one row, 29 July 2026, +3,264 s, 148 kt before and 61 minutes at the stand
after); on all 13 labelled Munich days (1,484 estimates) it withdraws nothing, so no validation result changes.

Evidence required, checked as gates:
  G1  base parquet hash matches the v52 manifest (candidate_sha256)
  G2  research_290/stand_detector.py is unchanged in the working tree (the validated v2 code)
  G3  the production-strength test at Munich passed: results/munich_validate_jan2025.json, all four checks
  G4  submission.pruefe passes; every row outside the covered Munich main-lane rows is byte-identical to v52;
      all finite
  G5  the guard withdraws no estimate on the labelled days (results/stay_guard_check.json)
A real build also needs the ADS-B tracks of ALL 62 scored days; --dry reports what the days present would change,
label-free, and writes no candidate.

Row-level outputs go to .cache/munich_candidate/ (never committed); the manifest to research_290/results/.
Run from the repository root:  python3 -u -B research_290/build_munich_candidate.py [--dry]
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADSB = Path("adsb-lol")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ADSB))
import offblock_from_tracks as OB  # noqa: E402
import submission as SUB  # noqa: E402
from research_290 import munich_dev as MD  # noqa: E402
from research_290 import scored_coverage_2026 as SC  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402

AIRPORT = "EDDM"
BASE = ROOT / ".cache/research_280/release/candidate_v52_combined.parquet"
OUT = ROOT / ".cache/munich_candidate"
MANIFEST = ROOT / "research_290/results/munich_candidate_manifest.json"
GUARD_WINDOW_S, GUARD_STAY_S = 1800.0, 1200.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scored_days():
    frame = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["PHASE_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"])
    frame = frame[frame.PHASE_mvt.eq("DEP") & frame.TAXITIME_SEC_mvt.isna()]
    return sorted(pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").unique())


def estimates(date, mapping, table):
    """Detector taxi times for the scored Munich main-lane departures of one day: MVT_ID_mvt -> seconds."""
    tracks = pd.read_parquet(ADSB / f"tracks/{date}.parquet")
    t = tracks[tracks.airport.astype(str).eq(AIRPORT)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
    dep = SC.scored_departures(date)
    dep = dep[dep.airport.eq(AIRPORT)]
    joined = dep.merge(OB.match(dep, OB.visits(t), mapping), on=["MVT_ID_mvt", "airport"], how="inner",
                       suffixes=("", "_v"))
    joined = joined[joined.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
    out, withdrawn = {}, 0
    for row in joined[joined.lane.eq("main")].itertuples(index=False):
        vertices, _ = SD.lookup(table, row.STAND_mvt)
        if vertices is None:
            continue
        ts, lat, lon, gs, _ = MD.points_of(t, row)
        det = SD.detect_onset_v2(ts, lat, lon, gs, vertices, row.mvt_epoch, SD.R_PARK_M)
        if det["status"] != "ok":
            continue
        at = (ts >= det["t_after"]) & (SD._metres(lat, lon, vertices) <= SD.R_STAND_M)
        stay = ts[at].max() - det["t_after"] if at.any() else 0.0
        before = (ts >= det["t_before"] - GUARD_WINDOW_S) & (ts < det["t_after"]) & np.isfinite(gs)
        if stay > GUARD_STAY_S and before.any() and gs[before].max() > SD.FAST_KT:
            withdrawn += 1
            continue
        out[row.MVT_ID_mvt] = row.mvt_epoch - (det["t_before"] + det["t_after"]) / 2
    return len(dep), out, withdrawn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    gates = {}
    v52 = json.loads((ROOT / "results/upbeat-goblin_v52.json").read_text())
    gates["G1_base_is_v52"] = sha(BASE) == v52["candidate_sha256"]
    gates["G2_detector_unchanged"] = subprocess.run(["git", "diff", "--quiet", "--", "research_290/stand_detector.py"],
                                                    cwd=ROOT).returncode == 0
    test = ROOT / "research_290/results/munich_validate_jan2025.json"
    gates["G3_production_test_passed"] = test.exists() and bool(json.loads(test.read_text()).get("justifies_further_work"))
    guard = ROOT / "research_290/results/stay_guard_check.json"
    gates["G5_guard_neutral_on_labelled_days"] = guard.exists() and \
        json.loads(guard.read_text())["labelled"]["combined_guard_withdraws"] == 0
    print("gates", gates, flush=True)
    if not (gates["G1_base_is_v52"] and gates["G2_detector_unchanged"] and gates["G5_guard_neutral_on_labelled_days"]):
        raise SystemExit("G1, G2 or G5 failed")
    if not args.dry and not gates["G3_production_test_passed"]:
        raise SystemExit("G3: the production-strength test has not passed; only --dry is allowed")

    days = scored_days()
    present = [d for d in days if (ADSB / f"tracks/{d}.parquet").exists()]
    if not args.dry and len(present) != len(days):
        raise SystemExit(f"tracks for {len(days) - len(present)} of {len(days)} scored days are missing")
    mapping, table, taxi, per_day = OB.callsign_mapping(), SD.stand_table(AIRPORT), {}, {}
    for date in present:
        n, est, withdrawn = estimates(date, mapping, table)
        taxi.update(est)
        per_day[date] = {"munich_scored_departures": n, "replaced": len(est), "withdrawn_by_guard": withdrawn}
        print(date, per_day[date], flush=True)

    base = pd.read_parquet(BASE)
    cand = base.copy()
    hit = cand.MVT_ID_mvt.isin(taxi.keys()).to_numpy()
    new = cand.loc[hit, "MVT_ID_mvt"].map(taxi).to_numpy(float)
    shift = new - cand.loc[hit, "TAXITIME_SEC_mvt"].to_numpy(float)
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    cand.loc[hit, "TAXITIME_SEC_mvt"] = new
    cand["TAXITIME_SEC_mvt"] = cand.TAXITIME_SEC_mvt.astype(template.TAXITIME_SEC_mvt.dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = bool(cand.loc[~hit, "TAXITIME_SEC_mvt"].equals(base.loc[~hit, "TAXITIME_SEC_mvt"]))
    gates["G4"] = {"pruefe_errors": [str(e) for e in errors], "untouched_rows_identical": untouched,
                   "all_finite": bool(np.isfinite(cand.TAXITIME_SEC_mvt).all())}
    report = {"dry_run": args.dry, "gates": gates, "scored_days": len(days), "days_with_tracks": len(present),
              "rows_replaced": int(hit.sum()), "share_of_scored_rows": float(hit.mean()),
              "withdrawn_by_guard": int(sum(v["withdrawn_by_guard"] for v in per_day.values())),
              "shift_s": {"median": float(np.median(shift)) if len(shift) else None,
                          "p10": float(np.percentile(shift, 10)) if len(shift) else None,
                          "p90": float(np.percentile(shift, 90)) if len(shift) else None},
              "per_day": per_day}
    ok = not errors and untouched and gates["G4"]["all_finite"]
    if not args.dry:
        if not ok:
            raise SystemExit("GATE G4 failed")
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v52_munich_detector.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_sha256"] = sha(target)
    MANIFEST.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_day"}, indent=1))


if __name__ == "__main__":
    main()
