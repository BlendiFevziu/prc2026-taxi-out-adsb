"""Build v74 = v73 plus the Amsterdam and Munich detectors on the 2026 fallback rows they cover
(research_550_review/PREREG_fallback_detector.md). NM-free, no fit. No upload.

Per scored day and airport, exactly the matching and estimation of fallback_detector.day_rows, applied to the scored
departures of ranking.parquet (CALLSIGN_flt joined for the claimed set). Every changed row must be a fallback-owned row
(v62 provenance) whose v73 value is untouched since v67 at its airport. New value max(MVT - onset, 300).
Prediction fixed here before the build runs: the test's gain per covered row per airport times the rows found, central
0.8, worst 0.3, best 1.2 of full transfer, against v73's central 252.76.
Run from the repository root:  python3 -u -B research_550_review/build_v74_fallback.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_550_review"):
    sys.path.insert(0, str(p))
import fallback_detector as FD  # noqa: E402
from research_290 import scored_coverage_2026 as SC  # noqa: E402
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402

AT, GC, SD, MD = FD.AT, FD.GC, FD.SD, FD.MD
ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 252.76
V73 = ROOT / ".cache/research_550/v73/candidate_v73.parquet"
OUT = ROOT / ".cache/research_550/v74"
REPORT = ROOT / "research_550_review/results/v74_build.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scored_days():
    f = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["PHASE_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"])
    f = f[f.PHASE_mvt.eq("DEP") & f.TAXITIME_SEC_mvt.isna()]
    return sorted(pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").unique())


def day_values(date, ap, mapping, maps, filed):
    t = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet")
    t = t[t.airport.astype(str).eq(ap)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
    if not len(t):
        return {}
    dep = SC.scored_departures(date)
    dep = dep[dep.airport.eq(ap)].merge(filed, on=ID, how="left").reset_index(drop=True)
    visits = AT.OB.visits(t)
    if not len(visits):
        return {}
    allm = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "plus_callsign_flt", mapping)), on=[ID, "airport"],
                     how="inner", suffixes=("", "_v"))
    claimed_all = set(zip(allm.hex, allm.t_first))
    own = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "flight_mvt", mapping)), on=[ID, "airport"],
                    how="inner", suffixes=("", "_v"))
    own = own[own.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)].set_index(ID)
    out = {}
    for r in dep[dep.lane.eq("fallback")].itertuples(index=False):
        vertices, _ = AT.lookup("xplane_first", *maps, r.STAND_mvt)
        if vertices is None:
            continue
        est = None
        if r.MVT_ID_mvt in own.index:
            v = own.loc[r.MVT_ID_mvt]
            est = FD.onset(t, v.hex, v.t_first, v.t_last, vertices, r.mvt_epoch, FD.TOW[ap])
        else:
            kept = [e for v in visits[(visits.t_last - r.mvt_epoch).abs().le(300)].itertuples(index=False)
                    if (v.hex, v.t_first) not in claimed_all
                    for e in [FD.onset(t, v.hex, v.t_first, v.t_last, vertices, r.mvt_epoch, FD.TOW[ap])] if e is not None]
            if len(kept) == 1:
                est = kept[0]
        if est is not None:
            out[r.MVT_ID_mvt] = max(r.mvt_epoch - est, 300.0)
    return out


def main():
    build73 = json.loads((ROOT / "research_550_review/results/v73_build.json").read_text())
    test = json.loads((ROOT / "research_550_review/results/fallback_detector.json").read_text())
    checks = {"v73_is_built_file": sha(V73) == build73["candidate_sha256"], "test_passed": test["verdict"] == "pass"}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    mapping = AT.OB.callsign_mapping()
    filed = pd.read_parquet(ROOT / "data/ranking.parquet", columns=[ID, "CALLSIGN_flt"])
    values = {}
    found = {}
    by_airport = {}
    for ap in ("EHAM", "EDDM"):
        maps = (SD.stand_table(ap), GC.xplane_table(ap)[0])
        before = len(values)
        for d in scored_days():
            values.update(day_values(d, ap, mapping, maps, filed))
        found[ap] = len(values) - before
        by_airport[ap] = set(list(values)[before:])
        print(ap, found[ap], flush=True)
    own = H.ownership_v62().set_index(ID).owner
    checks.update(rows_found=found, all_fallback_owned=bool(own.reindex(list(values)).eq("fallback").all()))
    if not checks["all_fallback_owned"]:
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2, default=str) + "\n")
        raise SystemExit(f"a gate failed: {checks}")
    v73 = pd.read_parquet(V73)
    s = pd.Series(values)
    diff = s - v73.set_index(ID)[T].loc[s.index].to_numpy(float)
    supported = (diff > -600.0) & (diff <= 3600.0)            # addendum A: the disagreement range the 2025 test covers
    checks["outside_support_kept_at_v73"] = int((~supported).sum())
    s = s[supported]
    found = {ap: sum(1 for m in s.index if m in by_airport[ap]) for ap in by_airport}
    cand = v73.copy()
    hit = cand[ID].isin(s.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(s).to_numpy(float)
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v73.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v73.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v73=bool(untouched),
                  all_finite=bool(np.isfinite(cand[T]).all()), rows_changed=int(hit.sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2))), "max_abs": float(np.abs(change).max())})
    mse = sum(found[ap] * test["airports"][ap]["gain_per_covered_row"] for ap in found) / N
    at = lambda k: round(float(np.sqrt(REF ** 2 - k * mse)), 2) if mse > 0 else REF
    report = {"checks": checks, "base": {"version": "v73", "sha256": build73["candidate_sha256"], "central": REF},
              "prediction_fixed_before_upload": {"assumptions": "test gain per covered row per airport x rows found; central 0.8, worst 0.3, best 1.2",
                                                 "mse_per_scored_row_full": mse, "central": at(0.8), "worst": at(0.3), "best": at(1.2)}}
    ok = not errors and untouched and checks["all_finite"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v74.parquet"
        cand.to_parquet(target, index=False)
        s.rename("taxi").rename_axis(ID).reset_index().to_parquet(OUT / "fallback_detector_rows.parquet", index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
