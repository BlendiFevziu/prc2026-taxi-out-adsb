"""Build v74 = v73 plus the Amsterdam detector on the scored main-lane rows only the stand-parked free match finds
(research_550_review/PREREG_eham_free_match.md). NM-free, no fit. No upload.

Per scored day: research_300/build_combined_candidate.estimates (the deployed EHAM configuration: X-Plane-first stands,
guarded onset, tow guard) must reproduce exactly the 4,188 detector-owned EHAM rows and their v73 values; that proves
the tracks, code and matching equal what shipped. Then the free match (research_550_review/eham_free_match.free_only
logic, restricted to scored main rows the callsign matcher leaves) gives the new rows; value max(MVT - onset, 300).
Prediction fixed here before the build runs: the test's gain per row times the rows actually found, central 0.8,
worst 0.3, best 1.2 of full transfer, against v73's central 252.76.
Run from the repository root:  python3 -u -B research_550_review/build_v74.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_550_review", ROOT / "research_300"):
    sys.path.insert(0, str(p))
from research_300 import build_combined_candidate as BC  # noqa: E402
import eham_free_guard_probe as P  # noqa: E402
import harness as H  # noqa: E402

AT, GC, SD, MD, SC, SUB = BC.AT, BC.GC, BC.SD, BC.MD, BC.SC, BC.SUB
ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 252.76
V73 = ROOT / ".cache/research_550/v73/candidate_v73.parquet"
OUT = ROOT / ".cache/research_550/v74"
REPORT = ROOT / "research_550_review/results/v74_build.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def free_rows(tracks, date, covered, mapping, maps):
    t = tracks[tracks.airport.astype(str).eq("EHAM")].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
    dep = SC.scored_departures(date)
    dep = dep[dep.airport.eq("EHAM") & dep.lane.eq("main")].reset_index(drop=True)
    visits = AT.OB.visits(t) if len(t) else pd.DataFrame()
    if not len(visits):
        return {}
    j = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "flight_mvt", mapping)), on=["MVT_ID_mvt", "airport"],
                  how="inner", suffixes=("", "_v"))
    j = j[j.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
    claimed = set(zip(j.hex, j.t_first))
    free = {}
    for r in dep[~dep.MVT_ID_mvt.isin(set(j.MVT_ID_mvt))].itertuples(index=False):
        vertices, _ = AT.lookup("xplane_first", *maps, r.STAND_mvt)
        if vertices is None:
            continue
        kept = []
        for v in visits[(visits.t_last - r.mvt_epoch).abs().le(300)].itertuples(index=False):
            if (v.hex, v.t_first) in claimed:
                continue
            est = P.estimate(t, v.hex, v.t_first, v.t_last, vertices, r.mvt_epoch)
            if est is not None:
                kept.append((v.hex, v.t_first, r.mvt_epoch - est))
        if len(kept) == 1:
            free[r.MVT_ID_mvt] = kept[0]
    keys = pd.Series([f"{h}|{f}" for h, f, _ in free.values()]).value_counts()
    return {m: taxi for m, (h, f, taxi) in free.items() if keys[f"{h}|{f}"] == 1 and m not in covered}


def main():
    build73 = json.loads((ROOT / "research_550_review/results/v73_build.json").read_text())
    test = json.loads((ROOT / "research_550_review/results/eham_free_match.json").read_text())
    checks = {"v73_is_built_file": sha(V73) == build73["candidate_sha256"], "test_passed": test["verdict"] == "pass"}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    mapping = AT.OB.callsign_mapping()
    maps = (SD.stand_table("EHAM"), GC.xplane_table("EHAM")[0])
    own = H.ownership_v62()
    owned = set(own.loc[own.owner.eq("direct_detector") & own.airport.eq("EHAM"), ID])
    v73 = pd.read_parquet(V73)
    at73 = v73.set_index(ID)[T]
    replay, new = {}, {}
    for date in BC.scored_days():
        tracks = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet")
        est, _ = BC.estimates(tracks, date, "EHAM", "xplane_first", True, mapping, maps)
        replay.update(est)
        new.update(free_rows(tracks, date, set(est), mapping, maps))
        print(date, len(replay), len(new), flush=True)
    checks["replay_rows"] = len(replay)
    checks["replay_equals_owned_rows"] = set(replay) == owned
    rep_ids = [m for m in replay if m in owned]
    checks["replay_values_equal_v73"] = bool(np.array_equal(np.array([replay[m] for m in rep_ids]).astype(v73[T].dtype),
                                                             at73.loc[rep_ids].to_numpy()))
    checks["new_rows"] = len(new)
    checks["new_rows_disjoint_from_detector"] = not (set(new) & owned)
    if not (checks["replay_equals_owned_rows"] and checks["replay_values_equal_v73"] and checks["new_rows_disjoint_from_detector"]):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2, default=str) + "\n")
        raise SystemExit(f"a replay gate failed: {checks}")
    values = pd.Series({m: max(v, 300.0) for m, v in new.items()})
    cand = v73.copy()
    hit = cand[ID].isin(values.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(values).to_numpy(float)
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v73.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v73.loc[hit, T].to_numpy(float)
    owners = own.set_index(ID).owner.reindex(values.index).value_counts().to_dict()
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v73=bool(untouched),
                  all_finite=bool(np.isfinite(cand[T]).all()), rows_changed=int(hit.sum()), new_rows_previous_owner=owners,
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2)))})
    mse = len(values) * test["gain_mse"] / N
    at = lambda k: round(float(np.sqrt(REF ** 2 - k * mse)), 2) if mse > 0 else REF
    report = {"checks": checks, "base": {"version": "v73", "sha256": build73["candidate_sha256"], "central": REF},
              "prediction_fixed_before_upload": {"assumptions": "test gain per row x rows found; central 0.8, worst 0.3, best 1.2",
                                                 "mse_per_scored_row_full": mse, "central": at(0.8), "worst": at(0.3), "best": at(1.2)}}
    ok = not errors and untouched and checks["all_finite"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v74.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
        pd.DataFrame({ID: values.index, "taxi": values.to_numpy()}).to_parquet(OUT / "free_rows_v74.parquet", index=False)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
