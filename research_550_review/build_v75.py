"""Build v75 = v74 plus the v2 detector (OSM stands, arrival guard, no tow guard) on the 2026 fallback rows it covers at
LEBL and LSZH, within the support range (-600 s, +3,600 s] of the disagreement with v74
(research_550_review/PREREG_fallback_detector_more.md passed for LEBL and LSZH; EDDF failed). NM-free. No upload.
Prediction fixed here before the build runs: each airport's gain per covered row times its rows applied, central 0.8,
worst 0.3, best 1.2 of full transfer, against v74's central 252.58.
Run from the repository root:  python3 -u -B research_550_review/build_v75.py
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
import build_v74_fallback as B  # noqa: E402

FD, SUB, H = B.FD, B.SUB, B.H
ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 252.58
AIRPORTS = ("LEBL", "LSZH")
V74 = ROOT / ".cache/research_550/v74/candidate_v74.parquet"
OUT = ROOT / ".cache/research_550/v75"
REPORT = ROOT / "research_550_review/results/v75_build.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def values_2026():
    for ap in AIRPORTS:
        FD.TOW[ap] = False
    mapping = FD.AT.OB.callsign_mapping()
    filed = pd.read_parquet(ROOT / "data/ranking.parquet", columns=[ID, "CALLSIGN_flt"])
    out = {}
    for ap in AIRPORTS:
        maps = (FD.SD.stand_table(ap), {})
        vals = {}
        for d in B.scored_days():
            vals.update(B.day_values(d, ap, mapping, maps, filed))
        out[ap] = vals
        print(ap, len(vals), flush=True)
    return out


def main():
    build74 = json.loads((ROOT / "research_550_review/results/v74_build.json").read_text())
    test = json.loads((ROOT / "research_550_review/results/fallback_detector_more.json").read_text())
    checks = {"v74_is_built_file": sha(V74) == build74["candidate_sha256"],
              "test_passed_for_these_airports": sorted(test["passing_airports"]) == sorted(AIRPORTS)}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    per_ap = values_2026()
    own = H.ownership_v62().set_index(ID).owner
    v74 = pd.read_parquet(V74)
    at74 = v74.set_index(ID)[T]
    keep, applied = {}, {}
    for ap, vals in per_ap.items():
        s = pd.Series(vals, dtype=float)
        if len(s) and not own.reindex(s.index).eq("fallback").all():
            raise SystemExit(f"{ap}: a covered row is not fallback-owned")
        diff = s - at74.loc[s.index].to_numpy(float) if len(s) else s
        s = s[(diff > -600.0) & (diff <= 3600.0)] if len(s) else s
        keep.update(s.to_dict())
        applied[ap] = {"covered": len(vals), "applied": int(len(s))}
    checks["per_airport"] = applied
    new = pd.Series(keep)
    cand = v74.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v74.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v74.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v74=bool(untouched),
                  all_finite=bool(np.isfinite(cand[T]).all()), rows_changed=int(hit.sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2))), "max_abs": float(np.abs(change).max())})
    mse = sum(applied[ap]["applied"] * test["airports"][ap]["gain_mse"] for ap in AIRPORTS) / N
    at = lambda k: round(float(np.sqrt(REF ** 2 - k * mse)), 2)
    report = {"checks": checks, "base": {"version": "v74", "sha256": build74["candidate_sha256"], "central": REF},
              "prediction_fixed_before_upload": {"assumptions": "gain per covered row per airport x rows applied; central 0.8, worst 0.3, best 1.2",
                                                 "mse_per_scored_row_full": mse, "central": at(0.8), "worst": at(0.3), "best": at(1.2)}}
    ok = not errors and untouched and checks["all_finite"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v75.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
