"""Build v77 = v76 plus the drift-robust C1 on the base-owned 2026 partial rows of EGLL/07 and LEMD/01
(research_590_stand_new/PREREG_drift_robust_c1.md; LEMD/07 not admitted). NM-free, no fit. No upload.

Gates: (1) EGLL/07 rows are Codex's context-table rows (the frozen extraction and context C1 itself runs on); (2) LEMD/01
rows come from extract.py and must reproduce the 2026 coverage-audit cache's category and 23 summaries exactly, their
context from the v64 recipe, which must reproduce 3,000 of Codex's context rows; (3) every target row sits exactly at
the base in v76. New value max(base + correction, 300).
Prediction fixed here before the build runs: each cell's tested gain per row times its rows; both cells are sister-month
transfers, so central 0.6, best 1.0, worst -0.2 of full transfer, against v76's central 248.22.
Run from the repository root:  python3 -u -B research_590_stand_new/build_v77.py
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_400_cells", ROOT / "research_590_stand_new", ROOT / "research_550_review"):
    sys.path.insert(0, str(p))
import build_v64 as B64  # noqa: E402
import drift_robust_c1 as DC  # noqa: E402

S, H, MS, L = DC.S, DC.H, DC.MS, DC.L
import submission as SUB  # noqa: E402

ID, TT, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 248.22
V76 = ROOT / ".cache/research_590/v76/candidate_v76.parquet"
OUT = ROOT / ".cache/research_590/v77"
REPORT = ROOT / "research_590_stand_new/results/v77_build.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows_2026():
    own = H.ownership_v62().set_index(ID).owner
    ctx = pd.read_parquet(ROOT / ".cache/codex_v56/partial_release/context.parquet")
    eg = ctx[ctx.airport.eq("EGLL") & ctx.date.str[5:7].eq("07") & ctx.category.eq("partial_ground")]
    eg = eg[own.reindex(eg[ID]).eq("base").to_numpy()].reset_index(drop=True)
    le = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / ".cache/research_590/days/LEMD_2026-01-*.parquet")))], ignore_index=True)
    le = le[le.category.eq("partial_ground") & own.reindex(le[ID]).eq("base").to_numpy()].reset_index(drop=True)
    return eg, le


def main():
    test = json.loads((ROOT / "research_590_stand_new/results/drift_robust_c1.json").read_text())
    build76 = json.loads((ROOT / "research_590_stand_new/results/v76_build.json").read_text())
    checks = {"v76_is_built_file": sha(V76) == build76["candidate_sha256"], "test_passed": test["verdict"] == "pass",
              "admitted_as_expected": sorted(test["admitted"]) == ["EGLL/07", "LEMD/01"]}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    ctx = pd.read_parquet(ROOT / ".cache/codex_v56/partial_release/context.parquet")
    probe = ctx.sample(3000, random_state=64).reset_index(drop=True)
    rebuilt = B64.context_2026(probe[[c for c in probe.columns if c not in B64.CONTEXT and not c.startswith("airport_") and c != "is_july"]])
    checks["context_recipe_reproduces_codex"] = all(np.array_equal(rebuilt[c].to_numpy(float), probe[c].to_numpy(float)) for c in B64.CONTEXT)
    eg, le = rows_2026()
    audit = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / ".cache/research_320/airport_coverage/day_2026-*.parquet")))],
                      ignore_index=True).set_index(ID)
    a = audit.loc[le[ID]]
    checks["lemd_extraction_equals_audit"] = bool((a.category.to_numpy() == le.category.to_numpy()).all() and all(
        np.array_equal(a[f].to_numpy(float), le[f].to_numpy(float), equal_nan=True) for f in MS.FEATURES))
    le = B64.context_2026(le)
    frame = pd.concat([eg, le[eg.columns.intersection(le.columns)]], ignore_index=True)
    checks["rows"] = {"EGLL/07": int(len(eg)), "LEMD/01": int(len(le))}
    checks["rows_as_tested"] = checks["rows"]["EGLL/07"] == test["cells"]["EGLL/07"]["rows_2026"] and checks["rows"]["LEMD/01"] == test["cells"]["LEMD/01"]["rows_2026"]
    base = H.current_base_2026()
    b = frame.comparator.to_numpy(float)
    checks["comparator_equals_pre_nm_base"] = bool(np.array_equal(b, base.loc[frame[ID]].to_numpy(float)))
    v76 = pd.read_parquet(V76)
    at76 = v76.set_index(ID)[TT]
    checks["rows_at_base_in_v76"] = bool(np.array_equal(b.astype(v76[TT].dtype), at76.loc[frame[ID]].to_numpy()))
    if not (checks["context_recipe_reproduces_codex"] and checks["lemd_extraction_equals_audit"] and checks["rows_as_tested"]
            and checks["comparator_equals_pre_nm_base"] and checks["rows_at_base_in_v76"]):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2, default=str) + "\n")
        raise SystemExit(f"a gate failed: {checks}")
    cols = test["inputs"]
    ms = DC.models(None, cols)
    x = MS.inputs(frame, "real")[cols]
    corr = np.mean([m.predict(x) for m in ms], axis=0)
    rev = np.mean([m.predict(x.iloc[::-1]) for m in ms], axis=0)[::-1]
    checks["reverse_replay"] = bool(np.array_equal(corr, rev))
    new = pd.Series(np.maximum(b + corr, 300.0), index=frame[ID].to_numpy())
    cand = v76.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, TT] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    cand[TT] = cand[TT].astype(template[TT].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v76.loc[~hit].reset_index(drop=True))
    d = new.to_numpy() - at76.loc[frame[ID]].to_numpy(float)
    cell = (frame.airport + "/" + frame.date.str[5:7]).to_numpy()
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v76=bool(untouched), all_finite=bool(np.isfinite(cand[TT]).all()),
                  rows_changed=int((cand[TT].to_numpy() != v76[TT].to_numpy()).sum()),
                  change_by_cell={c: {"mean": float(d[cell == c].mean()), "rms": float(np.sqrt(np.mean(d[cell == c] ** 2)))} for c in ("EGLL/07", "LEMD/01")})
    gain = {"EGLL/07": test["cells"]["EGLL/07"]["test_EGLL/01"]["gain_mse"], "LEMD/01": test["cells"]["LEMD/01"]["test_LEMD/07_vs_base"]["gain_mse"]}
    mse = sum(gain[c] * checks["rows"][c] for c in gain) / N
    at = lambda k: round(float(np.sqrt(REF ** 2 - k * mse)), 2)
    report = {"checks": checks, "base": {"version": "v76", "central": REF, "sha256": build76["candidate_sha256"]},
              "prediction_fixed_before_upload": {"assumptions": "tested gain per row x rows per cell; sister months, central 0.6, best 1.0, worst -0.2",
                                                 "mse_per_scored_row_full": mse, "central": at(0.6), "best": at(1.0), "worst": at(-0.2)}}
    ok = not errors and untouched and checks["all_finite"] and checks["reverse_replay"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v77.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
