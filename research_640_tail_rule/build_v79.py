"""Build v79 = v78 with CatBoost averaged into the drift-robust stand model (EGLL/07, LEMD/07, LEMD/01 stand rows), the
drift-robust C1 (EGLL/07, LEMD/01 partial rows) and the old C1 (EDDM/01, EDDM/07, LEMD/07 partial rows), as admitted by
research_640_tail_rule/PREREG_catboost_more.md. NM-free, no fit. No upload.

Rows exactly as their layers were built (build_v76, build_v77, build_v64/v65 via the coverage-audit cache and the v64
context recipe). Gate: every row reproduces its v78 value as max(base + model, 300). New value
max(base + (model + CatBoost) / 2, 300). Prediction fixed here before the build runs: each tested cell's gain per row
times its admitted rows; half the rows are sister-month transfers, so central 0.7, worst 0.2, best 1.1 of full
transfer, against v78's central 246.16.
Run from the repository root:  python3 -u -B research_640_tail_rule/build_v79.py
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_400_cells", ROOT / "research_590_stand_new", ROOT / "research_550_review"):
    sys.path.insert(0, str(p))
import build_v64 as B64  # noqa: E402
import build_v77 as B77  # noqa: E402
import extract as E  # noqa: E402
import drift_robust as DRS  # noqa: E402

S, MS, L, H = DRS.S, B77.MS, DRS.L, DRS.H
import submission as SUB  # noqa: E402

ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 246.16
V78 = ROOT / ".cache/research_640/v78/candidate_v78.parquet"
OUT = ROOT / ".cache/research_640/v79"
REPORT = ROOT / "research_640_tail_rule/results/v79_build.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(pattern):
    return [joblib.load(ROOT / pattern.format(seed=s)) for s in L.SEEDS]


def stand_rows():
    own = H.ownership_v62().set_index(ID).owner
    eg = E.load("EGLL", sorted(d for d in E.departures_2026("EGLL").date.unique() if d[5:7] == "07"))
    le = E.load("LEMD", sorted(E.departures_2026("LEMD").date.unique()))
    r = pd.concat([eg, le], ignore_index=True)
    r = r[r.category.eq("observed_at_assigned_stand") & own.reindex(r[ID]).eq("base").to_numpy()]
    r = r[(r.airport + "/" + r.date.str[5:7]).isin(["EGLL/07", "LEMD/07", "LEMD/01"])]
    return B64.context_2026(r.reset_index(drop=True))


def old_c1_rows():
    own = H.ownership_v62().set_index(ID).owner
    feats = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / ".cache/research_320/airport_coverage/day_2026-*.parquet")))], ignore_index=True)
    cell = feats.airport + "/" + feats.date.str[5:7]
    r = feats[cell.isin(["EDDM/01", "EDDM/07", "LEMD/07"]).to_numpy() & feats.category.eq("partial_ground").to_numpy()
              & own.reindex(feats[ID]).eq("base").to_numpy()].reset_index(drop=True)
    return B64.context_2026(r)


def main():
    test = json.loads((ROOT / "research_640_tail_rule/results/catboost_more.json").read_text())
    build78 = json.loads((ROOT / "research_640_tail_rule/results/v78_build.json").read_text())
    checks = {"v78_is_built_file": sha(V78) == build78["candidate_sha256"], "test_passed": test["verdict"] == "pass",
              "all_admitted": len(test["admitted"]) == 8}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    v78 = pd.read_parquet(V78)
    at78 = v78.set_index(ID)[T]
    st_cols = DRS.cols()
    c1_cols = json.loads((ROOT / "research_590_stand_new/results/drift_robust_c1.json").read_text())["inputs"]
    eg, le = B77.rows_2026()
    groups = {
        "dr_stand": (stand_rows(), lambda f: S.inputs(f, "real")[st_cols], ".cache/research_590/drift_robust/dr_{seed}.pkl", ".cache/research_640/cat_dr_stand_{seed}.pkl"),
        "dr_c1": (pd.concat([eg, B64.context_2026(le)[eg.columns.intersection(B64.context_2026(le).columns)]], ignore_index=True),
                  lambda f: MS.inputs(f, "real")[c1_cols], ".cache/research_590/drift_robust_c1/drc1_{seed}.pkl", ".cache/research_640/cat_dr_c1_{seed}.pkl"),
        "old_c1": (old_c1_rows(), lambda f: MS.inputs(f, "real"), ".cache/research_320/volume/models/c1_real_{seed}.pkl", ".cache/research_640/cat_c1_E_{seed}.pkl")}
    expected_rows = {"dr_stand": 10242 + 2812 + 2176, "dr_c1": 9048 + 12478, "old_c1": 5397 + 7525 + 8692}          # v64/v65 rows (detector-owned rows excluded)
    base = H.current_base_2026()
    new, gains = {}, {}
    for name, (frame, xf, model_pat, cat_pat) in groups.items():
        b = frame.comparator.to_numpy(float)
        x = xf(frame)
        m = np.mean([mm.predict(x) for mm in load(model_pat)], axis=0)
        c = np.mean([mm.predict(x) for mm in load(cat_pat)], axis=0)
        checks[f"{name}_rows"] = int(len(frame))
        checks[f"{name}_rows_as_expected"] = len(frame) == expected_rows[name]
        checks[f"{name}_comparator_equals_base"] = bool(np.array_equal(b, base.loc[frame[ID]].to_numpy(float)))
        checks[f"{name}_reproduce_v78"] = int((np.maximum(b + m, 300.0).astype(v78[T].dtype) == at78.loc[frame[ID]].to_numpy()).sum())
        if not (checks[f"{name}_rows_as_expected"] and checks[f"{name}_comparator_equals_base"] and checks[f"{name}_reproduce_v78"] == len(frame)):
            REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2, default=str) + "\n")
            raise SystemExit(f"gate failed for {name}: {checks}")
        new.update(dict(zip(frame[ID], np.maximum(b + (m + c) / 2, 300.0))))
    s = pd.Series(new)
    cand = v78.copy()
    hit = cand[ID].isin(s.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(s).to_numpy(float)
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v78.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v78.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v78=bool(untouched), all_finite=bool(np.isfinite(cand[T]).all()),
                  rows_changed=int((cand[T].to_numpy() != v78[T].to_numpy()).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2)))})
    tested = {"LEMD/07 stand": "stand LEMD/07", "LEMD/01 stand": "stand LEMD/07", "EGLL/07 stand": "stand EGLL/01", "EGLL/07 partial": "c1 EGLL/01",
              "LEMD/01 partial": "c1 LEMD/07"}
    actual = {"EDDM/01": 5397, "EDDM/07": 7525}                    # the registration counted detector rows here
    mse = 0.0
    for tgt, n in test["admitted"].items():
        src = ("A", tested[tgt]) if tgt in tested else ("B", "EDDM/01" if tgt.startswith("EDDM") else "LEMD/07")
        mse += test[src[0]][src[1]]["gain_mse"] * actual.get(tgt, n) / N
    at = lambda k: round(float(np.sqrt(REF ** 2 - k * mse)), 2)
    report = {"checks": checks, "base": {"version": "v78", "central": REF, "sha256": build78["candidate_sha256"]},
              "prediction_fixed_before_upload": {"assumptions": "tested gain per row x admitted rows; central 0.7, worst 0.2, best 1.1",
                                                 "mse_per_scored_row_full": mse, "central": at(0.7), "worst": at(0.2), "best": at(1.1)}}
    ok = not errors and untouched and checks["all_finite"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v79.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
