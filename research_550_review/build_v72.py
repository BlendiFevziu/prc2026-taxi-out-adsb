"""Build v72 = v71 with C1+ (the C1 recipe trained on E, B1 and B2) in place of C1 on C1's rows in its eight evaluated
cells (research_550_review/PREREG_c1_more_data.md passed). NM-free, no fit (models saved by c1_more_data.py). No upload.

Rows: owner "partial" in the v62 provenance (EDDF/01, EGLL/01, EHAM/07, LEBL/01, LEBL/07, LSZH/01, LSZH/07) plus the
base-owned EDDF July partial rows switched on in v63. Every row must first reproduce its v71 value as
max(comparator + 1.0 x C1, 300) from Codex's 2026 context table. New value max(comparator + 1.0 x C1+, 300).
Prediction fixed here before the build runs: the test's gain per row (+1,163.68) times the rows, central 0.8 of full
transfer, worst 0.3, best 1.2, against v71's central 253.36.
Run from the repository root:  python3 -u -B research_550_review/build_v72.py
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 253.36
GAIN_PER_ROW = 1163.681648199804
V71 = HERE / ".cache/research_550/v71/candidate_v71.parquet"
OUT = HERE / ".cache/research_550/v72"
REPORT = HERE / "research_550_review/results/v72_build.json"
C1PLUS = HERE / ".cache/research_550/c1_more/c1plus_{seed}.pkl"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def c1_rows():
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet")
    ctx["cell"] = ctx.airport + "/" + ctx.date.str[5:7]
    own = H.ownership_v62().set_index(ID).owner
    ctx["owner"] = own.reindex(ctx[ID]).to_numpy()
    part = ctx.category.eq("partial_ground")
    keep = part & (ctx.owner.eq("partial") | (ctx.cell.eq("EDDF/07") & ctx.owner.eq("base")))
    return ctx[keep].reset_index(drop=True)


def correction(frame, models):
    x = L.inputs("partial", frame)
    c = np.mean([m.predict(x) for m in models], axis=0)
    rev = np.mean([m.predict(x.iloc[::-1]) for m in models], axis=0)[::-1]
    return c, bool(np.array_equal(c, rev))


def main():
    build71 = json.loads((HERE / "research_550_review/results/v71_build.json").read_text())
    test = json.loads((HERE / "research_550_review/results/c1_more_data.json").read_text())
    checks = {"v71_is_built_file": sha(V71) == build71["candidate_sha256"], "test_passed": test["verdict"] == "pass",
              "gain_as_fixed": abs(test["gain_mse"] - GAIN_PER_ROW) < 1e-6}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    f = c1_rows()
    checks.update(rows=int(len(f)), rows_as_planned=len(f) == 84970, rows_by_cell=f.cell.value_counts().to_dict())
    base = H.current_base_2026()
    checks["comparator_equals_pre_nm_base"] = bool(np.array_equal(f.comparator.to_numpy(float), base.loc[f[ID]].to_numpy(float)))
    v71 = pd.read_parquet(V71)
    at71 = v71.set_index(ID)[T]
    c_old, rep_old = correction(f, L.load_models("partial"))
    c_new, rep_new = correction(f, [joblib.load(str(C1PLUS).format(seed=s)) for s in L.SEEDS])
    b = f.comparator.to_numpy(float)
    checks["reverse_replay"] = rep_old and rep_new
    checks["rows_reproduce_v71_with_c1"] = int((np.maximum(b + c_old, 300.0).astype(v71[T].dtype) == at71.loc[f[ID]].to_numpy()).sum())
    checks["all_reproduce"] = checks["rows_reproduce_v71_with_c1"] == len(f)
    if not (checks["all_reproduce"] and checks["reverse_replay"] and checks["comparator_equals_pre_nm_base"] and checks["rows_as_planned"]):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2, default=str) + "\n")
        raise SystemExit(f"a replay gate failed: {checks}")
    new = pd.Series(np.maximum(b + c_new, 300.0), index=f[ID].to_numpy())
    cand = v71.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v71.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v71.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v71=bool(untouched),
                  all_finite=bool(np.isfinite(cand[T]).all()), rows_changed=int((cand[T].to_numpy() != v71[T].to_numpy()).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2)))})
    mse = len(f) * GAIN_PER_ROW / N
    at = lambda k: round(float(np.sqrt(REF ** 2 - k * mse)), 2)
    report = {"checks": checks, "base": {"version": "v71", "sha256": build71["candidate_sha256"], "central": REF},
              "prediction_fixed_before_upload": {"assumptions": "test gain per row x rows; central 0.8, worst 0.3, best 1.2 of full transfer",
                                                 "mse_per_scored_row_full": mse, "central": at(0.8), "worst": at(0.3), "best": at(1.2)}}
    ok = not errors and untouched and checks["all_finite"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v72.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
