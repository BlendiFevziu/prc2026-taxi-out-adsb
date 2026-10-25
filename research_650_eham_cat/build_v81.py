"""Build v81 = v80 with CatBoost averaged into the Amsterdam January add-on on its 9,752 rows, as admitted by
research_650_eham_cat/PREREG_eham_cat.md. NM-free, no fit. No upload.

Rows exactly as build_v80.eham_rows. Gate: every row reproduces its v80 value as max(base + (add-on + C1+) / 2, 300).
New value max(base + ((add-on + CatBoost) / 2 + C1+) / 2, 300). Prediction fixed here before the build runs: the fresh
gain per row times 9,752, central 0.8, worst 0.3, best 1.2 of full transfer, against v80's official 245.046.
Run from the repository root:  python3 -u -B research_650_eham_cat/build_v81.py
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research_640_tail_rule"))
import build_v80 as B80  # noqa: E402

L, H, SUB = B80.L, B80.H, B80.SUB
ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 245.046
V80 = ROOT / ".cache/research_640/v80/candidate_v80.parquet"
OUT = ROOT / ".cache/research_650/v81"
REPORT = ROOT / "research_650_eham_cat/results/v81_build.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    test = json.loads((ROOT / "research_650_eham_cat/results/eham_cat.json").read_text())
    rec80 = json.loads((ROOT / "results/upbeat-goblin_v80.json").read_text())
    checks = {"v80_is_uploaded_file": sha(V80) == rec80["file_sha256"], "test_passed": test["verdict"] == "pass"}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    v80 = pd.read_parquet(V80)
    at80 = v80.set_index(ID)[T]
    base = H.current_base_2026()
    eh = B80.eham_rows()
    b = eh.comparator.to_numpy(float)
    xa = L.eham_inputs(eh)
    addon = L.correction("eham_january", eh)
    cat = np.mean([joblib.load(ROOT / f".cache/research_650/cat_eham_{s}.pkl").predict(xa) for s in L.SEEDS], axis=0)
    plus = B80.mean_pred(".cache/research_550/c1_more/c1plus_{seed}.pkl", L.inputs("partial", eh))
    checks["rows"] = int(len(eh))
    checks["comparator_equals_base"] = bool(np.array_equal(b, base.loc[eh[ID]].to_numpy(float)))
    checks["reproduce_v80"] = int((np.maximum(b + (addon + plus) / 2, 300.0).astype(v80[T].dtype) == at80.loc[eh[ID]].to_numpy()).sum())
    if not (checks["rows"] == 9752 and checks["comparator_equals_base"] and checks["reproduce_v80"] == 9752):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2, default=str) + "\n")
        raise SystemExit(f"gate failed: {checks}")
    new = pd.Series(np.maximum(b + ((addon + cat) / 2 + plus) / 2, 300.0), index=eh[ID].to_numpy())
    cand = v80.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v80.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v80.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v80=bool(untouched), all_finite=bool(np.isfinite(cand[T]).all()),
                  rows_changed=int((cand[T].to_numpy() != v80[T].to_numpy()).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2)))})
    mse = 9752 * test["fresh"]["gain_mse"] / N
    at = lambda k: round(float(np.sqrt(REF ** 2 - k * mse)), 3)
    report = {"checks": checks, "base": {"version": "v80", "score": REF, "sha256": rec80["file_sha256"]},
              "prediction_fixed_before_upload": {"assumptions": "fresh gain per row x 9,752; central 0.8, worst 0.3, best 1.2",
                                                 "mse_per_scored_row_full": mse, "central": at(0.8), "worst": at(0.3), "best": at(1.2)}}
    ok = not errors and untouched and checks["all_finite"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v81.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
