"""Build v58 = the scored v57 with v56's partial-track rows at s = 1.0 (research_320/PREREG_shrink.md, component A
passed; component B failed, so v57's stand rows keep 0.5). No upload.

The rows are exactly those of Codex's release overlay (.cache/codex_v56/partial_release_v55/updates.parquet, the 66,292
eligible rows with their raw correction). Before anything changes, every correction is recomputed with the rebuilt
real models (byte-identical to Codex's) on his 2026 context and must equal his file, and comparator + 0.5 x correction,
floored at 300, must reproduce the scored v57 value on every row. New value: max(comparator + 1.0 x correction, 300).
Run from the repository root:  python3 -u -B research_320/build_shrink_candidate.py
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import stand_experiment as S  # noqa: E402

MS, R = S.MS, S.R
import submission as SUB  # noqa: E402
ID = "MVT_ID_mvt"
C = HERE / ".cache/codex_v56"
REBUILD = HERE / ".cache/research_320/v56_rebuild"
V57 = HERE / ".cache/research_320/stand_candidate/candidate_v57_stand.parquet"
OUT = HERE / ".cache/research_320/shrink_candidate"
REPORT = HERE / "research_320/results/shrink_candidate.json"
V57_SCORE = 260.9292


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rec57 = json.loads((HERE / "results/upbeat-goblin_v57.json").read_text())
    conf = json.loads((HERE / "research_320/results/shrink_confirm.json").read_text())
    checks = {"v57_is_scored_file": sha(V57) == rec57["file_sha256"],
              "component_A_passed": conf["A_v56_partial"]["verdict"] == "pass",
              "component_B_failed_so_unchanged": conf["B_v57_stand"]["verdict"] == "fail"}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    v57 = pd.read_parquet(V57)
    upd = pd.read_parquet(C / "partial_release_v55/updates.parquet")
    ctx = pd.read_parquet(C / "partial_release/context.parquet").set_index(ID).loc[upd[ID]]
    x = ctx[list(MS.COLS)]
    corr = np.mean([R.joblib.load(REBUILD / f"real_{s}.pkl").predict(x) for s in S.SEEDS], axis=0)
    comp = ctx.comparator.to_numpy(float)
    checks["corrections_equal_codex"] = bool(np.array_equal(corr, upd.residual.to_numpy(float)))
    checks["max_abs_correction_difference"] = float(np.max(np.abs(corr - upd.residual.to_numpy(float))))
    at_half = np.maximum(comp + 0.5 * corr, 300.0)
    cur = v57.set_index(ID).loc[upd[ID], "TAXITIME_SEC_mvt"].to_numpy()
    checks["half_weight_reproduces_v57"] = bool(np.array_equal(at_half.astype(cur.dtype), cur))
    if not (checks["corrections_equal_codex"] or checks["max_abs_correction_difference"] < 1e-9) or not checks["half_weight_reproduces_v57"]:
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2) + "\n")
        raise SystemExit(f"the overlay does not replay: {checks}")
    new = np.maximum(comp + 1.0 * corr, 300.0)
    cand = v57.copy()
    update = pd.Series(new, index=upd[ID].to_numpy())
    hit = cand[ID].isin(update.index).to_numpy()
    cand.loc[hit, "TAXITIME_SEC_mvt"] = cand.loc[hit, ID].map(update).to_numpy(float)
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    cand["TAXITIME_SEC_mvt"] = cand.TAXITIME_SEC_mvt.astype(template.TAXITIME_SEC_mvt.dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v57.loc[~hit].reset_index(drop=True))
    changed = int((cand.TAXITIME_SEC_mvt.to_numpy() != v57.TAXITIME_SEC_mvt.to_numpy()).sum())
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v57=bool(untouched),
                  all_finite=bool(np.isfinite(cand.TAXITIME_SEC_mvt).all()), overlay_rows=int(hit.sum()), rows_changed=changed)
    ok = not errors and untouched and checks["all_finite"]
    mse = conf["planning"]["mse_per_scored_row"]
    plan = {"mse_per_scored_row": mse, "full_s": conf["planning"]["full_s"], "half_s": conf["planning"]["half_s"],
            "prediction_fixed_before_upload": {"full_transfer": round(V57_SCORE - conf["planning"]["full_s"], 2),
                                               "half_transfer": round(V57_SCORE - conf["planning"]["half_s"], 2),
                                               "central": round(V57_SCORE - (conf["planning"]["full_s"] + conf["planning"]["half_s"]) / 2, 2)}}
    shift = (cand.TAXITIME_SEC_mvt.to_numpy(float) - v57.TAXITIME_SEC_mvt.to_numpy(float))[hit]
    report = {"checks": checks, "planning": plan, "base": {"version": "v57", "score": V57_SCORE, "sha256": rec57["file_sha256"]},
              "shift_vs_v57_s": {q: float(np.percentile(shift, p)) for q, p in (("p10", 10), ("median", 50), ("p90", 90))}}
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v58_shrink.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
