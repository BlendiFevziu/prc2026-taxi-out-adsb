"""Build v59 = the scored v58 plus Codex's EHAM January add-on at full weight (research_320/PREREG_eham_weight.md,
both comparisons passed). No upload.

Rows: exactly Codex's 8,968 eligible 2026 EHAM January rows (.cache/codex_eham/eham_on_scored_v56/updates.parquet).
Before anything changes: every correction is recomputed with his three real models (hash-checked copies) on his 2026
context and seven features and must equal his file; comparator + 0.5 x correction, floored at 300, must reproduce his
own predictions; each row's comparator must equal the scored v58 value; no row may be an NM row (v52 against v51) or a
guarded Amsterdam detector row (EHAM rows where v55 differs from v52). New value max(comparator + 1.0 x correction, 300).
Run from the repository root:  python3 -u -B research_320/build_eham_candidate.py
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import main_model  # noqa: F401,E402  (torch before LightGBM on macOS)
import joblib  # noqa: E402
import submission as SUB  # noqa: E402

ID = "MVT_ID_mvt"
C = HERE / ".cache/codex_eham"
V58 = HERE / ".cache/research_320/shrink_candidate/candidate_v58_shrink.parquet"
REL = HERE / ".cache/research_280/release"
OUT = HERE / ".cache/research_320/eham_candidate"
REPORT = HERE / "research_320/results/eham_candidate.json"
SCHEMA = ["comparator", "ersatz_s", "schedule", "gate", "hour_sin", "hour_cos", "first_age_s", "last_age_s", "span_s",
          "endpoint_distance_m", "weighted_speed_kt", "weighted_still_share", "observed_interval_share", "is_july"]
V58_SCORE = 259.9035


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rec58 = json.loads((HERE / "results/upbeat-goblin_v58.json").read_text())
    test = json.loads((HERE / "research_320/results/eham_weight.json").read_text())
    hashes = json.loads(subprocess.run(["git", "show", "8d98953:research_310/results/eham_recovery_screen.json"], cwd=HERE,
                                       capture_output=True, text=True, check=True).stdout)["model_hashes"]
    checks = {"v58_is_scored_file": sha(V58) == rec58["file_sha256"], "test_passed": test["verdict"] == "pass",
              "models_hash_identical": all(sha(C / "eham_recovery" / f"real_{s}.pkl") == hashes[f"real_{s}.pkl"] for s in (177, 277, 377))}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    upd = pd.read_parquet(C / "eham_on_scored_v56/updates.parquet")
    feats = pd.concat([pd.read_parquet(p) for p in sorted((C / "eham_recovery").glob("features_2026-01-*.parquet"))], ignore_index=True)
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet")
    x = upd[[ID]].merge(ctx[[ID] + SCHEMA[:6] + ["is_july"]], on=ID, validate="one_to_one").merge(
        feats[[ID] + SCHEMA[6:13]], on=ID, validate="one_to_one")
    if not x[ID].equals(upd[ID]):
        raise SystemExit("row alignment of the overlay failed")
    corr = np.mean([joblib.load(C / "eham_recovery" / f"real_{s}.pkl").predict(x[SCHEMA]) for s in (177, 277, 377)], axis=0)
    comp = x.comparator.to_numpy(float)
    checks["corrections_equal_codex"] = bool(np.array_equal(corr, upd.residual.to_numpy(float)))
    checks["max_abs_correction_difference"] = float(np.max(np.abs(corr - upd.residual.to_numpy(float))))
    checks["half_weight_reproduces_codex"] = bool(np.array_equal(np.maximum(comp + .5 * corr, 300.), upd.prediction.to_numpy(float)))
    v58 = pd.read_parquet(V58)
    cur = v58.set_index(ID).loc[upd[ID], "TAXITIME_SEC_mvt"].to_numpy()
    checks["comparator_equals_v58"] = bool(np.array_equal(comp.astype(cur.dtype), cur))
    t51 = pd.read_parquet(REL / "candidate_v51_combined.parquet").set_index(ID).TAXITIME_SEC_mvt
    t52 = pd.read_parquet(REL / "candidate_v52_combined.parquet").set_index(ID).TAXITIME_SEC_mvt
    t55 = pd.read_parquet(HERE / ".cache/combined_candidate/candidate_v55_combined.parquet").set_index(ID).TAXITIME_SEC_mvt
    ids = upd[ID].to_numpy()
    checks["no_nm_row"] = bool((t52.loc[ids] == t51.loc[ids]).all())
    checks["no_detector_row"] = bool((t55.loc[ids] == t52.loc[ids]).all())
    if not all(v for k, v in checks.items() if isinstance(v, bool)):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2) + "\n")
        raise SystemExit(f"a replay or protection gate failed: {checks}")
    new = pd.Series(np.maximum(comp + 1.0 * corr, 300.0), index=ids)
    cand = v58.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, "TAXITIME_SEC_mvt"] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    cand["TAXITIME_SEC_mvt"] = cand.TAXITIME_SEC_mvt.astype(template.TAXITIME_SEC_mvt.dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v58.loc[~hit].reset_index(drop=True))
    changed = int((cand.TAXITIME_SEC_mvt.to_numpy() != v58.TAXITIME_SEC_mvt.to_numpy()).sum())
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v58=bool(untouched),
                  all_finite=bool(np.isfinite(cand.TAXITIME_SEC_mvt).all()), overlay_rows=int(hit.sum()), rows_changed=changed)
    ok = not errors and untouched and checks["all_finite"]
    plan = test["planning"]
    report = {"checks": checks, "base": {"version": "v58", "score": V58_SCORE, "sha256": rec58["file_sha256"]},
              "planning": {"full_s": plan["full_s"], "half_s": plan["half_s"],
                           "prediction_fixed_before_upload": {"full_transfer": round(V58_SCORE - plan["full_s"], 2),
                                                              "half_transfer": round(V58_SCORE - plan["half_s"], 2),
                                                              "central": round(V58_SCORE - (plan["full_s"] + plan["half_s"]) / 2, 2)}}}
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v59_eham.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
