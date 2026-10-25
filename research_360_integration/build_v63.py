"""Build v63 = the exact scored v62 with the frozen partial-track correction C1 switched on for EDDF July, as fixed in
research_360_integration/PREREG_eddf_july.md after its confirmation passed. NM-free like v62. No fit, no upload.

Rows: the 11,993 EDDF July partial_ground rows of Codex's 2026 context table (the same IDs as his 62-day feature
table). v62 owns them at the base, so before anything changes v62's value on every one of them must equal the
comparator exactly; the new value is max(comparator + mean C1 correction, 300), the recipe v62 uses on its own
72,977 partial rows. Every other row stays byte-identical to v62.
The prediction for the official score is fixed here, before any upload, from the confirmation's own numbers.
Run from the repository root:  python3 -u -B research_360_integration/build_v63.py
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import extract_episodes as X  # noqa: E402

MS = X.v56_modules()
R = MS.R
import submission as SUB  # noqa: E402

ID, T, SEEDS = "MVT_ID_mvt", "TAXITIME_SEC_mvt", (177, 277, 377)
CELL, ROWS_REGISTERED, N, V62_SCORE = "EDDF/07", 11993, 344841, 258.5995
V62 = HERE / ".cache/research_320/v62/candidate_v62.parquet"
MODELS = HERE / ".cache/research_320/volume/models"
CTX = HERE / ".cache/codex_v56/partial_release/context.parquet"
FEATURES = HERE / ".cache/codex_v56/full_coverage_2026/features.parquet"
CONFIRM = HERE / "research_360_integration/results/eddf_july_confirm.json"
OUT = HERE / ".cache/research_360/v63"
REPORT = HERE / "research_360_integration/results/v63_build.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def delta_s(mse_per_scored_row):
    return float(V62_SCORE - np.sqrt(V62_SCORE ** 2 - mse_per_scored_row))


def prediction(confirm):
    """Fixed before upload. Group level = the registered rule-5 value; stratum level = each stratum with at least 30
    confirmation rows at its own gain and its 2026 share (research_360 part 1), smaller strata at zero."""
    shares = json.loads((HERE / "research_360_integration/results/error_by_track_quality.json").read_text())["partial_cells"][CELL]["strata"]
    stratum = sum(v["share_2026"] * confirm["by_stratum"][s]["gain_mse"] for s, v in shares.items()
                  if s in confirm["by_stratum"] and confirm["by_stratum"][s]["rows"] >= 30)
    group = confirm["reweighted_to_2026_mix"]
    per_row = lambda g: ROWS_REGISTERED * g / N
    return {"assumptions": "stratum-level gain x 2026 rows; central at v60's realised transfer 0.88 (as for v62), worst at 0.5, "
                           "best = the registered group-level value at full transfer; v62 realised 0.59",
            "stratum_level_gain_per_row": stratum, "group_level_gain_per_row": group,
            "central": round(V62_SCORE - delta_s(0.88 * per_row(stratum)), 2),
            "worst": round(V62_SCORE - delta_s(0.5 * per_row(stratum)), 2),
            "best": round(V62_SCORE - delta_s(per_row(group)), 2),
            "stratum_level_full_transfer": round(V62_SCORE - delta_s(per_row(stratum)), 2)}


def main():
    rec62 = json.loads((HERE / "results/upbeat-goblin_v62.json").read_text())
    cap = json.loads((HERE / "research_320/results/capacity2.json").read_text())
    confirm = json.loads(CONFIRM.read_text())
    checks = {"v62_is_scored_file": sha(V62) == rec62["file_sha256"],
              "confirmation_passed": confirm["verdict"] == "pass" and all(confirm["rules"].values()),
              "c1_models_as_tested": all(sha(MODELS / f"c1_real_{s}.pkl") == cap["model_hashes"][f"c1_real_{s}.pkl"] for s in SEEDS)}
    if not all(checks.values()):
        raise SystemExit(f"a precondition failed: {checks}")
    v62, ctx = pd.read_parquet(V62), pd.read_parquet(CTX)
    ctx["cell"] = ctx.airport + "/" + ctx.date.str[5:7]
    rows = ctx[ctx.cell.eq(CELL) & ctx.category.eq("partial_ground")].reset_index(drop=True)
    feat = pd.read_parquet(FEATURES, columns=[ID, "airport", "date", "category"])
    feat_ids = set(feat.loc[(feat.airport + "/" + feat.date.str[5:7]).eq(CELL) & feat.category.eq("partial_ground"), ID])
    comp = rows.comparator.to_numpy(float)
    at62 = v62.set_index(ID).loc[rows[ID], T].to_numpy()
    checks.update(rows=int(len(rows)), rows_equal_registered=len(rows) == ROWS_REGISTERED,
                  ids_equal_feature_table=set(rows[ID]) == feat_ids, ids_unique=bool(rows[ID].is_unique),
                  v62_rows_at_comparator=bool(np.array_equal(comp.astype(at62.dtype), at62)))
    x = MS.inputs(rows, "real")
    c1 = np.mean([R.joblib.load(MODELS / f"c1_real_{s}.pkl").predict(x) for s in SEEDS], axis=0)
    c1_rev = np.mean([R.joblib.load(MODELS / f"c1_real_{s}.pkl").predict(x.iloc[::-1]) for s in SEEDS], axis=0)[::-1]
    checks["c1_reverse_replay"] = bool(np.array_equal(c1, c1_rev))
    gates = ("rows_equal_registered", "ids_equal_feature_table", "ids_unique", "v62_rows_at_comparator", "c1_reverse_replay")
    if not all(checks[g] for g in gates):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2) + "\n")
        raise SystemExit(f"a replay gate failed: {checks}")
    new = pd.Series(np.maximum(comp + c1, 300.0), index=rows[ID].to_numpy())
    cand = v62.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v62.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v62.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v62=bool(untouched),
                  all_finite=bool(np.isfinite(cand[T]).all()), rows_changed=int((cand[T].to_numpy() != v62[T].to_numpy()).sum()),
                  rows_at_floor=int((new.to_numpy() == 300.0).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2))),
                            "p05": float(np.percentile(change, 5)), "p95": float(np.percentile(change, 95))})
    ok = not errors and untouched and checks["all_finite"]
    report = {"checks": checks, "base": {"version": "v62", "score": rec62["score"], "sha256": rec62["file_sha256"]},
              "prediction_fixed_before_upload": prediction(confirm)}
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v63.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
