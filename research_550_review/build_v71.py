"""Build v71 = v70 with the stand model K2 (E, 400 x 15) in place of the deployed stand model (S, 200 x 7) on every row
the stand model corrects (research_550_review/PREREG_stand_capacity_retest.md passed). NM-free, no fit. No upload.

Rows exactly as in research_450_stand_january/build_v67.py: context rows observed_at_assigned_stand, the v57 cells with
owner "stand" plus the LSZH/EHAM January and July cells with owner "base". Each must first reproduce its v70 value as
max(comparator + 1.0 x deployed correction, 300), with detector_taxi_s from v67's cached 2026 re-extraction. New value
max(comparator + 1.0 x K2 correction, 300).
Prediction fixed here before the build runs: the retest's gain per row (+2,672.15) times the rows, central 0.8 of full
transfer (the stand model realised 0.87 in v67), worst 0.3, best 1.2, against v70's central 253.57.
Run from the repository root:  python3 -u -B research_550_review/build_v71.py
"""
import glob
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
from research_320 import stand_experiment as S  # noqa: E402
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402

ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 253.57
GAIN_PER_ROW = 2672.1531614765663
V57_CELLS = ("EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07")
LATER_CELLS = ("LSZH/07", "EHAM/07", "LSZH/01", "EHAM/01")
V70 = HERE / ".cache/research_510/v70/candidate_v70.parquet"
OUT = HERE / ".cache/research_550/v71"
REPORT = HERE / "research_550_review/results/v71_build.json"
K2 = HERE / ".cache/research_320/stand_capacity/models/k2_real_{seed}.pkl"
DEPLOYED = HERE / ".cache/research_320/stand/models/real_{seed}.pkl"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stand_rows():
    context = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet")
    context["cell"] = context.airport + "/" + context.date.str[5:7]
    st = context[context.category.eq("observed_at_assigned_stand") & context.cell.isin(V57_CELLS + LATER_CELLS)].copy()
    own = H.ownership_v62().set_index(ID).owner
    st["owner"] = own.loc[st[ID]].to_numpy()
    st = st[(st.cell.isin(V57_CELLS) & st.owner.eq("stand")) | (st.cell.isin(LATER_CELLS) & st.owner.eq("base"))].copy()
    det = pd.concat([pd.read_parquet(p, columns=[ID, "detector_taxi_s"])
                     for p in sorted(glob.glob(str(HERE / ".cache/research_450/stand_2026/day_*.parquet")))]).drop_duplicates(ID).set_index(ID)
    st["detector_taxi_s"] = det.detector_taxi_s.reindex(st[ID]).to_numpy(float)
    return st.reset_index(drop=True), bool(st[ID].isin(det.index).all())


def correction(frame, pattern):
    x = frame[S.COLS]
    models = [joblib.load(str(pattern).format(seed=s)) for s in S.SEEDS]
    c = np.mean([m.predict(x) for m in models], axis=0)
    rev = np.mean([m.predict(x.iloc[::-1]) for m in models], axis=0)[::-1]
    return c, bool(np.array_equal(c, rev))


def main():
    build70 = json.loads((HERE / "research_510_rome_classifier/results/v70_build.json").read_text())
    retest = json.loads((HERE / "research_550_review/results/stand_capacity_retest.json").read_text())
    checks = {"v70_is_built_file": sha(V70) == build70["candidate_sha256"], "retest_passed": retest["decision"] == "pass",
              "gain_as_fixed": abs(retest["k2"]["gain_mse"] - GAIN_PER_ROW) < 1e-6}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    st, covered = stand_rows()
    checks.update(rows=int(len(st)), rows_by_cell=st.cell.value_counts().to_dict(), detector_input_for_every_row=covered)
    base = H.current_base_2026()
    checks["comparator_equals_pre_nm_base"] = bool(np.array_equal(st.comparator.to_numpy(float), base.loc[st[ID]].to_numpy(float)))
    v70 = pd.read_parquet(V70)
    at70 = v70.set_index(ID)[T]
    c_dep, rep_dep = correction(st, DEPLOYED)
    c_k2, rep_k2 = correction(st, K2)
    b = st.comparator.to_numpy(float)
    replay = np.maximum(b + c_dep, 300.0).astype(v70[T].dtype)
    checks["reverse_replay"] = rep_dep and rep_k2
    checks["rows_reproduce_v70_with_deployed_model"] = int((replay == at70.loc[st[ID]].to_numpy()).sum())
    checks["all_reproduce"] = checks["rows_reproduce_v70_with_deployed_model"] == len(st)
    if not (checks["all_reproduce"] and checks["reverse_replay"] and checks["comparator_equals_pre_nm_base"] and covered):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2, default=str) + "\n")
        raise SystemExit(f"a replay gate failed: {checks}")
    new = pd.Series(np.maximum(b + c_k2, 300.0), index=st[ID].to_numpy())
    cand = v70.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v70.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v70.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v70=bool(untouched),
                  all_finite=bool(np.isfinite(cand[T]).all()), rows_changed=int((cand[T].to_numpy() != v70[T].to_numpy()).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2)))})
    mse = len(st) * GAIN_PER_ROW / N
    at = lambda k: round(float(np.sqrt(REF ** 2 - k * mse)), 2)
    report = {"checks": checks, "base": {"version": "v70", "sha256": build70["candidate_sha256"], "central": REF},
              "prediction_fixed_before_upload": {"assumptions": "retest gain per row x rows; central 0.8, worst 0.3, best 1.2 of full transfer",
                                                 "mse_per_scored_row_full": mse, "central": at(0.8), "worst": at(0.3), "best": at(1.2)}}
    ok = not errors and untouched and checks["all_finite"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v71.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
