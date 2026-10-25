"""Build v68 = the exact scored v67 with the Amsterdam January add-on averaged with C1 on the rows the add-on corrects
(research_480_weights/PREREG_weights.md, test 2 passed). NM-free. No upload.

Rows: owner "eham_january" in the v62 provenance (9,752), untouched since v59. Each must first reproduce its v67 value
exactly as max(base + add-on correction, 300) from Codex's 2026 context table and his 2026 add-on feature files; that
proves the inputs equal what shipped. New value max(base + (add-on correction + C1 correction) / 2, 300).
Run from the repository root:  python3 -u -B research_480_weights/build_v68.py
"""
import glob
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
from research_320 import extract_episodes as X  # noqa: E402

MS = X.v56_modules()
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 256.0858
V67 = HERE / ".cache/research_450/v67/candidate_v67.parquet"
OUT = HERE / ".cache/research_480/v68"
REPORT = HERE / "research_480_weights/results/v68_build.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rec67 = json.loads((HERE / "results/upbeat-goblin_v67.json").read_text())
    test = json.loads((HERE / "research_480_weights/results/weights.json").read_text())["test2_eham_average"]
    checks = {"v67_is_scored_file": sha(V67) == rec67["file_sha256"], "test2_passed": test["verdict"] == "pass"}
    if not all(checks.values()):
        raise SystemExit(f"a precondition failed: {checks}")
    own = H.ownership_v62()
    ids = own.loc[own.owner.eq("eham_january"), ID]
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet").set_index(ID)
    checks["owned_rows"] = int(len(ids))
    checks["owned_rows_in_context"] = int(ids.isin(ctx.index).sum())
    feats = pd.concat([pd.read_parquet(p, columns=[ID] + list(L.EHAM_TRACK))
                       for p in sorted(glob.glob(str(HERE / ".cache/codex_eham/eham_recovery/features_2026-01-*.parquet")))], ignore_index=True)
    frame = ctx.loc[ids].reset_index().merge(feats.drop_duplicates(ID), on=ID, how="inner", validate="one_to_one")
    checks["rows_with_features"] = int(len(frame))
    base = H.current_base_2026()
    checks["comparator_equals_pre_nm_base"] = bool(np.array_equal(frame.comparator.to_numpy(float), base.loc[frame[ID]].to_numpy(float)))
    v67 = pd.read_parquet(V67)
    at67 = v67.set_index(ID).loc[frame[ID], T].to_numpy()
    addon = L.correction("eham_january", frame)
    c1 = L.correction("partial", frame)
    b = frame.comparator.to_numpy(float)
    checks["rows_reproduce_v67_from_addon"] = int((np.maximum(b + addon, 300.0).astype(at67.dtype) == at67).sum())
    checks["all_rows_reproduce"] = checks["rows_reproduce_v67_from_addon"] == len(frame) == len(ids)
    rev = L.correction("partial", frame.iloc[::-1].reset_index(drop=True))[::-1]
    checks["c1_reverse_replay"] = bool(np.array_equal(c1, rev))
    if not (checks["all_rows_reproduce"] and checks["comparator_equals_pre_nm_base"] and checks["c1_reverse_replay"]):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2) + "\n")
        raise SystemExit(f"a replay gate failed: {checks}")
    new = pd.Series(np.maximum(b + (addon + c1) / 2, 300.0), index=frame[ID].to_numpy())
    cand = v67.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v67.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v67.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v67=bool(untouched),
                  all_finite=bool(np.isfinite(cand[T]).all()), rows_changed=int((cand[T].to_numpy() != v67[T].to_numpy()).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2)))})
    ok = not errors and untouched and checks["all_finite"]
    mse = len(frame) * test["confirmation"]["gain_mse"] / N
    at = lambda k: round(REF - (REF - np.sqrt(REF ** 2 - k * mse)), 2)
    report = {"checks": checks, "base": {"version": "v67", "score": rec67["score"], "sha256": rec67["file_sha256"]},
              "prediction_fixed_before_upload": {"assumptions": "fresh-January gain per row times the 2026 rows; central 0.8, worst 0.3, best 1.2",
                                                 "mse_per_scored_row_full": mse, "central": at(0.8), "worst": at(0.3), "best": at(1.2)}}
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v68.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
