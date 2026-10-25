"""Build v76 = v75 plus the drift-robust stand model on the base-owned 2026 stand rows of the cells admitted by
research_590_stand_new/PREREG_drift_robust.md (LEMD/07, LEMD/01, EGLL/07). NM-free, no fit. No upload.

Gates before any change: (1) the v64 context recipe reproduces 3,000 rows of Codex's context table exactly; (2) the
extraction with the detector input reproduces the reference extraction's category and 23 summaries on every target
row (the 2026 coverage-audit cache at LEMD, Codex's context table at EGLL); (3) every target row sits exactly at the
base in v75. New value max(base + correction, 300).
Prediction fixed here before the build runs: each admitted cell's tested gain per row times its rows; because EGLL/07
rests on a sister-month transfer to far denser tracks, central 0.6, best 1.0 and worst -0.2 of full transfer, against
v75's official 249.3908.
Run from the repository root:  python3 -u -B research_590_stand_new/build_v76.py
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
import extract as E  # noqa: E402
import drift_robust as DR  # noqa: E402

S, H, T = DR.S, DR.H, DR.T
import submission as SUB  # noqa: E402

ID, TT, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 249.3908
V75 = ROOT / ".cache/research_550/v75/candidate_v75.parquet"
OUT = ROOT / ".cache/research_590/v76"
REPORT = ROOT / "research_590_stand_new/results/v76_build.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    test = json.loads((ROOT / "research_590_stand_new/results/drift_robust.json").read_text())
    rec75 = json.loads((ROOT / "results/upbeat-goblin_v75.json").read_text())
    checks = {"v75_is_uploaded_file": sha(V75) == rec75["file_sha256"], "test_passed": test["verdict"] == "pass"}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    cells = test["admitted"]
    ctx = pd.read_parquet(ROOT / ".cache/codex_v56/partial_release/context.parquet")
    probe = ctx.sample(3000, random_state=64).reset_index(drop=True)
    rebuilt = B64.context_2026(probe[[c for c in probe.columns if c not in B64.CONTEXT and not c.startswith("airport_") and c != "is_july"]])
    checks["context_recipe_reproduces_codex"] = all(np.array_equal(rebuilt[c].to_numpy(float), probe[c].to_numpy(float)) for c in B64.CONTEXT)
    own = H.ownership_v62().set_index(ID).owner
    frames, ref_ok = [], {}
    audit = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / ".cache/research_320/airport_coverage/day_2026-*.parquet")))],
                      ignore_index=True).set_index(ID)
    for ap in sorted({c[:4] for c in cells}):
        if ap == "EGLL":
            dep = E.departures_2026("EGLL")
            days = sorted(d for d in dep.date.unique() if d[5:7] in {c[-2:] for c in cells if c.startswith("EGLL")})
            E.extract("EGLL", dep, days)
            r = E.load("EGLL", days).assign(month=lambda x: x.date.str[5:7])
            ref = ctx.set_index(ID)
        else:
            r = T.rows_2026(ap)
            ref = audit
        r = r[r.category.eq("observed_at_assigned_stand") & (r.airport + "/" + r.month).isin(cells) & own.reindex(r[ID]).eq("base").to_numpy()]
        a = ref.loc[r[ID]]
        ref_ok[ap] = bool((a.category.to_numpy() == r.category.to_numpy()).all() and all(
            np.array_equal(a[f].to_numpy(float), r[f].to_numpy(float), equal_nan=True) for f in S.MS.FEATURES))
        frames.append(r)
    rows = pd.concat(frames, ignore_index=True)
    checks["extraction_equals_reference"] = ref_ok
    frame = B64.context_2026(rows.drop(columns=["month"]))
    v75 = pd.read_parquet(V75)
    at75 = v75.set_index(ID)[TT]
    b = frame.comparator.to_numpy(float)
    cell = (frame.airport + "/" + frame.date.str[5:7]).to_numpy()
    checks["rows"] = {c: int((cell == c).sum()) for c in cells}
    checks["rows_as_tested"] = all(checks["rows"][c] == test["cells"][c]["rows_2026"] for c in cells)
    checks["rows_at_base_in_v75"] = bool(np.array_equal(b.astype(v75[TT].dtype), at75.loc[frame[ID]].to_numpy()))
    if not (checks["context_recipe_reproduces_codex"] and all(ref_ok.values()) and checks["rows_at_base_in_v75"] and checks["rows_as_tested"]):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2, default=str) + "\n")
        raise SystemExit(f"a gate failed: {checks}")
    columns = DR.cols()
    ms = DR.models(None, columns)
    x = S.inputs(frame, "real")[columns]
    corr = np.mean([m.predict(x) for m in ms], axis=0)
    rev = np.mean([m.predict(x.iloc[::-1]) for m in ms], axis=0)[::-1]
    checks["reverse_replay"] = bool(np.array_equal(corr, rev))
    new = pd.Series(np.maximum(b + corr, 300.0), index=frame[ID].to_numpy())
    cand = v75.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, TT] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    cand[TT] = cand[TT].astype(template[TT].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v75.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, TT].to_numpy(float) - v75.loc[hit, TT].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v75=bool(untouched), all_finite=bool(np.isfinite(cand[TT]).all()),
                  rows_changed=int((cand[TT].to_numpy() != v75[TT].to_numpy()).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2)))})
    d = new.to_numpy() - at75.loc[frame[ID]].to_numpy(float)
    checks["change_by_cell"] = {c: {"mean": float(d[cell == c].mean()), "rms": float(np.sqrt(np.mean(d[cell == c] ** 2)))} for c in cells}
    gain = {c: test["cells"][c]["test"]["gain_mse"] if "test" in test["cells"][c] else test["cells"][test["cells"][c]["sister_of"]]["test"]["gain_mse"]
            for c in cells}
    mse = sum(gain[c] * checks["rows"][c] for c in cells) / N
    at = lambda k: round(float(np.sqrt(REF ** 2 - k * mse)), 2)
    report = {"checks": checks, "base": {"version": "v75", "score": REF, "sha256": rec75["file_sha256"]},
              "prediction_fixed_before_upload": {"assumptions": "tested gain per row x rows per admitted cell; central 0.6, best 1.0, worst -0.2 of full transfer",
                                                 "mse_per_scored_row_full": mse, "central": at(0.6), "best": at(1.0), "worst": at(-0.2)}}
    ok = not errors and untouched and checks["all_finite"] and checks["reverse_replay"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v76.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
