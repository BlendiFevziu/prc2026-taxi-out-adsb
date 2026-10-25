"""Build v78 = v77 with CatBoost averaged into every family that passed research_640_tail_rule/PREREG_catboost_layers.md:
C1+ rows (84,970, rows as build_v72) and/or K2+ rows (17,147, rows as build_v71). NM-free, no fit. No upload.

Gate: each row first reproduces its v77 value as max(base + deployed correction, 300). New value
max(base + (deployed + CatBoost) / 2, 300). Prediction fixed here before the build runs: each passing family's fresh-day
gain per row times its rows, central 0.8, worst 0.3, best 1.2 of full transfer, against v77's official 246.3837.
Run from the repository root:  python3 -u -B research_640_tail_rule/build_v78.py
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_550_review", ROOT / "research_640_tail_rule"):
    sys.path.insert(0, str(p))
import build_v71 as B71  # noqa: E402
import build_v72 as B72  # noqa: E402

S, L, H = B71.S, B72.L, B72.H
import submission as SUB  # noqa: E402

ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 246.3837
V77 = ROOT / ".cache/research_590/v77/candidate_v77.parquet"
OUT = ROOT / ".cache/research_640/v78"
REPORT = ROOT / "research_640_tail_rule/results/v78_build.json"
DEPLOYED = {"c1": ".cache/research_550/c1_more/c1plus_{seed}.pkl", "stand": ".cache/research_550/k2_more/k2plus_{seed}.pkl"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def corrections(fam, frame):
    x = L.inputs("partial", frame) if fam == "c1" else frame[S.COLS]
    dep = np.mean([joblib.load(ROOT / DEPLOYED[fam].format(seed=s)).predict(x) for s in L.SEEDS], axis=0)
    cat = np.mean([joblib.load(ROOT / f".cache/research_640/cat_{fam}_EB1B2_{s}.pkl").predict(x) for s in L.SEEDS], axis=0)
    return dep, cat


def main():
    test = json.loads((ROOT / "research_640_tail_rule/results/catboost_layers.json").read_text())
    rec77 = json.loads((ROOT / "results/upbeat-goblin_v77.json").read_text())
    passing = [f for f in ("c1", "stand") if test[f]["verdict"] == "pass"]
    checks = {"v77_is_uploaded_file": sha(V77) == rec77["file_sha256"], "families_passing": passing}
    if not checks["v77_is_uploaded_file"] or not passing:
        raise SystemExit(f"precondition failed: {checks}")
    v77 = pd.read_parquet(V77)
    at77 = v77.set_index(ID)[T]
    base = H.current_base_2026()
    new, rows = {}, {}
    for fam in passing:
        frame = B72.c1_rows() if fam == "c1" else B71.stand_rows()[0]
        b = frame.comparator.to_numpy(float)
        dep, cat = corrections(fam, frame)
        checks[f"{fam}_comparator_equals_base"] = bool(np.array_equal(b, base.loc[frame[ID]].to_numpy(float)))
        checks[f"{fam}_rows_reproduce_v77"] = int((np.maximum(b + dep, 300.0).astype(v77[T].dtype) == at77.loc[frame[ID]].to_numpy()).sum())
        checks[f"{fam}_rows"] = int(len(frame))
        if checks[f"{fam}_rows_reproduce_v77"] != len(frame) or not checks[f"{fam}_comparator_equals_base"]:
            REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2, default=str) + "\n")
            raise SystemExit(f"replay gate failed for {fam}: {checks}")
        new.update(dict(zip(frame[ID], np.maximum(b + (dep + cat) / 2, 300.0))))
        rows[fam] = len(frame)
    s = pd.Series(new)
    cand = v77.copy()
    hit = cand[ID].isin(s.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(s).to_numpy(float)
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v77.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v77.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v77=bool(untouched), all_finite=bool(np.isfinite(cand[T]).all()),
                  rows_changed=int((cand[T].to_numpy() != v77[T].to_numpy()).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2)))})
    mse = sum(rows[f] * test[f]["gain_mse"] for f in passing) / N
    at = lambda k: round(float(np.sqrt(REF ** 2 - k * mse)), 2)
    report = {"checks": checks, "base": {"version": "v77", "score": REF, "sha256": rec77["file_sha256"]},
              "prediction_fixed_before_upload": {"assumptions": "fresh-day gain per row x rows per passing family; central 0.8, worst 0.3, best 1.2",
                                                 "mse_per_scored_row_full": mse, "central": at(0.8), "worst": at(0.3), "best": at(1.2)}}
    ok = not errors and untouched and checks["all_finite"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v78.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
