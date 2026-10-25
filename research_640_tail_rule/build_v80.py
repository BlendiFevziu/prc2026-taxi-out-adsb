"""Build v80 = v79 plus the two near-misses admitted by the owner's retroactive exception (POLICY_tail_rule.md):
(a) Paris: the drift-robust C1 on the base-owned 2026 LFPG partial rows (January and July), as tested in
    research_590_stand_new/PREREG_drift_robust_c1_paris.md (model alone, all airport flags zero);
(b) Amsterdam January: C1+ in place of the old C1 inside the add-on average on the 9,752 eham_january rows.
NM-free, no fit. No upload.

Gates: (a) LFPG rows from the coverage-audit cache with the v64 context recipe (which must reproduce Codex's rows), every
row at the base in v79; (b) rows exactly as build_v68, every row reproducing its v79 value as
max(base + (add-on + old C1) / 2, 300). New values: (a) max(base + drift-robust C1, 300); (b) max(base + (add-on + C1+) / 2, 300).
Prediction fixed here before the build runs, discounted for the post-hoc admission: half the planned gain as central,
the full planned gain as best, and a worst case of minus half the planned gain, against v79's central 245.87.
Run from the repository root:  python3 -u -B research_640_tail_rule/build_v80.py
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
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_400_cells"):
    sys.path.insert(0, str(p))
import build_v64 as B64  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402
import submission as SUB  # noqa: E402

ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 245.87
GAIN = {"paris": 15653.632929818397, "eham": 1154.2212682663508}        # tested gains per row (original test data)
V79 = ROOT / ".cache/research_640/v79/candidate_v79.parquet"
OUT = ROOT / ".cache/research_640/v80"
REPORT = ROOT / "research_640_tail_rule/results/v80_build.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mean_pred(pattern, x):
    return np.mean([joblib.load(ROOT / pattern.format(seed=s)).predict(x) for s in L.SEEDS], axis=0)


def paris_rows():
    own = H.ownership_v62().set_index(ID).owner
    a = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / ".cache/research_320/airport_coverage/day_2026-*.parquet")))], ignore_index=True)
    a = a[a.airport.eq("LFPG") & a.category.eq("partial_ground") & own.reindex(a[ID]).eq("base").to_numpy()].reset_index(drop=True)
    return B64.context_2026(a)


def eham_rows():
    own = H.ownership_v62()
    ids = own.loc[own.owner.eq("eham_january"), ID]
    ctx = pd.read_parquet(ROOT / ".cache/codex_v56/partial_release/context.parquet").set_index(ID)
    feats = pd.concat([pd.read_parquet(p, columns=[ID] + list(L.EHAM_TRACK))
                       for p in sorted(glob.glob(str(ROOT / ".cache/codex_eham/eham_recovery/features_2026-01-*.parquet")))], ignore_index=True)
    return ctx.loc[ids].reset_index().merge(feats.drop_duplicates(ID), on=ID, how="inner", validate="one_to_one")


def main():
    build79 = json.loads((ROOT / "research_640_tail_rule/results/v79_build.json").read_text())
    checks = {"v79_is_built_file": sha(V79) == build79["candidate_sha256"]}
    ctx = pd.read_parquet(ROOT / ".cache/codex_v56/partial_release/context.parquet")
    probe = ctx.sample(3000, random_state=64).reset_index(drop=True)
    rebuilt = B64.context_2026(probe[[c for c in probe.columns if c not in B64.CONTEXT and not c.startswith("airport_") and c != "is_july"]])
    checks["context_recipe_reproduces_codex"] = all(np.array_equal(rebuilt[c].to_numpy(float), probe[c].to_numpy(float)) for c in B64.CONTEXT)
    v79 = pd.read_parquet(V79)
    at79 = v79.set_index(ID)[T]
    base = H.current_base_2026()
    # (a) Paris
    pa = paris_rows()
    cols = json.loads((ROOT / "research_590_stand_new/results/drift_robust_c1.json").read_text())["inputs"]
    from research_320 import stand_experiment as S
    xa = S.MS.inputs(pa, "real")[cols]
    ba = pa.comparator.to_numpy(float)
    checks["paris_rows"] = int(len(pa))
    checks["paris_comparator_equals_base"] = bool(np.array_equal(ba, base.loc[pa[ID]].to_numpy(float)))
    checks["paris_rows_at_base_in_v79"] = bool(np.array_equal(ba.astype(v79[T].dtype), at79.loc[pa[ID]].to_numpy()))
    new_a = np.maximum(ba + mean_pred(".cache/research_590/drift_robust_c1/drc1_{seed}.pkl", xa), 300.0)
    # (b) Amsterdam January
    eh = eham_rows()
    be = eh.comparator.to_numpy(float)
    addon = L.correction("eham_january", eh)
    old = L.correction("partial", eh)
    plus = mean_pred(".cache/research_550/c1_more/c1plus_{seed}.pkl", L.inputs("partial", eh))
    checks["eham_rows"] = int(len(eh))
    checks["eham_comparator_equals_base"] = bool(np.array_equal(be, base.loc[eh[ID]].to_numpy(float)))
    checks["eham_reproduce_v79"] = int((np.maximum(be + (addon + old) / 2, 300.0).astype(v79[T].dtype) == at79.loc[eh[ID]].to_numpy()).sum())
    gates = (checks["context_recipe_reproduces_codex"] and checks["paris_comparator_equals_base"] and checks["paris_rows_at_base_in_v79"]
             and checks["eham_comparator_equals_base"] and checks["eham_reproduce_v79"] == len(eh) == 9752 and checks["paris_rows"] == 3786)
    if not gates:
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2, default=str) + "\n")
        raise SystemExit(f"a gate failed: {checks}")
    new = pd.Series({**dict(zip(pa[ID], new_a)), **dict(zip(eh[ID], np.maximum(be + (addon + plus) / 2, 300.0)))})
    cand = v79.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v79.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v79.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v79=bool(untouched), all_finite=bool(np.isfinite(cand[T]).all()),
                  rows_changed=int((cand[T].to_numpy() != v79[T].to_numpy()).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2)))})
    mse = (GAIN["paris"] * len(pa) + GAIN["eham"] * len(eh)) / N
    at = lambda k: round(float(np.sqrt(REF ** 2 - k * mse)), 2)
    report = {"checks": checks, "base": {"version": "v79", "central": REF, "sha256": build79["candidate_sha256"]},
              "admission": "owner's retroactive exception, research_640_tail_rule/POLICY_tail_rule.md",
              "prediction_fixed_before_upload": {"assumptions": "tested gain per row x rows; discounted: central 0.5, best 1.0, worst -0.5",
                                                 "mse_per_scored_row_full": mse, "central": at(0.5), "best": at(1.0), "worst": at(-0.5)}}
    ok = not errors and untouched and checks["all_finite"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v80.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
