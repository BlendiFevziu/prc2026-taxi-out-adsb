"""Build v62 = the exact scored v61 with the partial-track correction C1 (800 x 31, research_320/PREREG_capacity2.md)
in place of arm 4 (400 x 15) on the same 72,979 rows. NM-free like v61. No fit, no upload.

Rows: v56's 66,292 overlay IDs plus the 6,687 partial rows the NM-free build added. Before anything changes, v61's
value on every one of them must equal max(comparator + mean arm 4 correction, 300) exactly (it is the deployed
model); the new value is max(comparator + mean C1 correction, 300). Every other row stays byte-identical to v61.
Run from the repository root:  python3 -u -B research_320/build_v62.py
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
V61 = HERE / ".cache/research_320/nmfree_plus/candidate_nmfree_plus.parquet"
REF = HERE / "research_330_nmfree/private/external_nm_free_reference.parquet"
MODELS = HERE / ".cache/research_320/volume/models"
OUT = HERE / ".cache/research_320/v62"
REPORT = HERE / "research_320/results/v62_build.json"
PARTIAL_CELLS = ("EDDF/01", "EGLL/01", "EHAM/07", "LEBL/01", "LEBL/07", "LSZH/01", "LSZH/07")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rec61 = json.loads((HERE / "results/upbeat-goblin_v61.json").read_text())
    cap = json.loads((HERE / "research_320/results/capacity2.json").read_text())
    vol = json.loads((HERE / "research_320/results/volume_capacity.json").read_text())
    checks = {"v61_is_scored_file": sha(V61) == rec61["file_sha256"],
              "c1_passed": cap["planning"]["chosen"] == "c1" and "c1" in cap["passing_both"],
              "c1_models_as_tested": all(sha(MODELS / f"c1_real_{s}.pkl") == cap["model_hashes"][f"c1_real_{s}.pkl"] for s in SEEDS),
              "arm4_models_as_deployed": all(sha(MODELS / f"arm4_real_{s}.pkl") == vol["model_hashes"][f"arm4_real_{s}.pkl"] for s in SEEDS)}
    if not all(checks.values()):
        raise SystemExit(f"a precondition failed: {checks}")
    v61, ref = pd.read_parquet(V61), pd.read_parquet(REF)
    upd = pd.read_parquet(HERE / ".cache/codex_v56/partial_release_v55/updates.parquet")
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet")
    ctx["cell"] = ctx.airport + "/" + ctx.date.str[5:7]
    # the NM-free build's own rule (build_nmfree_plus.py): partial rows of the seven cells on days 6/13/20 that sat at
    # the comparator in Codex's reference; selecting by "value changed" would miss rows clipped to the 300 s floor
    at_ref = ref.set_index(ID)[T]
    nm_day = ctx[ctx.date.str[8:10].isin(["06", "13", "20"])]
    nm_partial = nm_day[nm_day.cell.isin(PARTIAL_CELLS) & nm_day.category.eq("partial_ground")
                        & (at_ref.loc[nm_day[ID]].to_numpy(float) == nm_day.comparator.to_numpy(float))][ID]
    nm_partial = nm_partial[~nm_partial.isin(upd[ID])]
    ids = pd.Index(upd[ID]).append(pd.Index(nm_partial))
    checks["rows"] = {"v56_overlay": int(len(upd)), "nm_free_partial": int(len(nm_partial)), "total": int(len(ids))}
    checks["ids_unique"] = bool(ids.is_unique)
    rows = ctx.set_index(ID).loc[ids].reset_index()
    x = MS.inputs(rows, "real")
    comp = rows.comparator.to_numpy(float)
    arm4 = np.mean([R.joblib.load(MODELS / f"arm4_real_{s}.pkl").predict(x) for s in SEEDS], axis=0)
    c1 = np.mean([R.joblib.load(MODELS / f"c1_real_{s}.pkl").predict(x) for s in SEEDS], axis=0)
    c1_rev = np.mean([R.joblib.load(MODELS / f"c1_real_{s}.pkl").predict(x.iloc[::-1]) for s in SEEDS], axis=0)[::-1]
    at61 = v61.set_index(ID).loc[ids, T]
    checks["v61_rows_equal_arm4_formula"] = bool(np.array_equal(np.maximum(comp + arm4, 300.0).astype(at61.dtype), at61.to_numpy()))
    checks["c1_reverse_replay"] = bool(np.array_equal(c1, c1_rev))
    if not (checks["v61_rows_equal_arm4_formula"] and checks["c1_reverse_replay"] and checks["ids_unique"]):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2) + "\n")
        raise SystemExit(f"a replay gate failed: {checks}")
    new = pd.Series(np.maximum(comp + c1, 300.0), index=ids)
    cand = v61.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v61.loc[~hit].reset_index(drop=True))
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v61=bool(untouched),
                  all_finite=bool(np.isfinite(cand[T]).all()), rows_changed=int((cand[T].to_numpy() != v61[T].to_numpy()).sum()),
                  rms_change_s=float(np.sqrt(np.mean((c1 - arm4) ** 2))))
    ok = not errors and untouched and checks["all_finite"]
    report = {"checks": checks, "base": {"version": "v61", "score": rec61["score"], "sha256": rec61["file_sha256"]},
              "planning": cap["planning"]}
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v62.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
