"""Independent pre-upload check of v66 (research_440_fresh_july/build_v66.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v65 and the template; submission.pruefe clean; all finite
2. the changed set, without the ownership table: stand-observed rows of LSZH July and EHAM July in Codex's 2026 FEATURE
   table (the build used his context table for the category) whose v65 value equals the pre-NM base
   (harness.current_base_2026); every changed row lies in it, everything outside is identical to v65
3. every v66 value in the set equals max(base + 0.5 x mean stand correction, 300), recomputed from the stand models and
   the cached re-extraction, with the comparator taken from the pre-NM base file
Run from the repository root:  python3 -u -B research_440_fresh_july/verify_v66.py
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
from research_320 import stand_experiment as S  # noqa: E402

R = S.R
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
V66 = HERE / ".cache/research_440/v66/candidate_v66.parquet"
V65 = HERE / ".cache/research_400/v65/candidate_v65.parquet"
OUT = HERE / "research_440_fresh_july/results/v66_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((HERE / "research_440_fresh_july/results/v66_build.json").read_text())
    rec65 = json.loads((HERE / "results/upbeat-goblin_v65.json").read_text())
    c = {"v66_sha_equals_build": sha(V66) == build["candidate_sha256"], "v65_is_scored_file": sha(V65) == rec65["file_sha256"]}
    v66, v65 = pd.read_parquet(V66), pd.read_parquet(V65)
    tmpl = pd.read_parquet(HERE / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v66[ID].equals(v65[ID]) and v66[ID].equals(tmpl[ID]))
    c["schema_equal_v65"] = bool(list(v66.columns) == list(v65.columns) and (v66.dtypes == v65.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v66, tmpl, None)]
    c["all_finite"] = bool(np.isfinite(v66[T]).all())
    feats = pd.read_parquet(HERE / ".cache/codex_v56/full_coverage_2026/features.parquet", columns=[ID, "airport", "date", "category"])
    cell = feats.airport + "/" + feats.date.str[5:7]
    stand = feats[cell.isin(["LSZH/07", "EHAM/07"]) & feats.category.eq("observed_at_assigned_stand")]
    base = H.current_base_2026()
    at65 = v65.set_index(ID)[T]
    chosen = stand[at65.loc[stand[ID]].to_numpy(float) == base.loc[stand[ID]].to_numpy(float)].reset_index(drop=True)
    c["set_rows"] = int(len(chosen))
    c["set_equals_build_rows"] = int(len(chosen)) == sum(build["checks"]["rows_by_cell"].values())
    diff = v66[T].to_numpy() != v65[T].to_numpy()
    c["rows_changed"] = int(diff.sum())
    c["every_change_in_set"] = bool(v66.loc[diff, ID].isin(chosen[ID]).all())
    c["outside_set_identical_to_v65"] = bool(np.array_equal(v66.loc[~v66[ID].isin(chosen[ID]), T].to_numpy(),
                                                            v65.loc[~v65[ID].isin(chosen[ID]), T].to_numpy()))
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet").set_index(ID).loc[chosen[ID]].reset_index()
    ctx["comparator"] = base.loc[ctx[ID]].to_numpy(float)
    mine = pd.concat([pd.read_parquet(p) for p in sorted((HERE / ".cache/research_440/stand_2026").glob("day_*.parquet"))]).set_index(ID)
    ctx["detector_taxi_s"] = mine.loc[ctx[ID], "detector_taxi_s"].to_numpy(float)
    corr = np.mean([R.joblib.load(S.CACHE / "models" / f"real_{s}.pkl").predict(ctx[S.COLS]) for s in S.SEEDS], axis=0)
    expect = np.maximum(ctx.comparator.to_numpy(float) + 0.5 * corr, 300.0).astype(v66[T].dtype)
    c["values_equal_recomputed_stand_model"] = bool(np.array_equal(v66.set_index(ID).loc[ctx[ID], T].to_numpy(), expect))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
