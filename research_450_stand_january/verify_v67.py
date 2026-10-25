"""Independent pre-upload check of v67 (research_450_stand_january/build_v67.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v66 and the template; submission.pruefe clean; all finite
2. the changed set (rows at the 300 s floor, where every definition gives 300, are reported apart and must be unchanged),
   from Codex's 2026 FEATURE table (the build used his context table): stand-observed rows of the nine
   stand cells whose v66 value is either the pre-NM base (new rows) or differs from it (rows the stand model already
   corrected), excluding Munich/Amsterdam detector rows by value (a detector value is neither); every change lies in
   the set and everything outside is identical to v66
3. every v67 value in the set equals max(base + 1.0 x mean stand correction, 300), with the comparator from the pre-NM
   base file and the 36th input from the build's cached re-extraction
Run from the repository root:  python3 -u -B research_450_stand_january/verify_v67.py
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
CELLS = ("EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07", "LSZH/07", "EHAM/07", "LSZH/01", "EHAM/01")
V67 = HERE / ".cache/research_450/v67/candidate_v67.parquet"
V66 = HERE / ".cache/research_440/v66/candidate_v66.parquet"
OUT = HERE / "research_450_stand_january/results/v67_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((HERE / "research_450_stand_january/results/v67_build.json").read_text())
    rec66 = json.loads((HERE / "results/upbeat-goblin_v66.json").read_text())
    c = {"v67_sha_equals_build": sha(V67) == build["candidate_sha256"], "v66_is_scored_file": sha(V66) == rec66["file_sha256"]}
    v67, v66 = pd.read_parquet(V67), pd.read_parquet(V66)
    tmpl = pd.read_parquet(HERE / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v67[ID].equals(v66[ID]) and v67[ID].equals(tmpl[ID]))
    c["schema_equal_v66"] = bool(list(v67.columns) == list(v66.columns) and (v67.dtypes == v66.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v67, tmpl, None)]
    c["all_finite"] = bool(np.isfinite(v67[T]).all())
    feats = pd.read_parquet(HERE / ".cache/codex_v56/full_coverage_2026/features.parquet", columns=[ID, "airport", "date", "category"])
    feats["cell"] = feats.airport + "/" + feats.date.str[5:7]
    stand = feats[feats.cell.isin(CELLS) & feats.category.eq("observed_at_assigned_stand")].reset_index(drop=True)
    base = H.current_base_2026()
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet").set_index(ID).loc[stand[ID]].reset_index()
    ctx["comparator"] = base.loc[ctx[ID]].to_numpy(float)
    mine = pd.concat([pd.read_parquet(p) for p in sorted((HERE / ".cache/research_450/stand_2026").glob("day_*.parquet"))]).set_index(ID)
    ctx = ctx[ctx[ID].isin(mine.index)].reset_index(drop=True)
    ctx["detector_taxi_s"] = mine.loc[ctx[ID], "detector_taxi_s"].to_numpy(float)
    models = [R.joblib.load(S.CACHE / "models" / f"real_{s}.pkl") for s in S.SEEDS]
    corr = np.mean([m.predict(ctx[S.COLS]) for m in models], axis=0)
    at66 = v66.set_index(ID).loc[ctx[ID], T].to_numpy(float)
    b = ctx.comparator.to_numpy(float)
    new_rows = at66 == b                                                    # still at the base in v66
    old_rows = at66 == np.maximum(b + 0.5 * corr, 300.0).astype(v66[T].dtype)   # carrying the stand model at 0.5
    floor = (b == 300.0) & (np.maximum(b + corr, 300.0) == 300.0)       # at the floor every definition gives 300
    c["floor_rows_ambiguous"] = int((floor & (new_rows | old_rows)).sum())
    c["floor_rows_unchanged"] = bool(np.array_equal(v67.set_index(ID).loc[ctx[ID][floor], T].to_numpy(), at66[floor]))
    new_rows, old_rows = new_rows & ~floor, old_rows & ~floor
    chosen = ctx[new_rows | old_rows].reset_index(drop=True)
    c["set_rows"] = int(len(chosen))
    c["set_plus_changed_floor_rows_within_build"] = int(len(chosen)) <= sum(build["checks"]["existing_rows_by_cell"].values()) + sum(build["checks"]["new_rows_by_cell"].values())
    c["build_rows_minus_set"] = sum(build["checks"]["existing_rows_by_cell"].values()) + sum(build["checks"]["new_rows_by_cell"].values()) - int(len(chosen))
    diff = v67[T].to_numpy() != v66[T].to_numpy()
    c["rows_changed"] = int(diff.sum())
    c["every_change_in_set"] = bool(v67.loc[diff, ID].isin(chosen[ID]).all())
    c["outside_set_identical_to_v66"] = bool(np.array_equal(v67.loc[~v67[ID].isin(chosen[ID]), T].to_numpy(),
                                                            v66.loc[~v66[ID].isin(chosen[ID]), T].to_numpy()))
    keep = (new_rows | old_rows)
    expect = np.maximum(b[keep] + corr[keep], 300.0).astype(v67[T].dtype)
    c["values_equal_weight_one"] = bool(np.array_equal(v67.set_index(ID).loc[chosen[ID], T].to_numpy(), expect))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
