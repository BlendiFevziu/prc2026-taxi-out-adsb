"""Independent pre-upload check of v65 (research_400_cells/build_v65.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v64 and the template; submission.pruefe clean; all finite
2. the changed set is the EDDM July partial rows of the 2026 audit cache that sit at the base in v64;
   checked without the ownership table: every changed row equals the pre-NM base in v64, and every left-out partial row
   of the cell carries a value different from the base (the Munich detector's own value)
3. every v64 value in the set equals max(comparator + C1, 300), with C1 recomputed through the models and the frozen
   input builder directly (the build went through layers.correction)
4. outside the set identical to v64; v64 is NM-free and so is C1, so v65 is NM-free
Run from the repository root:  python3 -u -B research_400_cells/verify_v65.py
"""
import glob
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
sys.path.insert(0, str(HERE / "research_400_cells"))
import build_v64 as B  # noqa: E402  (its context recipe was proven against Codex's rows)
import harness as H  # noqa: E402

MS, SUB = B.MS, B.SUB
ID, T, SEEDS = "MVT_ID_mvt", "TAXITIME_SEC_mvt", (177, 277, 377)
V65 = HERE / ".cache/research_400/v65/candidate_v65.parquet"
V64 = HERE / ".cache/research_400/v64/candidate_v64.parquet"
OUT = HERE / "research_400_cells/results/v65_verification.json"
CELLS = ("EDDM/07",)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((HERE / "research_400_cells/results/v65_build.json").read_text())
    rec64 = json.loads((HERE / "results/upbeat-goblin_v64.json").read_text())
    c = {"v65_sha_equals_build": sha(V65) == build["candidate_sha256"], "v64_is_scored_file": sha(V64) == rec64["file_sha256"],
         "registration_rules_held": all(build["checks"]["rules"].values())}
    v64, v63 = pd.read_parquet(V65), pd.read_parquet(V64)
    tmpl = pd.read_parquet(HERE / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v64[ID].equals(v63[ID]) and v64[ID].equals(tmpl[ID]))
    c["schema_equal_v64"] = bool(list(v64.columns) == list(v63.columns) and (v64.dtypes == v63.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v64, tmpl, None)]
    c["all_finite"] = bool(np.isfinite(v64[T]).all())

    feats = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(B.AUDIT / "day_2026-*.parquet")))], ignore_index=True)
    feats["cell"] = feats.airport + "/" + feats.date.str[5:7]
    part = feats[feats.cell.isin(CELLS) & feats.category.eq("partial_ground")].drop(columns="cell").reset_index(drop=True)
    base = H.current_base_2026()
    at63 = v63.set_index(ID)[T]
    at_base = at63.loc[part[ID]].to_numpy(float) == base.loc[part[ID]].to_numpy(float)
    chosen = part[at_base].reset_index(drop=True)
    c["set_rows"] = int(len(chosen))
    c["left_out_rows_carry_own_value"] = int((~at_base).sum())
    c["set_equals_build_rows"] = int(len(chosen)) == build["checks"]["partial_rows_by_owner"]["base"]
    diff = v64[T].to_numpy() != v63[T].to_numpy()
    c["rows_changed"] = int(diff.sum())
    c["every_change_in_set"] = bool(v64.loc[diff, ID].isin(chosen[ID]).all())
    c["outside_set_identical_to_v64"] = bool(np.array_equal(v64.loc[~v64[ID].isin(chosen[ID]), T].to_numpy(),
                                                            v63.loc[~v63[ID].isin(chosen[ID]), T].to_numpy()))
    frame = B.context_2026(chosen)
    x = MS.inputs(frame, "real")
    corr = np.mean([joblib.load(HERE / f".cache/research_320/volume/models/c1_real_{s}.pkl").predict(x) for s in SEEDS], axis=0)
    expect = np.maximum(frame.comparator.to_numpy(float) + corr, 300.0).astype(v64[T].dtype)
    c["values_equal_recomputed_c1"] = bool(np.array_equal(v64.set_index(ID).loc[frame[ID], T].to_numpy(), expect))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
