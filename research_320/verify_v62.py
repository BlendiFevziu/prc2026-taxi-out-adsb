"""Independent pre-upload check of v62 (research_320/build_v62.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v61 and the template; submission.pruefe clean; all finite
2. only rows of the partial-track set differ from v61; that set is v56's overlay plus the NM-free build's partial
   rows, and the two overlap only where both sit at the comparator
3. every v62 value in the set equals max(comparator + mean C1 correction, 300), recomputed here
4. v61 is NM-free and v62 changes only partial-track rows, so v62 is NM-free
Run from the repository root:  python3 -u -B research_320/verify_v62.py
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
V62 = HERE / ".cache/research_320/v62/candidate_v62.parquet"
V61 = HERE / ".cache/research_320/nmfree_plus/candidate_nmfree_plus.parquet"
OUT = HERE / "research_320/results/v62_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((HERE / "research_320/results/v62_build.json").read_text())
    rec61 = json.loads((HERE / "results/upbeat-goblin_v61.json").read_text())
    c = {"v62_sha_equals_build": sha(V62) == build["candidate_sha256"], "v61_is_scored_file": sha(V61) == rec61["file_sha256"]}
    v62, v61 = pd.read_parquet(V62), pd.read_parquet(V61)
    tmpl = pd.read_parquet(HERE / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v62[ID].equals(v61[ID]) and v62[ID].equals(tmpl[ID]))
    c["schema_equal_v61"] = bool(list(v62.columns) == list(v61.columns) and (v62.dtypes == v61.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v62, tmpl, None)]
    c["all_finite"] = bool(np.isfinite(v62[T]).all())

    nm_build = json.loads((HERE / "research_320/results/nmfree_plus.json").read_text())
    upd = set(pd.read_parquet(HERE / ".cache/codex_v56/partial_release_v55/updates.parquet")[ID])
    ref = pd.read_parquet(HERE / "research_330_nmfree/private/external_nm_free_reference.parquet").set_index(ID)[T]
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet")
    ctx["cell"] = ctx.airport + "/" + ctx.date.str[5:7]
    cells = set(nm_build["cells"]["partial"])
    nm_day = ctx[ctx.date.str[8:10].isin(["06", "13", "20"]) & ctx.cell.isin(cells) & ctx.category.eq("partial_ground")]
    nm_part = set(nm_day.loc[ref.loc[nm_day[ID]].to_numpy(float) == nm_day.comparator.to_numpy(float), ID])
    c["nm_free_partial_rows"] = len(nm_part)
    c["nm_free_partial_rows_equal_build"] = len(nm_part) == nm_build["checks"]["rows_by_layer"]["partial"]
    overlap = upd & nm_part
    c["overlap_with_v56_overlay"] = len(overlap)
    c["overlap_rows_at_comparator_floor"] = bool(all(ref.loc[i] == 300.0 for i in overlap))
    allowed = upd | nm_part
    c["partial_set_rows"] = len(allowed)
    diff = v62[T].to_numpy() != v61[T].to_numpy()
    c["rows_changed"] = int(diff.sum())
    c["every_change_in_partial_set"] = bool(v62.loc[diff, ID].isin(allowed).all())

    rows = ctx.set_index(ID).loc[sorted(allowed)].reset_index()
    corr = np.mean([R.joblib.load(HERE / f".cache/research_320/volume/models/c1_real_{s}.pkl").predict(MS.inputs(rows, "real")) for s in SEEDS], axis=0)
    expect = np.maximum(rows.comparator.to_numpy(float) + corr, 300.0).astype(v62[T].dtype)
    c["values_equal_recomputed_c1"] = bool(np.array_equal(v62.set_index(ID).loc[rows[ID], T].to_numpy(), expect))
    c["outside_set_identical_to_v61"] = bool(np.array_equal(v62.loc[~v62[ID].isin(allowed), T].to_numpy(), v61.loc[~v61[ID].isin(allowed), T].to_numpy()))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
