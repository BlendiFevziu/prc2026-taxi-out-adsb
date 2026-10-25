"""Independent pre-upload check of v63 (research_360_integration/build_v63.py), in a fresh process. No upload.

Checked through other routes than the build where one exists:
1. hashes; ids, order and schema equal v62 and the template; submission.pruefe clean; all finite
2. the changed set is recomputed from Codex's 62-day FEATURE table (the build used his context table): EDDF July
   partial_ground; only rows of that set differ from v62
3. every row of the set is owned by the BASE in v62 (harness.ownership_v62, Codex's provenance), and the context
   table's comparator equals the pre-NM base (harness.current_base_2026) on it, so the correction starts from v62's own
   value
4. every v63 value in the set equals max(comparator + mean C1 correction, 300), recomputed here with the models the
   capacity study recorded
5. v62 is NM-free and v63 changes only these rows with an NM-free model, so v63 is NM-free
Run from the repository root:  python3 -u -B research_360_integration/verify_v63.py
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
from research_320 import extract_episodes as X  # noqa: E402

MS = X.v56_modules()
R = MS.R
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402

ID, T, SEEDS, CELL = "MVT_ID_mvt", "TAXITIME_SEC_mvt", (177, 277, 377), "EDDF/07"
V63 = HERE / ".cache/research_360/v63/candidate_v63.parquet"
V62 = HERE / ".cache/research_320/v62/candidate_v62.parquet"
MODELS = HERE / ".cache/research_320/volume/models"
OUT = HERE / "research_360_integration/results/v63_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((HERE / "research_360_integration/results/v63_build.json").read_text())
    rec62 = json.loads((HERE / "results/upbeat-goblin_v62.json").read_text())
    cap = json.loads((HERE / "research_320/results/capacity2.json").read_text())
    c = {"v63_sha_equals_build": sha(V63) == build["candidate_sha256"], "v62_is_scored_file": sha(V62) == rec62["file_sha256"],
         "c1_models_as_tested": all(sha(MODELS / f"c1_real_{s}.pkl") == cap["model_hashes"][f"c1_real_{s}.pkl"] for s in SEEDS)}
    v63, v62 = pd.read_parquet(V63), pd.read_parquet(V62)
    tmpl = pd.read_parquet(HERE / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v63[ID].equals(v62[ID]) and v63[ID].equals(tmpl[ID]))
    c["schema_equal_v62"] = bool(list(v63.columns) == list(v62.columns) and (v63.dtypes == v62.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v63, tmpl, None)]
    c["all_finite"] = bool(np.isfinite(v63[T]).all())

    feat = pd.read_parquet(HERE / ".cache/codex_v56/full_coverage_2026/features.parquet", columns=[ID, "airport", "date", "category"])
    allowed = set(feat.loc[(feat.airport + "/" + feat.date.str[5:7]).eq(CELL) & feat.category.eq("partial_ground"), ID])
    c["set_rows"] = len(allowed)
    diff = v63[T].to_numpy() != v62[T].to_numpy()
    c["rows_changed"] = int(diff.sum())
    c["every_change_in_set"] = bool(v63.loc[diff, ID].isin(allowed).all())
    c["outside_set_identical_to_v62"] = bool(np.array_equal(v63.loc[~v63[ID].isin(allowed), T].to_numpy(),
                                                            v62.loc[~v62[ID].isin(allowed), T].to_numpy()))

    own = H.ownership_v62()
    own = own.set_index(ID).owner if isinstance(own, pd.DataFrame) else own
    c["set_owned_by_base_in_v62"] = bool(own.loc[sorted(allowed)].eq("base").all())
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet")
    rows = ctx.set_index(ID).loc[sorted(allowed)].reset_index()
    base = H.current_base_2026().loc[rows[ID]].to_numpy(float)
    c["comparator_equals_pre_nm_base"] = bool(np.array_equal(rows.comparator.to_numpy(float), base))
    c["v62_equals_comparator_on_set"] = bool(np.array_equal(v62.set_index(ID).loc[rows[ID], T].to_numpy(float), base))

    corr = np.mean([R.joblib.load(MODELS / f"c1_real_{s}.pkl").predict(MS.inputs(rows, "real")) for s in SEEDS], axis=0)
    expect = np.maximum(rows.comparator.to_numpy(float) + corr, 300.0).astype(v63[T].dtype)
    c["values_equal_recomputed_c1"] = bool(np.array_equal(v63.set_index(ID).loc[rows[ID], T].to_numpy(), expect))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
