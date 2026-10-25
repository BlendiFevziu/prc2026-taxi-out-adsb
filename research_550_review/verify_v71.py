"""Independent pre-upload check of v71 (research_550_review/build_v71.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v70 and the template; submission.pruefe clean
2. every changed row is an observed_at_assigned_stand row of the stand model's nine cells; everything else equals v70
3. on the changed rows, v70 = max(base + deployed correction, 300) and v71 = max(base + K2 correction, 300), both
   corrections recomputed through stand_experiment.inputs (not the build's column slice) and the base read from the
   pre-NM base file (not the context table)
Run from the repository root:  python3 -u -B research_550_review/verify_v71.py
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
from research_320 import stand_experiment as S  # noqa: E402
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
V71 = HERE / ".cache/research_550/v71/candidate_v71.parquet"
V70 = HERE / ".cache/research_510/v70/candidate_v70.parquet"
OUT = HERE / "research_550_review/results/v71_verification.json"
CELLS = ("EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07", "LSZH/07", "EHAM/07", "LSZH/01", "EHAM/01")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((HERE / "research_550_review/results/v71_build.json").read_text())
    build70 = json.loads((HERE / "research_510_rome_classifier/results/v70_build.json").read_text())
    c = {"v71_sha_equals_build": sha(V71) == build["candidate_sha256"], "v70_sha_equals_its_build": sha(V70) == build70["candidate_sha256"]}
    v71, v70 = pd.read_parquet(V71), pd.read_parquet(V70)
    tmpl = pd.read_parquet(HERE / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v71[ID].equals(v70[ID]) and v71[ID].equals(tmpl[ID]))
    c["schema_equal_v70"] = bool(list(v71.columns) == list(v70.columns) and (v71.dtypes == v70.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v71, tmpl, None)]
    a, b = v71.set_index(ID)[T], v70.set_index(ID)[T]
    changed = a.index[a.to_numpy() != b.to_numpy()]
    c["rows_changed"] = int(len(changed))
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet")
    ctx["cell"] = ctx.airport + "/" + ctx.date.str[5:7]
    stand_ids = set(ctx.loc[ctx.category.eq("observed_at_assigned_stand") & ctx.cell.isin(CELLS), ID])
    c["every_change_is_a_stand_row"] = bool(pd.Index(changed).isin(list(stand_ids)).all())
    f = ctx[ctx[ID].isin(changed)].copy()
    det = pd.concat([pd.read_parquet(p, columns=[ID, "detector_taxi_s"])
                     for p in sorted(glob.glob(str(HERE / ".cache/research_450/stand_2026/day_*.parquet")))]).drop_duplicates(ID).set_index(ID)
    f["detector_taxi_s"] = det.detector_taxi_s.loc[f[ID]].to_numpy(float)
    f["comparator"] = H.current_base_2026().loc[f[ID]].to_numpy(float)
    x = S.inputs(f, "real")
    corr = lambda pat: np.mean([joblib.load(pat.format(seed=s)).predict(x) for s in S.SEEDS], axis=0)
    dep = corr(str(HERE / ".cache/research_320/stand/models/real_{seed}.pkl"))
    k2 = corr(str(HERE / ".cache/research_320/stand_capacity/models/k2_real_{seed}.pkl"))
    base = f.comparator.to_numpy(float)
    c["v70_equals_deployed"] = bool(np.array_equal(np.maximum(base + dep, 300.0).astype(b.dtype), b.loc[f[ID]].to_numpy()))
    c["v71_equals_k2"] = bool(np.array_equal(np.maximum(base + k2, 300.0).astype(a.dtype), a.loc[f[ID]].to_numpy()))
    c["unchanged_stand_rows"] = int(build["checks"]["rows"] - len(changed))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
