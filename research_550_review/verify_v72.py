"""Independent pre-upload check of v72 (research_550_review/build_v72.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v71 and the template; submission.pruefe clean
2. every changed row is a partial_ground row in C1's eight cells; everything else equals v71
3. on the changed rows, v71 = max(base + C1, 300) and v72 = max(base + C1+, 300), both recomputed by loading the seed
   models directly and building inputs through the frozen v56 module (MS.inputs, not layers.inputs), with the base
   from the pre-NM base file
Run from the repository root:  python3 -u -B research_550_review/verify_v72.py
"""
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
from research_320 import extract_episodes as X  # noqa: E402

MS = X.v56_modules()
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
V72 = HERE / ".cache/research_550/v72/candidate_v72.parquet"
V71 = HERE / ".cache/research_550/v71/candidate_v71.parquet"
OUT = HERE / "research_550_review/results/v72_verification.json"
CELLS = ("EDDF/01", "EDDF/07", "EGLL/01", "EHAM/07", "LEBL/01", "LEBL/07", "LSZH/01", "LSZH/07")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((HERE / "research_550_review/results/v72_build.json").read_text())
    build71 = json.loads((HERE / "research_550_review/results/v71_build.json").read_text())
    c = {"v72_sha_equals_build": sha(V72) == build["candidate_sha256"], "v71_sha_equals_its_build": sha(V71) == build71["candidate_sha256"]}
    v72, v71 = pd.read_parquet(V72), pd.read_parquet(V71)
    tmpl = pd.read_parquet(HERE / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v72[ID].equals(v71[ID]) and v72[ID].equals(tmpl[ID]))
    c["schema_equal_v71"] = bool(list(v72.columns) == list(v71.columns) and (v72.dtypes == v71.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v72, tmpl, None)]
    a, b = v72.set_index(ID)[T], v71.set_index(ID)[T]
    changed = a.index[a.to_numpy() != b.to_numpy()]
    c["rows_changed"] = int(len(changed))
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet")
    ctx["cell"] = ctx.airport + "/" + ctx.date.str[5:7]
    allowed = set(ctx.loc[ctx.category.eq("partial_ground") & ctx.cell.isin(CELLS), ID])
    c["every_change_is_a_c1_cell_partial_row"] = bool(pd.Index(changed).isin(list(allowed)).all())
    f = ctx[ctx[ID].isin(changed)].copy()
    f["comparator"] = H.current_base_2026().loc[f[ID]].to_numpy(float)
    x = MS.inputs(f, "real")
    old = np.mean([joblib.load(HERE / L.LAYERS["partial"].models.format(seed=s)).predict(x) for s in L.SEEDS], axis=0)
    new = np.mean([joblib.load(HERE / f".cache/research_550/c1_more/c1plus_{s}.pkl").predict(x) for s in L.SEEDS], axis=0)
    base = f.comparator.to_numpy(float)
    c["v71_equals_c1"] = bool(np.array_equal(np.maximum(base + old, 300.0).astype(b.dtype), b.loc[f[ID]].to_numpy()))
    c["v72_equals_c1plus"] = bool(np.array_equal(np.maximum(base + new, 300.0).astype(a.dtype), a.loc[f[ID]].to_numpy()))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
