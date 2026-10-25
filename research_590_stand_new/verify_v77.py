"""Independent pre-upload check of v77 (research_590_stand_new/build_v77.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v76 and the template; submission.pruefe clean
2. every changed row is a base-owned partial row of EGLL July or LEMD January; everything else equals v76
3. on the changed rows, v77 = max(base + correction, 300): base from the pre-NM file, inputs selected by the saved
   models' own feature names through the frozen v56 module, the correction recomputed from the seed models
Run from the repository root:  python3 -u -B research_590_stand_new/verify_v77.py
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
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_400_cells", ROOT / "research_590_stand_new"):
    sys.path.insert(0, str(p))
import build_v64 as B64  # noqa: E402
from research_320 import extract_episodes as X  # noqa: E402

MS = X.v56_modules()
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402

ID, TT = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
V77 = ROOT / ".cache/research_590/v77/candidate_v77.parquet"
V76 = ROOT / ".cache/research_590/v76/candidate_v76.parquet"
OUT = ROOT / "research_590_stand_new/results/v77_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((ROOT / "research_590_stand_new/results/v77_build.json").read_text())
    build76 = json.loads((ROOT / "research_590_stand_new/results/v76_build.json").read_text())
    c = {"v77_sha_equals_build": sha(V77) == build["candidate_sha256"], "v76_sha_equals_its_build": sha(V76) == build76["candidate_sha256"]}
    v77, v76 = pd.read_parquet(V77), pd.read_parquet(V76)
    tmpl = pd.read_parquet(ROOT / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v77[ID].equals(v76[ID]) and v77[ID].equals(tmpl[ID]))
    c["schema_equal_v76"] = bool(list(v77.columns) == list(v76.columns) and (v77.dtypes == v76.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v77, tmpl, None)]
    a, b = v77.set_index(ID)[TT], v76.set_index(ID)[TT]
    changed = a.index[a.to_numpy() != b.to_numpy()]
    c["rows_changed"] = int(len(changed))
    own = H.ownership_v62().set_index(ID).owner
    ctx = pd.read_parquet(ROOT / ".cache/codex_v56/partial_release/context.parquet")
    eg = ctx[ctx.airport.eq("EGLL") & ctx.date.str[5:7].eq("07") & ctx.category.eq("partial_ground")]
    eg = eg[own.reindex(eg[ID]).eq("base").to_numpy()]
    le = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / ".cache/research_590/days/LEMD_2026-01-*.parquet")))], ignore_index=True)
    le = B64.context_2026(le[le.category.eq("partial_ground") & own.reindex(le[ID]).eq("base").to_numpy()].reset_index(drop=True))
    frame = pd.concat([eg, le[eg.columns.intersection(le.columns)]], ignore_index=True)
    c["every_change_is_a_target_row"] = bool(pd.Index(changed).isin(frame[ID]).all())
    frame["comparator"] = H.current_base_2026().loc[frame[ID]].to_numpy(float)
    ms = [joblib.load(p) for p in sorted(glob.glob(str(ROOT / ".cache/research_590/drift_robust_c1/drc1_*.pkl")))]
    x = MS.inputs(frame, "real")[list(ms[0].feature_name_)]
    expect = pd.Series(np.maximum(frame.comparator.to_numpy(float) + np.mean([m.predict(x) for m in ms], axis=0), 300.0), index=frame[ID].to_numpy())
    c["values_equal"] = bool(np.array_equal(expect.loc[changed].to_numpy().astype(a.dtype), a.loc[changed].to_numpy()))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
