"""Independent pre-upload check of v75 (research_550_review/build_v75.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v74 and the template; submission.pruefe clean
2. every changed row is a fallback-owned LEBL or LSZH row, its change lies in (-600 s, +3,600 s]; all else equals v74
3. every changed value is re-derived here from the tracks in a fresh process
Run from the repository root:  python3 -u -B research_550_review/verify_v75.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_550_review"):
    sys.path.insert(0, str(p))
import build_v75 as B  # noqa: E402

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
V75 = ROOT / ".cache/research_550/v75/candidate_v75.parquet"
OUT = ROOT / "research_550_review/results/v75_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((ROOT / "research_550_review/results/v75_build.json").read_text())
    build74 = json.loads((ROOT / "research_550_review/results/v74_build.json").read_text())
    c = {"v75_sha_equals_build": sha(V75) == build["candidate_sha256"], "v74_sha_equals_its_build": sha(B.V74) == build74["candidate_sha256"]}
    v75, v74 = pd.read_parquet(V75), pd.read_parquet(B.V74)
    tmpl = pd.read_parquet(ROOT / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v75[ID].equals(v74[ID]) and v75[ID].equals(tmpl[ID]))
    c["schema_equal_v74"] = bool(list(v75.columns) == list(v74.columns) and (v75.dtypes == v74.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in B.SUB.pruefe(v75, tmpl, None)]
    a, b = v75.set_index(ID)[T], v74.set_index(ID)[T]
    changed = a.index[a.to_numpy() != b.to_numpy()]
    c["rows_changed"] = int(len(changed))
    own = B.H.ownership_v62().set_index(ID)
    c["every_change_fallback_owned_at_lebl_or_lszh"] = bool(own.loc[changed, "owner"].eq("fallback").all()
                                                             and own.loc[changed, "airport"].isin(list(B.AIRPORTS)).all())
    diff = a.loc[changed].to_numpy(float) - b.loc[changed].to_numpy(float)
    c["changes_in_support"] = bool(((diff > -600.0 - 1e-3) & (diff <= 3600.0 + 1e-3)).all())
    vals = {}
    for ap, v in B.values_2026().items():
        vals.update(v)
    s = pd.Series(vals, dtype=float)
    c["changed_rows_rederived"] = bool(set(changed) <= set(s.index))
    c["rederived_values_equal"] = bool(np.array_equal(s.loc[changed].to_numpy(float).astype(a.dtype), a.loc[changed].to_numpy()))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
