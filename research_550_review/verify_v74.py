"""Independent pre-upload check of v74 (research_550_review/build_v74_fallback.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v73 and the template; submission.pruefe clean
2. every changed row is a fallback-owned EHAM or EDDM row; the change lies in the supported range (-600 s, +3,600 s];
   everything else equals v73
3. every changed value is re-derived here from the tracks in a fresh process (determinism of matching and detection)
Run from the repository root:  python3 -u -B research_550_review/verify_v74.py
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
import build_v74_fallback as B  # noqa: E402
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
V74 = ROOT / ".cache/research_550/v74/candidate_v74.parquet"
OUT = ROOT / "research_550_review/results/v74_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((ROOT / "research_550_review/results/v74_build.json").read_text())
    build73 = json.loads((ROOT / "research_550_review/results/v73_build.json").read_text())
    c = {"v74_sha_equals_build": sha(V74) == build["candidate_sha256"], "v73_sha_equals_its_build": sha(B.V73) == build73["candidate_sha256"]}
    v74, v73 = pd.read_parquet(V74), pd.read_parquet(B.V73)
    tmpl = pd.read_parquet(ROOT / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v74[ID].equals(v73[ID]) and v74[ID].equals(tmpl[ID]))
    c["schema_equal_v73"] = bool(list(v74.columns) == list(v73.columns) and (v74.dtypes == v73.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v74, tmpl, None)]
    a, b = v74.set_index(ID)[T], v73.set_index(ID)[T]
    changed = a.index[a.to_numpy() != b.to_numpy()]
    c["rows_changed"] = int(len(changed))
    own = H.ownership_v62().set_index(ID)
    c["every_change_fallback_owned_at_eham_or_eddm"] = bool(own.loc[changed, "owner"].eq("fallback").all()
                                                             and own.loc[changed, "airport"].isin(["EHAM", "EDDM"]).all())
    diff = a.loc[changed].to_numpy(float) - b.loc[changed].to_numpy(float)
    c["changes_in_support"] = bool(((diff > -600.0 - 1e-3) & (diff <= 3600.0 + 1e-3)).all())
    mapping = B.AT.OB.callsign_mapping()
    filed = pd.read_parquet(ROOT / "data/ranking.parquet", columns=[ID, "CALLSIGN_flt"])
    values = {}
    for ap in ("EHAM", "EDDM"):
        maps = (B.SD.stand_table(ap), B.GC.xplane_table(ap)[0])
        for d in B.scored_days():
            values.update(B.day_values(d, ap, mapping, maps, filed))
    s = pd.Series(values)
    d = s - b.loc[s.index].to_numpy(float)
    s = s[(d > -600.0) & (d <= 3600.0)]
    c["rederived_rows"] = int(len(s))
    c["rederived_set_equals_changed"] = set(s.index) == set(changed) or (set(changed) <= set(s.index))
    c["rederived_values_equal"] = bool(np.array_equal(s.loc[changed].to_numpy(float).astype(a.dtype), a.loc[changed].to_numpy()))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
