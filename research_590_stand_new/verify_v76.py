"""Independent pre-upload check of v76 (research_590_stand_new/build_v76.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v75 and the template; submission.pruefe clean
2. every changed row is a base-owned stand row of an admitted cell; everything else equals v75
3. on the changed rows, v76 = max(base + correction, 300): the base read from the pre-NM base file, the inputs chosen by
   the saved models' own feature names (not the build's column list), the correction recomputed from the seed models
Run from the repository root:  python3 -u -B research_590_stand_new/verify_v76.py
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
import extract as E  # noqa: E402
from research_320 import stand_experiment as S  # noqa: E402
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402

ID, TT = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
V76 = ROOT / ".cache/research_590/v76/candidate_v76.parquet"
V75 = ROOT / ".cache/research_550/v75/candidate_v75.parquet"
OUT = ROOT / "research_590_stand_new/results/v76_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((ROOT / "research_590_stand_new/results/v76_build.json").read_text())
    test = json.loads((ROOT / "research_590_stand_new/results/drift_robust.json").read_text())
    rec75 = json.loads((ROOT / "results/upbeat-goblin_v75.json").read_text())
    c = {"v76_sha_equals_build": sha(V76) == build["candidate_sha256"], "v75_is_uploaded_file": sha(V75) == rec75["file_sha256"]}
    v76, v75 = pd.read_parquet(V76), pd.read_parquet(V75)
    tmpl = pd.read_parquet(ROOT / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v76[ID].equals(v75[ID]) and v76[ID].equals(tmpl[ID]))
    c["schema_equal_v75"] = bool(list(v76.columns) == list(v75.columns) and (v76.dtypes == v75.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v76, tmpl, None)]
    a, b = v76.set_index(ID)[TT], v75.set_index(ID)[TT]
    changed = a.index[a.to_numpy() != b.to_numpy()]
    c["rows_changed"] = int(len(changed))
    cells = test["admitted"]
    own = H.ownership_v62().set_index(ID).owner
    frames = []
    for ap in sorted({x[:4] for x in cells}):
        dep = E.departures_2026(ap)
        r = E.load(ap, sorted(dep.date.unique()) if ap != "EGLL" else sorted(d for d in dep.date.unique() if d[5:7] == "07"))
        r = r[r.category.eq("observed_at_assigned_stand") & (r.airport + "/" + r.date.str[5:7]).isin(cells) & own.reindex(r[ID]).eq("base").to_numpy()]
        frames.append(r)
    rows = pd.concat(frames, ignore_index=True)
    c["every_change_is_an_admitted_stand_row"] = bool(pd.Index(changed).isin(rows[ID]).all())
    frame = B64.context_2026(rows)
    frame["comparator"] = H.current_base_2026().loc[frame[ID]].to_numpy(float)
    ms = [joblib.load(p) for p in sorted(glob.glob(str(ROOT / ".cache/research_590/drift_robust/dr_*.pkl")))]
    x = S.inputs(frame, "real")[list(ms[0].feature_name_)]
    corr = np.mean([m.predict(x) for m in ms], axis=0)
    expect = pd.Series(np.maximum(frame.comparator.to_numpy(float) + corr, 300.0), index=frame[ID].to_numpy())
    c["values_equal"] = bool(np.array_equal(expect.loc[changed].to_numpy().astype(a.dtype), a.loc[changed].to_numpy()))
    c["inputs_used"] = len(ms[0].feature_name_)
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
