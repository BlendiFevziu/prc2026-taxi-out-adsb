"""Independent pre-upload check of v79 (research_640_tail_rule/build_v79.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v78 and the template; submission.pruefe clean
2. every changed row belongs to one of the three groups; everything else equals v78
3. on the changed rows, v79 = max(base + (model + CatBoost) / 2, 300): base from the pre-NM file (not the context
   column), each model's inputs selected by its own saved feature names, seed models loaded directly
Run from the repository root:  python3 -u -B research_640_tail_rule/verify_v79.py
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_400_cells", ROOT / "research_590_stand_new", ROOT / "research_550_review",
          ROOT / "research_640_tail_rule"):
    sys.path.insert(0, str(p))
import build_v79 as B  # noqa: E402

S, MS, L, H, SUB = B.S, B.MS, B.L, B.H, B.SUB
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
V79 = ROOT / ".cache/research_640/v79/candidate_v79.parquet"
OUT = ROOT / "research_640_tail_rule/results/v79_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pred(pattern, frame, route):
    ms = [joblib.load(ROOT / pattern.format(seed=s)) for s in L.SEEDS]
    names = list(ms[0].feature_name_) if hasattr(ms[0], "feature_name_") else list(ms[0].feature_names_)
    x = route(frame, "real")[names]
    return np.mean([m.predict(x) for m in ms], axis=0)


def main():
    build = json.loads((ROOT / "research_640_tail_rule/results/v79_build.json").read_text())
    build78 = json.loads((ROOT / "research_640_tail_rule/results/v78_build.json").read_text())
    c = {"v79_sha_equals_build": sha(V79) == build["candidate_sha256"], "v78_sha_equals_its_build": sha(B.V78) == build78["candidate_sha256"]}
    v79, v78 = pd.read_parquet(V79), pd.read_parquet(B.V78)
    tmpl = pd.read_parquet(ROOT / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v79[ID].equals(v78[ID]) and v79[ID].equals(tmpl[ID]))
    c["schema_equal_v78"] = bool(list(v79.columns) == list(v78.columns) and (v79.dtypes == v78.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v79, tmpl, None)]
    a, b = v79.set_index(ID)[T], v78.set_index(ID)[T]
    changed = a.index[a.to_numpy() != b.to_numpy()]
    c["rows_changed"] = int(len(changed))
    eg, le = B.B77.rows_2026()
    le = B.B64.context_2026(le)
    groups = [(B.stand_rows(), S.inputs, ".cache/research_590/drift_robust/dr_{seed}.pkl", ".cache/research_640/cat_dr_stand_{seed}.pkl"),
              (pd.concat([eg, le[eg.columns.intersection(le.columns)]], ignore_index=True), MS.inputs,
               ".cache/research_590/drift_robust_c1/drc1_{seed}.pkl", ".cache/research_640/cat_dr_c1_{seed}.pkl"),
              (B.old_c1_rows(), MS.inputs, ".cache/research_320/volume/models/c1_real_{seed}.pkl", ".cache/research_640/cat_c1_E_{seed}.pkl")]
    base = H.current_base_2026()
    expect = {}
    for frame, route, mp, cp in groups:
        bb = base.loc[frame[ID]].to_numpy(float)
        expect.update(dict(zip(frame[ID], np.maximum(bb + (pred(mp, frame, route) + pred(cp, frame, route)) / 2, 300.0))))
    e = pd.Series(expect)
    c["every_change_in_groups"] = bool(pd.Index(changed).isin(e.index).all())
    c["values_equal"] = bool(np.array_equal(e.loc[changed].to_numpy().astype(a.dtype), a.loc[changed].to_numpy()))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
