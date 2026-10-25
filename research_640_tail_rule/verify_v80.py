"""Independent pre-upload check of v80 (research_640_tail_rule/build_v80.py), in a fresh process. No upload.
Every changed row must be a base-owned LFPG partial row or an eham_january row; everything else equals v79; values
recomputed with the base from the pre-NM file and each model's own saved feature names through the frozen v56 module.
Run from the repository root:  python3 -u -B research_640_tail_rule/verify_v80.py
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_400_cells", ROOT / "research_640_tail_rule"):
    sys.path.insert(0, str(p))
import build_v80 as B  # noqa: E402
from research_320 import extract_episodes as X  # noqa: E402

MS = X.v56_modules()
ID, T, L, H = "MVT_ID_mvt", "TAXITIME_SEC_mvt", B.L, B.H
V80 = ROOT / ".cache/research_640/v80/candidate_v80.parquet"
OUT = ROOT / "research_640_tail_rule/results/v80_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pred(pattern, frame, route):
    ms = [joblib.load(ROOT / pattern.format(seed=s)) for s in L.SEEDS]
    return np.mean([m.predict(route(frame)[list(m.feature_name_)]) for m in ms], axis=0)


def main():
    build = json.loads((ROOT / "research_640_tail_rule/results/v80_build.json").read_text())
    c = {"v80_sha_equals_build": sha(V80) == build["candidate_sha256"]}
    v80, v79 = pd.read_parquet(V80), pd.read_parquet(B.V79)
    tmpl = pd.read_parquet(ROOT / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v80[ID].equals(v79[ID]) and v80[ID].equals(tmpl[ID]))
    c["pruefe_errors"] = [str(e) for e in B.SUB.pruefe(v80, tmpl, None)]
    a, b = v80.set_index(ID)[T], v79.set_index(ID)[T]
    changed = a.index[a.to_numpy() != b.to_numpy()]
    c["rows_changed"] = int(len(changed))
    base = H.current_base_2026()
    pa, eh = B.paris_rows(), B.eham_rows()
    route = lambda f: MS.inputs(f, "real")
    exp_a = np.maximum(base.loc[pa[ID]].to_numpy(float) + pred(".cache/research_590/drift_robust_c1/drc1_{seed}.pkl", pa, route), 300.0)
    be = base.loc[eh[ID]].to_numpy(float)
    addon = np.mean([joblib.load(ROOT / L.LAYERS["eham_january"].models.format(seed=s)).predict(eh[list(L.CONTEXT) + list(L.EHAM_TRACK) + ["is_july"]])
                     for s in L.SEEDS], axis=0)
    plus = pred(".cache/research_550/c1_more/c1plus_{seed}.pkl", eh, route)
    exp_e = np.maximum(be + (addon + plus) / 2, 300.0)
    e = pd.Series({**dict(zip(pa[ID], exp_a)), **dict(zip(eh[ID], exp_e))})
    c["every_change_is_a_target_row"] = bool(pd.Index(changed).isin(e.index).all())
    c["values_equal"] = bool(np.array_equal(e.loc[changed].to_numpy().astype(a.dtype), a.loc[changed].to_numpy()))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
