"""Independent pre-upload check of v81 (research_650_eham_cat/build_v81.py), in a fresh process. No upload.
Every changed row must be an eham_january row; everything else equals v80; values recomputed with the base from the
pre-NM file and each model's own saved feature names (C1+ through the frozen v56 module).
Run from the repository root:  python3 -u -B research_650_eham_cat/verify_v81.py
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_400_cells", ROOT / "research_640_tail_rule", ROOT / "research_650_eham_cat"):
    sys.path.insert(0, str(p))
import build_v81 as B  # noqa: E402
from research_320 import extract_episodes as X  # noqa: E402

MS = X.v56_modules()
ID, T, L, H = "MVT_ID_mvt", "TAXITIME_SEC_mvt", B.L, B.H
V81 = ROOT / ".cache/research_650/v81/candidate_v81.parquet"
OUT = ROOT / "research_650_eham_cat/results/v81_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def by_names(pattern, x):
    ms = [joblib.load(ROOT / pattern.format(seed=s)) for s in L.SEEDS]
    return np.mean([m.predict(x[list(getattr(m, "feature_name_", None) or m.feature_names_)]) for m in ms], axis=0)


def main():
    build = json.loads((ROOT / "research_650_eham_cat/results/v81_build.json").read_text())
    c = {"v81_sha_equals_build": sha(V81) == build["candidate_sha256"]}
    v81, v80 = pd.read_parquet(V81), pd.read_parquet(B.V80)
    tmpl = pd.read_parquet(ROOT / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v81[ID].equals(v80[ID]) and v81[ID].equals(tmpl[ID]))
    c["pruefe_errors"] = [str(e) for e in B.SUB.pruefe(v81, tmpl, None)]
    a, b = v81.set_index(ID)[T], v80.set_index(ID)[T]
    changed = a.index[a.to_numpy() != b.to_numpy()]
    c["rows_changed"] = int(len(changed))
    eh = B.B80.eham_rows()
    be = H.current_base_2026().loc[eh[ID]].to_numpy(float)
    addon = by_names(L.LAYERS["eham_january"].models, eh)
    cat = by_names(".cache/research_650/cat_eham_{seed}.pkl", eh)
    plus = by_names(".cache/research_550/c1_more/c1plus_{seed}.pkl", MS.inputs(eh, "real"))
    e = pd.Series(np.maximum(be + ((addon + cat) / 2 + plus) / 2, 300.0), index=eh[ID].to_numpy())
    c["every_change_is_a_target_row"] = bool(pd.Index(changed).isin(e.index).all())
    c["values_equal"] = bool(np.array_equal(e.loc[changed].to_numpy().astype(a.dtype), a.loc[changed].to_numpy()))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
