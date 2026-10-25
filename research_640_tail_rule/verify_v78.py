"""Independent pre-upload check of v78 (research_640_tail_rule/build_v78.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v77 and the template; submission.pruefe clean
2. every changed row is a C1+ partial row or a K2+ stand row; everything else equals v77
3. on the changed rows, v78 = max(base + (deployed + CatBoost) / 2, 300): base from the pre-NM file, inputs through
   the frozen v56 module (C1) and stand_experiment.inputs (stand), not the build's routes, seed models loaded directly
Run from the repository root:  python3 -u -B research_640_tail_rule/verify_v78.py
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
for p in (ROOT, ROOT / "research_360_integration"):
    sys.path.insert(0, str(p))
from research_320 import extract_episodes as X  # noqa: E402

MS = X.v56_modules()
from research_320 import stand_experiment as S  # noqa: E402
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402

ID, T, SEEDS = "MVT_ID_mvt", "TAXITIME_SEC_mvt", (177, 277, 377)
V78 = ROOT / ".cache/research_640/v78/candidate_v78.parquet"
V77 = ROOT / ".cache/research_590/v77/candidate_v77.parquet"
OUT = ROOT / "research_640_tail_rule/results/v78_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mean_pred(pattern, x):
    return np.mean([joblib.load(ROOT / pattern.format(seed=s)).predict(x) for s in SEEDS], axis=0)


def main():
    build = json.loads((ROOT / "research_640_tail_rule/results/v78_build.json").read_text())
    rec77 = json.loads((ROOT / "results/upbeat-goblin_v77.json").read_text())
    c = {"v78_sha_equals_build": sha(V78) == build["candidate_sha256"], "v77_is_uploaded_file": sha(V77) == rec77["file_sha256"]}
    v78, v77 = pd.read_parquet(V78), pd.read_parquet(V77)
    tmpl = pd.read_parquet(ROOT / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v78[ID].equals(v77[ID]) and v78[ID].equals(tmpl[ID]))
    c["schema_equal_v77"] = bool(list(v78.columns) == list(v77.columns) and (v78.dtypes == v77.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v78, tmpl, None)]
    a, b = v78.set_index(ID)[T], v77.set_index(ID)[T]
    changed = a.index[a.to_numpy() != b.to_numpy()]
    c["rows_changed"] = int(len(changed))
    base = H.current_base_2026()
    own = H.ownership_v62().set_index(ID).owner
    ctx = pd.read_parquet(ROOT / ".cache/codex_v56/partial_release/context.parquet")
    ctx["cell"] = ctx.airport + "/" + ctx.date.str[5:7]
    o = own.reindex(ctx[ID]).to_numpy()
    c1 = ctx[ctx.category.eq("partial_ground").to_numpy() & ((o == "partial") | (ctx.cell.eq("EDDF/07").to_numpy() & (o == "base")))].copy()
    st = ctx[ctx.category.eq("observed_at_assigned_stand")].copy()
    st = st[((st.cell.isin(["EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07"]) & own.reindex(st[ID]).eq("stand").to_numpy()) |
             (st.cell.isin(["LSZH/07", "EHAM/07", "LSZH/01", "EHAM/01"]) & own.reindex(st[ID]).eq("base").to_numpy()))].copy()
    det = pd.concat([pd.read_parquet(p, columns=[ID, "detector_taxi_s"]) for p in sorted(glob.glob(str(ROOT / ".cache/research_450/stand_2026/day_*.parquet")))]).drop_duplicates(ID).set_index(ID)
    st["detector_taxi_s"] = det.detector_taxi_s.reindex(st[ID]).to_numpy(float)
    c["c1_rows"], c["stand_rows"] = int(len(c1)), int(len(st))
    c["every_change_in_c1_or_stand"] = bool(pd.Index(changed).isin(pd.concat([c1[ID], st[ID]])).all())
    expect = {}
    for frame, x, dep, cat in ((c1, MS.inputs(c1, "real"), ".cache/research_550/c1_more/c1plus_{seed}.pkl", ".cache/research_640/cat_c1_EB1B2_{seed}.pkl"),
                               (st, S.inputs(st, "real"), ".cache/research_550/k2_more/k2plus_{seed}.pkl", ".cache/research_640/cat_stand_EB1B2_{seed}.pkl")):
        bb = base.loc[frame[ID]].to_numpy(float)
        expect.update(dict(zip(frame[ID], np.maximum(bb + (mean_pred(dep, x) + mean_pred(cat, x)) / 2, 300.0))))
    e = pd.Series(expect)
    c["values_equal"] = bool(np.array_equal(e.loc[changed].to_numpy().astype(a.dtype), a.loc[changed].to_numpy()))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
