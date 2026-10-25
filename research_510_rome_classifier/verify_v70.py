"""Independent pre-upload check of v70 (research_510_rome_classifier/build_v70.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v69 and the template; submission.pruefe clean
2. the slice recomputed with the standard library (zoneinfo) from the ranking file: Rome, no AOBT_3, 1 h < delay <= 8 h,
   and not (delay > 8 h without a crossing); every changed row lies in it and everything outside equals v69
3. the classifier refitted from scratch in this process reproduces the build's values exactly (determinism of the fit)
Run from the repository root:  python3 -u -B research_510_rome_classifier/verify_v70.py
"""
import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import submission as SUB  # noqa: E402

ID, T, ROME = "MVT_ID_mvt", "TAXITIME_SEC_mvt", ZoneInfo("Europe/Rome")
V70 = ROOT / ".cache/research_510/v70/candidate_v70.parquet"
V69 = ROOT / ".cache/research_500/v69/candidate_v69.parquet"
OUT = HERE / "results/v70_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((HERE / "results/v70_build.json").read_text())
    build69 = json.loads((ROOT / "research_500_midnight/results/v69_build.json").read_text())
    c = {"v70_sha_equals_build": sha(V70) == build["candidate_sha256"], "v69_sha_equals_its_build": sha(V69) == build69["candidate_sha256"]}
    v70, v69 = pd.read_parquet(V70), pd.read_parquet(V69)
    tmpl = pd.read_parquet(ROOT / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v70[ID].equals(v69[ID]) and v70[ID].equals(tmpl[ID]))
    c["schema_equal_v69"] = bool(list(v70.columns) == list(v69.columns) and (v70.dtypes == v69.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v70, tmpl, None)]
    rk = pd.read_parquet(ROOT / "data/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    rk = rk[rk.PHASE_mvt.eq("DEP") & rk.ADEP_mvt.eq("LIRF") & rk.AOBT_3_flt.isna()]
    in_slice = set()
    for mid, mvt, sched in zip(rk[ID], rk.MVT_TIME_UTC_mvt, rk.SCHED_TIME_UTC_mvt):
        mvt, sched = mvt.to_pydatetime(), sched.to_pydatetime()
        delay = (mvt - sched).total_seconds()
        crossing = ((mvt - timedelta(minutes=15)).astimezone(ROME).date() - sched.astimezone(ROME).date()).days
        if 3600 < delay <= 28800 and not (delay > 28800 and crossing == 0):
            in_slice.add(mid)
    c["slice_rows_recomputed"] = len(in_slice)
    c["equals_build_slice"] = len(in_slice) == build["checks"]["slice_rows"]
    a, b = v70.set_index(ID)[T], v69.set_index(ID)[T]
    changed = set(a.index[a.to_numpy() != b.to_numpy()])
    c["rows_changed"] = len(changed)
    c["every_change_in_slice"] = changed <= in_slice
    outside = ~a.index.isin(list(in_slice))
    c["outside_rows_identical_to_v69"] = bool(np.array_equal(a[outside].to_numpy(), b[outside].to_numpy()))
    sys.path.insert(0, str(HERE))
    import build_v70 as B  # noqa: E402
    k, exp, _, _ = B.expected_2026()
    expect = ((b.loc[k[ID]].to_numpy(float) + exp) / 2).astype(a.dtype)
    c["refit_reproduces_values"] = bool(np.array_equal(a.loc[k[ID]].to_numpy(), expect))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
