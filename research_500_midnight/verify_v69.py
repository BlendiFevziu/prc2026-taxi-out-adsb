"""Independent pre-upload check of v69 (research_500_midnight/build_v69.py), in a fresh process. No upload.

The rule is recomputed row by row with the standard library (zoneinfo, datetime), not with rule.py or pandas time zones:
every changed row must be a Rome fallback departure with delay above 28,800 s and no local-midnight crossing, carry its
delay exactly, and every row outside the rule must equal v68.
Run from the repository root:  python3 -u -B research_500_midnight/verify_v69.py
"""
import hashlib
import json
from datetime import timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import submission as SUB  # noqa: E402

ID, T, ROME = "MVT_ID_mvt", "TAXITIME_SEC_mvt", ZoneInfo("Europe/Rome")
V69 = ROOT / ".cache/research_500/v69/candidate_v69.parquet"
V68 = ROOT / ".cache/research_480/v68/candidate_v68.parquet"
OUT = ROOT / "research_500_midnight/results/v69_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((ROOT / "research_500_midnight/results/v69_build.json").read_text())
    build68 = json.loads((ROOT / "research_480_weights/results/v68_build.json").read_text())
    c = {"v69_sha_equals_build": sha(V69) == build["candidate_sha256"], "v68_sha_equals_its_build": sha(V68) == build68["candidate_sha256"]}
    v69, v68 = pd.read_parquet(V69), pd.read_parquet(V68)
    tmpl = pd.read_parquet(ROOT / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v69[ID].equals(v68[ID]) and v69[ID].equals(tmpl[ID]))
    c["schema_equal_v68"] = bool(list(v69.columns) == list(v68.columns) and (v69.dtypes == v68.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v69, tmpl, None)]
    rk = pd.read_parquet(ROOT / "data/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    rk = rk[rk.PHASE_mvt.eq("DEP") & rk.ADEP_mvt.eq("LIRF") & rk.AOBT_3_flt.isna()]
    expect = {}
    for mid, mvt, sched in zip(rk[ID], rk.MVT_TIME_UTC_mvt, rk.SCHED_TIME_UTC_mvt):
        mvt, sched = mvt.to_pydatetime(), sched.to_pydatetime()
        delay = (mvt - sched).total_seconds()
        if delay > 28800 and (mvt - timedelta(minutes=15)).astimezone(ROME).date() == sched.astimezone(ROME).date():
            expect[mid] = delay
    c["rule_rows_recomputed"] = len(expect)
    c["equals_build_rows"] = len(expect) == build["checks"]["rows_selected"]
    a, b = v69.set_index(ID)[T], v68.set_index(ID)[T]
    changed = set(a.index[a.to_numpy() != b.to_numpy()])
    c["rows_changed"] = len(changed)
    c["every_change_is_a_rule_row"] = changed <= set(expect)
    c["rule_rows_carry_delay"] = bool(all(np.isclose(a.loc[k], np.float64(v).astype(a.dtype), rtol=0, atol=1e-6) for k, v in expect.items()))
    outside = ~a.index.isin(list(expect))
    c["outside_rows_identical_to_v68"] = bool(np.array_equal(a[outside].to_numpy(), b[outside].to_numpy()))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
