"""Independent pre-upload check of v82 (research_680_rome_overnight/build_v82.py), in a fresh process. No upload.
Recomputes delay and the local-midnight crossing row by row with zoneinfo and datetime (not the rule module), and the
+24 h median from the raw 2025 files; every changed row must be a Rome fallback crossing row with delay above 8 h.
Run from the repository root:  python3 -u -B research_680_rome_overnight/verify_v82.py
"""
import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import submission as SUB  # noqa: E402

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
ROME = ZoneInfo("Europe/Rome")
V81, V82 = ROOT / ".cache/research_650/v81/candidate_v81.parquet", ROOT / ".cache/research_680/v82/candidate_v82.parquet"
OUT = ROOT / "research_680_rome_overnight/results/v82_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def annotate(f):
    mvt = [t.to_pydatetime() for t in pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True)]
    sch = [t.to_pydatetime() for t in pd.to_datetime(f.SCHED_TIME_UTC_mvt, utc=True)]
    delay = np.array([(m - s).total_seconds() for m, s in zip(mvt, sch)])
    cross = np.array([((m - timedelta(minutes=15)).astimezone(ROME).date() - s.astimezone(ROME).date()).days for m, s in zip(mvt, sch)])
    return delay, cross


def main():
    build = json.loads((ROOT / "research_680_rome_overnight/results/v82_build.json").read_text())
    c = {"v82_sha_equals_build": sha(V82) == build["candidate_sha256"]}
    v82, v81 = pd.read_parquet(V82), pd.read_parquet(V81)
    tmpl = pd.read_parquet(ROOT / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v82[ID].equals(v81[ID]) and v82[ID].equals(tmpl[ID]))
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v82, tmpl, None)]
    raw = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "data").glob("training_2025-*.parquet"))], ignore_index=True)
    r = raw[(raw.PHASE_mvt == "DEP") & (raw.ADEP_mvt == "LIRF") & raw.AOBT_3_flt.isna()]
    y = r[T].to_numpy(float)
    y24 = float(np.median(y[(y > 72000) & (y < 100800)]))
    c["y24_recomputed"] = y24
    k = pd.read_parquet(ROOT / "data/ranking.parquet")
    k = k[(k.PHASE_mvt == "DEP") & (k.ADEP_mvt == "LIRF") & k.AOBT_3_flt.isna()].reset_index(drop=True)
    delay, cross = annotate(k)
    long_, mid = (cross == 1) & (delay >= 50400), (cross == 1) & (delay > 28800) & (delay < 50400)
    exp = pd.Series(np.r_[(delay[long_] + y24) / 2, delay[mid]], index=np.r_[k[ID].to_numpy()[long_], k[ID].to_numpy()[mid]])
    a, b = v82.set_index(ID)[T], v81.set_index(ID)[T]
    changed = a.index[a.to_numpy() != b.to_numpy()]
    c["rows_changed"], c["rows_expected"] = int(len(changed)), int(len(exp))
    c["every_change_is_a_target_row"] = bool(pd.Index(changed).isin(exp.index).all())
    c["values_equal"] = bool(np.array_equal(exp.loc[exp.index].to_numpy().astype(a.dtype), a.loc[exp.index].to_numpy()))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"] and c["rows_changed"] == c["rows_expected"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
