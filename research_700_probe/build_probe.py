"""Build one single-row probe: v80 with exactly one Rome row moved by STEP seconds (policy: POLICY_probing.md).
Writes the candidate, ready_for_upload file and an 'intent' line in ledger.csv. No upload.
Run from the repository root:  python3 -u -B research_700_probe/build_probe.py <probe_no>
"""
import csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import submission as SUB  # noqa: E402
ID, T, STEP = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 20000.0
PROBES = {1: ("213145102.0", "upbeat-goblin_v83.parquet"),
          2: ("213153677.0", "upbeat-goblin_v84.parquet"),
          3: ("213145579.0", "upbeat-goblin_v85.parquet"),
          4: ("213152353.0", "upbeat-goblin_v86.parquet"),
          5: ("202680942.0", "upbeat-goblin_v87.parquet"),
          6: ("213142469.0", "upbeat-goblin_v88.parquet"),
          7: ("202680695.0", "upbeat-goblin_v89.parquet"),
          8: ("213147307.0", "upbeat-goblin_v90.parquet"),
          9: ("213154026.0", "upbeat-goblin_v91.parquet"),
          10: ("202689507.0", "upbeat-goblin_v92.parquet")}
LEDGER = ROOT / "research_700_probe/ledger.csv"
FIELDS = ["probe", "key", "row_id", "old", "new", "step", "sha256", "intent_at", "official_score", "dSSE_vs_v80"]
no = int(sys.argv[1]); rid, key = PROBES[no]
v80 = pd.read_parquet(ROOT / ".cache/research_640/v80/candidate_v80.parquet")
hit = v80[ID].eq(float(rid)).to_numpy()
assert hit.sum() == 1
old = float(v80.loc[hit, T].iloc[0]); new = old + STEP
cand = v80.copy(); cand.loc[hit, T] = new
template = pd.read_parquet(ROOT / "data/submitting.parquet")
cand[T] = cand[T].astype(template[T].dtype)
errs = SUB.pruefe(cand, template, None)
assert not errs, errs
assert int((cand[T].to_numpy() != v80[T].to_numpy()).sum()) == 1 and np.isfinite(cand[T]).all()
out = ROOT / f".cache/research_700/probe{no}"; out.mkdir(parents=True, exist_ok=True)
path = out / f"candidate_probe{no}.parquet"; cand.to_parquet(path, index=False)
digest = hashlib.sha256(path.read_bytes()).hexdigest()
new_w = float(cand.loc[hit, T].iloc[0])
(out / f"ready_for_upload_probe{no}.json").write_text(json.dumps(
    {"version": f"probe{no}", "key": key, "candidate_path": str(path), "candidate_sha256": digest,
     "recipe": f"v80 with row {rid} moved {old:.3f} -> {new_w:.3f}", "authorised": "Owner, 2 October 2026: 'do it' (batch 1), 'do all three' (batch 2)"}, indent=2) + "\n")
rows = list(csv.DictReader(LEDGER.open())) if LEDGER.exists() else []
rows = [r for r in rows if r["probe"] != str(no)]
rows.append(dict(probe=no, key=key, row_id=rid, old=old, new=new_w, step=STEP, sha256=digest,
                 intent_at=datetime.now(timezone.utc).isoformat(), official_score="", dSSE_vs_v80=""))
with LEDGER.open("w", newline="") as f:
    w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
print(f"probe {no}: row {rid} old {old:.3f} new {new_w:.3f} sha {digest[:16]}")
