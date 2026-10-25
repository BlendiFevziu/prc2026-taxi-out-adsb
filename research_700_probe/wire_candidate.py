"""Make the combined candidate (apply_learned.py output) uploadable as probe 16: write its ready_for_upload file and a ledger row.
Does NOT upload. The upload is done by:  python3 -B research_700_probe/upload_probe_once.py 16
Run from the repository root:  python3 -B research_700_probe/wire_candidate.py
"""
import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
NO, KEY = 16, "upbeat-goblin_v98.parquet"
src = ROOT / ".cache/research_700/learned"
rec = json.loads((src / "build.json").read_text())
path = src / "candidate_v80_learned.parquet"
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == rec["sha256"], "candidate changed since build.json; rerun apply_learned.py"
out = ROOT / f".cache/research_700/probe{NO}"; out.mkdir(parents=True, exist_ok=True)
(out / f"ready_for_upload_probe{NO}.json").write_text(json.dumps(
    {"version": "v98", "key": KEY, "candidate_path": str(path), "candidate_sha256": digest,
     "recipe": "v80 + 5 learned rows + group shifts (leaderboard-probed, research_700_probe); NM-free",
     "predicted_rmse": rec["predicted_rmse"], "authorised": "Owner, 4 October 2026: combined candidate, upload run by the owner"}, indent=2) + "\n")
LED = ROOT / "research_700_probe/ledger.csv"
FIELDS = ["probe", "key", "row_id", "old", "new", "step", "sha256", "intent_at", "official_score", "dSSE_vs_v80"]
rows = [r for r in csv.DictReader(LED.open()) if r["probe"] != str(NO)]
rows.append(dict(probe=NO, key=KEY, row_id="candidate:v80_learned_groups", old=rec["rows_changed"], new="", step="", sha256=digest,
                 intent_at=datetime.now(timezone.utc).isoformat(), official_score="", dSSE_vs_v80=""))
with LED.open("w", newline="") as f:
    w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
print(f"probe {NO} ready: {KEY} sha {digest[:16]} predicted RMSE {rec['predicted_rmse']}")
