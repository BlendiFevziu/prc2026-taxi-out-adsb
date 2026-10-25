"""Upload ONE probe candidate (research_700_probe/build_probe.py) once and record its official score in ledger.csv.
Same safeguards as upload_v82_once.py. Run from the repository root:  python3 -B research_700_probe/upload_probe_once.py <probe_no>
"""
import csv
import hashlib
import sys
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

utc = lambda: datetime.now(timezone.utc).isoformat()
NO = int(sys.argv[1])
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".cache" / "research_700" / f"probe{NO}"
ready = json.loads((RUN / f"ready_for_upload_probe{NO}.json").read_text())
if not ready.get("authorised"):
    raise SystemExit("The owner has not authorised this upload (ready file: authorised is empty).")
candidate = Path(ready["candidate_path"])
digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
assert digest == ready["candidate_sha256"], "candidate changed since the checks; not uploading"
MC, BUCKET, KEY = "/opt/homebrew/bin/mc", "prc26/prc-2026-upbeat-goblin", ready["key"]
intent_path, receipt_path = RUN / f"upload_intent_probe{NO}.json", RUN / f"upload_receipt_probe{NO}.json"
if receipt_path.exists():
    raise SystemExit("A receipt already exists: never upload twice.")
if intent_path.exists():
    raise SystemExit("An intent already exists: a previous run may have uploaded. Check the bucket by hand.")

listing = subprocess.run([MC, "ls", "--json", BUCKET + "/"], capture_output=True, text=True, timeout=180)
if listing.returncode != 0 or '"status":"error"' in listing.stdout:
    raise SystemExit("Cannot list the bucket (network or endpoint down). Nothing uploaded; try again later.")
keys, today = [], datetime.now(timezone.utc).date().isoformat()
for line in listing.stdout.splitlines():
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if row.get("key", "").endswith(".parquet"):
        keys.append((row["key"], row.get("lastModified", "")))
if any(k.endswith(KEY) for k, _ in keys):
    raise SystemExit(f"{KEY} is already in the bucket: never upload twice.")
todays = [k for k, t in keys if t.startswith(today)]
if len(todays) >= 5:
    raise SystemExit(f"Five submissions already carry {today}: the daily limit is reached.")
print(f"bucket holds {len(keys)} parquets, {len(todays)} today; uploading {KEY}", flush=True)

intent_path.write_text(json.dumps({"key": KEY, "sha256": digest, "at": utc(),
                                   "recipe": ready["recipe"]}, indent=2) + "\n")
upload = subprocess.run([MC, "cp", str(candidate), f"{BUCKET}/{KEY}"], capture_output=True, text=True, timeout=900)
if upload.returncode != 0:
    raise SystemExit("mc cp failed; NOT retrying. Check the bucket by hand before any further attempt.")
stat = subprocess.run([MC, "stat", "--json", f"{BUCKET}/{KEY}"], capture_output=True, text=True, timeout=180)
remote = json.loads(stat.stdout) if stat.returncode == 0 else {}
receipt_path.write_text(json.dumps({"key": KEY, "sha256": digest, "at": utc(),
                                    "remote_size": remote.get("size"),
                                    "local_size": candidate.stat().st_size}, indent=2) + "\n")
print(f"uploaded; remote size {remote.get('size')} local {candidate.stat().st_size}", flush=True)

deadline = time.time() + 600
while time.time() < deadline:
    result = subprocess.run([MC, "cat", f"{BUCKET}/{KEY}_result.json"], capture_output=True, text=True, timeout=180)
    if result.returncode == 0 and result.stdout.strip():
        print("OFFICIAL RESULT " + result.stdout.strip(), flush=True)
        (RUN / f"result_probe{NO}.json").write_text(result.stdout.strip() + "\n")
        score = json.loads(result.stdout)["score"]
        L = ROOT / "research_700_probe/ledger.csv"
        rows = list(csv.DictReader(L.open()))
        for r in rows:
            if r["probe"] == str(NO):
                r["official_score"] = score
                r["dSSE_vs_v80"] = 344841 * (float(score) ** 2 - 245.046 ** 2)
        with L.open("w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
        break
    time.sleep(20)
else:
    print("No result yet. Do NOT upload again; check the bucket later.", flush=True)
