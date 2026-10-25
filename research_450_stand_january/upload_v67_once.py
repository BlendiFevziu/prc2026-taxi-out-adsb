"""Upload candidate v67 (v66 plus stand model on LSZH/EHAM January and stand weight 1.0) exactly once and retrieve its official result.

Same safeguards as research_440_fresh_july/upload_v66_once.py: verify the candidate hash against ready_for_upload_v67.json,
refuse when a receipt or an intent already exists, list the bucket and refuse if the key is already there, refuse
when five team submissions already carry today's UTC date, write the intent BEFORE the single mc cp, write the
receipt with the remote size afterwards, never retry on error, then poll for the official result. Credentials are
never printed.

Recipe and the prediction fixed before upload: in the ready file (research_450_stand_january/build_v67.py,
research_450_stand_january/verify_v67.py). Authorised by the owner on 29 September 2026.
Run from the repository root:  python3 -B research_450_stand_january/upload_v67_once.py
"""
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

utc = lambda: datetime.now(timezone.utc).isoformat()
RUN = Path(__file__).resolve().parents[1] / ".cache" / "research_450" / "v67"
ready = json.loads((RUN / "ready_for_upload_v67.json").read_text())
if not ready.get("authorised"):
    raise SystemExit("The owner has not authorised this upload (ready file: authorised is empty).")
candidate = Path(ready["candidate_path"])
digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
assert digest == ready["candidate_sha256"], "candidate changed since the checks; not uploading"
MC, BUCKET, KEY = "/opt/homebrew/bin/mc", "prc26/prc-2026-upbeat-goblin", ready["key"]
intent_path, receipt_path = RUN / "upload_intent_v67.json", RUN / "upload_receipt_v67.json"
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
        (RUN / "result_v67.json").write_text(result.stdout.strip() + "\n")
        break
    time.sleep(20)
else:
    print("No result yet. Do NOT upload again; check the bucket later.", flush=True)
