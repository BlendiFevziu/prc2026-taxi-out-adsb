"""Build several single-row probes in one go (calls build_probe.py once per number). Never touches probes already scored:
refuses numbers that have an official score in ledger.csv. No upload.
Run from the repository root:  python3 -u -B research_700_probe/build_probes_batch.py 6 7 8 9 10
"""
import csv, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
done = {r["probe"] for r in csv.DictReader((ROOT / "research_700_probe/ledger.csv").open()) if r["official_score"]}
for no in sys.argv[1:]:
    if no in done:
        raise SystemExit(f"probe {no} already has an official score; refusing to rebuild it")
for no in sys.argv[1:]:
    r = subprocess.run([sys.executable, "-u", "-B", str(ROOT / "research_700_probe/build_probe.py"), no], capture_output=True, text=True)
    print((r.stdout or r.stderr).strip().splitlines()[-1], flush=True)
