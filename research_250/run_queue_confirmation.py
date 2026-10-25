"""Run queue source pairs sequentially; defaults to five confirmation pairs.

For January/July reproduction, pass --pairs '1,7' and point --baseline-dir at the
original screen directory containing main_predictions_1_7.parquet.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--core-dir", type=Path, required=True)
parser.add_argument("--baseline-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--pairs", default="2,8;3,9;4,10;5,11;6,12")
parser.add_argument("--jobs", type=int, default=8)
args = parser.parse_args()
lookup = {}
for p in sorted(args.core_dir.glob("core_*.json")):
    source = json.loads(p.read_text())
    key = tuple(sorted(source["source_months"]))
    if key in lookup:
        raise ValueError(f"Ambiguous source-month reference {key}")
    lookup[key] = args.core_dir/source["core_file"]
for pair in [tuple(sorted(map(int, p.split(",")))) for p in args.pairs.split(";")]:
    assert len(pair) == len(set(pair)) == 2 and set(pair) <= set(range(1, 13))
    suffix = "_".join(map(str, pair))
    baseline_name = f"main_predictions_{suffix}.parquet" if pair == (1, 7) else f"time_cv_predictions_{suffix}.parquet"
    baseline = args.baseline_dir/baseline_name
    assert baseline.exists()
    print("LOCKED QUEUE PAIR", pair, flush=True)
    subprocess.run([sys.executable, "-u", "-B", str(Path(__file__).with_name("queue_screen.py")),
        "--project", str(args.project), "--core-train", str(lookup[tuple(sorted(set(range(1,13))-set(pair)))]),
        "--core-valid", str(lookup[pair]), "--baseline-predictions", str(baseline),
        "--output", str(args.output), "--jobs", str(args.jobs)], check=True)
