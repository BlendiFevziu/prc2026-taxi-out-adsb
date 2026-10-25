"""Create a run folder that trains the frozen v45 candidate-stage ensemble (research_250 release worker, unchanged) with another
outer holdout pair, so that the ADS-B corrector can learn over the ensemble's own held-out predictions on more months (pilot 11).

The manifest is the September run's (`.cache/clock_queue_release_20260912/run.json`, same code, input and library hashes, checked
by the worker itself) with only the outer pair, timestamps and notes changed. The candidate stage takes its fallback from
`baseline/model.pkl`; here that is a link to the September baseline model, whose fallback never enters a main-lane prediction.
The tune stage's inner pair stays March/September, so the round count is chosen without reading the new outer pair.

Run from the repository root:
  python3 -B research_740_ensemble_holdout/make_run.py 2 6
  python3 -u -B research_250/release_worker.py --project . --run .cache/ensemble_holdout_2_6 --stage tune
  python3 -u -B research_250/release_worker.py --project . --run .cache/ensemble_holdout_2_6 --stage candidate
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".cache/clock_queue_release_20260912"
INNER = [3, 9]


def main(outer):
    assert len(outer) == 2 and not set(outer) & set(INNER) and set(outer) <= set(range(1, 13))
    run = ROOT / f".cache/ensemble_holdout_{outer[0]}_{outer[1]}"
    if (run / "run.json").exists():
        print(f"{run} already exists"); return
    manifest = json.loads((SOURCE / "run.json").read_text())
    manifest.update({"created_at": datetime.now(timezone.utc).isoformat(), "outer_source_months": outer, "inner_source_months": INNER,
                     "derived_from": str(SOURCE / "run.json"),
                     "authorization": "Owner in chat, 19 October 2026 (own project after the close): 'start it now alongside the download'. No upload, no submission.",
                     "policy": f"Held-out predictions of the frozen v45 candidate-stage ensemble for source months {outer}, for research_730 corrector training only. Stages tune and candidate only."})
    (run / "baseline").mkdir(parents=True)
    (run / "data").mkdir()
    (run / "baseline/model.pkl").symlink_to(SOURCE / "baseline/model.pkl")
    # the inner validation table depends only on the files of March and September, which are hash-checked unchanged
    (run / "data/months_3_9.parquet").symlink_to(SOURCE / "data/months_3_9.parquet")
    (run / "run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(f"created {run}")


if __name__ == "__main__":
    main(sorted(int(m) for m in sys.argv[1:]))
