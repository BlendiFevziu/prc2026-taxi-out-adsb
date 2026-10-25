"""Download and extract given adsb.lol days, BATCH at a time, extraction under the shared heavy-work lock.

Reuses the tested helpers of research_290/fetch_scored_days.py (outage waits, subprocess wrapper). BATCH is 2
because the disk had 26 GB free on 25 September and a July 2025 day is about 3.3 GB; the next batch downloads while
the previous one is extracted. A failed day is queued again, at most MAX_TRIES times. Days with tracks are skipped.
Downloads only; no labels are read.

Run from the repository root:  python3 -u -B research_300/fetch_dates.py 2025-01-09 2025-01-13 ...
"""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_290 import fetch_scored_days as F  # noqa: E402
from research_300.heavy_lock import heavy_work  # noqa: E402

BATCH, MAX_TRIES = 2, 3


def main():
    days = sys.argv[1:]
    queue = [d for d in days if not F.have(d)]
    F.log(f"{len(days)} days requested, {len(days) - len(queue)} already have tracks, {len(queue)} to fetch, "
          f"{F.free_gb():.0f} GB free")
    tries, failed, pending = {d: 0 for d in queue}, [], []
    while queue or pending:
        fetcher, batch = None, []
        if queue:
            if not F.github_up():
                F.log("github.com unreachable, waiting")
                if not F.wait_for_github():
                    F.log("still unreachable after 6 hours, stopping; rerun to continue")
                    break
            batch, queue = queue[:BATCH], queue[BATCH:]
            for d in batch:
                tries[d] += 1
            F.log(f"fetch {', '.join(batch)}")
            fetcher = F.start(batch, "fetch")
        if pending:
            with heavy_work(f"adsb extraction {', '.join(pending)}", log=F.log):
                F.log(f"extract {', '.join(pending)}")
                F.finish(F.start(pending, "extract"))
            for d in pending:
                if not F.have(d):
                    (queue if tries[d] < MAX_TRIES else failed).append(d)
            pending = []
        if fetcher is not None:
            F.finish(fetcher)
            for d in batch:
                if (F.ADSB / f"raw/{d}/COMPLETE").exists():
                    pending.append(d)
                else:
                    (queue if tries[d] < MAX_TRIES else failed).append(d)
        F.log(f"progress: {sum(F.have(d) for d in days)} of {len(days)} days have tracks; {F.free_gb():.0f} GB free")
    missing = [d for d in days if not F.have(d)]
    F.log(f"FINISHED: {len(days) - len(missing)} of {len(days)}; missing {missing}; given up: {failed}")


if __name__ == "__main__":
    main()
