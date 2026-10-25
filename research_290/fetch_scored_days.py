"""Fetch and extract the ADS-B tracks of every scored day, in batches that fit the disk. Downloads only, no labels.

The Munich-only candidate (build_munich_candidate.py) needs the tracks of all 62 scored days (January and July
2026). Each day is a 3 to 4 GB adsb.lol release, about 214 GB in all against 60 GB free on 24 September, so days
are fetched BATCH at a time while the previous batch is extracted, and fetch_days.extract deletes each day's raw
parts as soon as its tracks are written.

Outages were frequent on 24 September (github.com unreachable four times, for up to 90 minutes, each after an hour
or more of heavy downloading; DNS once; stalls every few minutes). The driver waits up to 6 hours for github.com.
fetch_days.download resumes a stalled part; a day whose fetch or extraction still fails is queued again, after
github.com answers, at most MAX_TRIES times. Days that already have tracks are skipped, so a rerun continues.

Run from the repository root:  python3 -u -B research_290/fetch_scored_days.py [--plan]
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADSB = Path("adsb-lol")
BATCH, MAX_TRIES, MIN_FREE_GB = 4, 3, 25.0


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def scored_days():
    frame = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["PHASE_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"])
    frame = frame[frame.PHASE_mvt.eq("DEP") & frame.TAXITIME_SEC_mvt.isna()]
    return sorted(pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").unique())


def have(day):
    return (ADSB / f"tracks/{day}.parquet").exists()


def free_gb():
    return shutil.disk_usage(ADSB).free / 1e9


def github_up():
    result = subprocess.run(["curl", "-s", "-o", "/dev/null", "-m", "15", "-w", "%{http_code}", "https://github.com"],
                            capture_output=True, text=True)
    return result.stdout.strip() == "200"


def wait_for_github(minutes=360):
    for _ in range(minutes):
        if github_up():
            return True
        time.sleep(60)
    return False


def start(days, phase):
    return subprocess.Popen([sys.executable, "-u", "-B", str(ADSB / "fetch_days.py"), "--dates", ",".join(days),
                             "--phase", phase, "--workers", "2"], cwd=ADSB, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


def finish(process):
    for line in process.communicate()[0].splitlines():
        if line.strip() and not line.startswith(("DONE", "1 dates", "2 dates", "3 dates", "4 dates")):
            log("  " + line[:200])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="print the plan and stop")
    args = parser.parse_args()
    days = scored_days()
    queue = [d for d in days if not have(d)]
    log(f"{len(days)} scored days, {len(days) - len(queue)} already have tracks, {len(queue)} to fetch, "
        f"{free_gb():.0f} GB free")
    if args.plan:
        for i in range(0, len(queue), BATCH):
            log("  batch " + ", ".join(queue[i:i + BATCH]))
        return
    tries, failed, pending, started = {d: 0 for d in queue}, [], [], time.time()
    while queue or pending:
        fetcher, batch = None, []
        if queue:
            if not github_up():
                log("github.com unreachable, waiting")
                if not wait_for_github():
                    log("still unreachable after 6 hours, stopping; rerun to continue")
                    break
            while free_gb() < MIN_FREE_GB and not pending:
                log(f"only {free_gb():.0f} GB free, waiting")
                time.sleep(60)
            batch, queue = queue[:BATCH], queue[BATCH:]
            for d in batch:
                tries[d] += 1
            log(f"fetch {', '.join(batch)}")
            fetcher = start(batch, "fetch")
        if pending:                                   # extract the previous batch while this one downloads
            log(f"extract {', '.join(pending)}")
            finish(start(pending, "extract"))
            for d in pending:
                if not have(d):
                    (queue if tries[d] < MAX_TRIES else failed).append(d)
            pending = []
        if fetcher is not None:
            finish(fetcher)
            for d in batch:
                if (ADSB / f"raw/{d}/COMPLETE").exists():
                    pending.append(d)
                else:
                    (queue if tries[d] < MAX_TRIES else failed).append(d)
        done = sum(have(d) for d in days)
        rate = (time.time() - started) / max(done - (len(days) - len(tries)), 1)
        log(f"progress: {done} of {len(days)} days have tracks; {free_gb():.0f} GB free; "
            f"about {rate * (len(days) - done) / 3600:.1f} h left at this pace")
    missing = [d for d in days if not have(d)]
    log(f"FINISHED: {len(days) - len(missing)} of {len(days)} days have tracks; missing {missing}; "
        f"given up after {MAX_TRIES} tries: {failed}")


if __name__ == "__main__":
    main()
