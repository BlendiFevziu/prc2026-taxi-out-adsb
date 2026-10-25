"""Fetch and extract adsb.lol days for February and June, using fetch_days.py unchanged.

Why a wrapper: fetch_days.py hard-codes the release instance planes-readsb-prod-0, and several June 2025 days only
exist under another instance name (listed in preferred/PREFERRED_RELEASES_2025.txt, e.g. planes-readsb-staging-0tmp).
This wrapper resolves the instance per day (prod-0 first, to stay consistent with the January/July tracks, then the
instances in the preferred list), patches fetch_days.urls, and runs fetch and extract as a pipeline:
fetch (3 parallel connections) -> extract (3 parallel processes) as soon as a day is complete.

Order: first the 2025 validation days (labels exist, used to check frozen layers on February/June), then all 2026
days of February and June (the rows to be scored in the final phase).
Source: github.com/adsblol/globe_history_YYYY, ODbL 1.0, attribution required.

Run:  python3 -u -B fetch_months.py            (log lines to stdout)
"""
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_days as F  # noqa: E402

VAL_2025 = [f"2025-02-{d:02d}" for d in (3, 5, 7, 10, 12, 14)] + \
           [f"2025-06-{d:02d}" for d in (2, 4, 6, 9, 11, 13)]
DAYS_2026 = F.expand("2026-02-01..2026-02-28,2026-06-01..2026-06-30")
DATES = VAL_2025 + DAYS_2026
INSTANCE = {}
WORKERS = int(os.environ.get('FETCH_WORKERS', '6'))


def listed_instances(date):
    path = HERE / "preferred" / f"PREFERRED_RELEASES_{date[:4]}.txt"
    tag = "v" + date.replace("-", ".") + "-"
    found = []
    if path.exists():
        for m in re.findall(re.escape(tag) + r"(planes-[a-z0-9-]+?)/", path.read_text()):
            if m not in found:
                found.append(m)
    return found


def head_ok(date, instance):
    tag = f"v{date.replace('-', '.')}-{instance}"
    url = f"https://github.com/adsblol/globe_history_{date[:4]}/releases/download/{tag}/{tag}.tar.aa"
    out = subprocess.run(["curl", "-sIL", "-m", "30", "-o", "/dev/null", "-w", "%{http_code}", url],
                         capture_output=True, text=True).stdout.strip()[-3:]
    return out == "200"


def resolve(date):
    for instance in ["planes-readsb-prod-0"] + listed_instances(date):
        if head_ok(date, instance):
            return instance
    return None


def urls(date):
    tag = f"v{date.replace('-', '.')}-{INSTANCE[date]}"
    base = f"https://github.com/adsblol/globe_history_{date[:4]}/releases/download/{tag}"
    return [(f"{base}/{tag}.tar.a{letter}", F.RAW / date / f"part.a{letter}") for letter in "abcdefghij"]


F.urls = urls


def download(url, target, attempts=8):
    """fetch_days.download, but a partial file from an earlier run is always resumed (curl -C -)."""
    code, exit_code, error = "000", -1, ""
    for attempt in range(attempts):
        resume = ["-C", "-"] if target.exists() and target.stat().st_size > 0 else []
        result = subprocess.run(["curl", "-sSL", *resume, "--connect-timeout", "30", "--speed-limit", "50000",
                                 "--speed-time", "120", "-w", "%{http_code}", "-o", str(target), url],
                                capture_output=True, text=True)
        code, exit_code, error = result.stdout.strip()[-3:], result.returncode, result.stderr.strip()
        if exit_code == 0 and code in ("200", "206"):
            return "200", 0, ""
        if code == "404":
            break
        if code == "416":  # range not satisfiable: the partial file is already complete
            return "200", 0, ""
    return code, exit_code, error


F.download = download


def stamp(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    todo = [d for d in DATES if not (F.TRACKS / f"{d}.parquet").exists()]
    stamp(f"{len(DATES)} dates wanted, {len(todo)} without tracks")
    for d in todo:
        INSTANCE[d] = resolve(d)
        if INSTANCE[d] is None:
            stamp(f"NO RELEASE FOUND for {d}")
    todo = [d for d in todo if INSTANCE.get(d)]
    stamp("instances: " + ", ".join(sorted({f'{v}' for v in INSTANCE.values() if v})))
    extract_pool = ThreadPoolExecutor(3)

    def do_extract(date):
        stamp(F.extract(date))

    def do_fetch(date):
        if (F.TRACKS / f"{date}.parquet").exists():
            return None
        lock = HERE / "raw" / f".lock_{date}"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # one process per day
            os.close(fd)
        except FileExistsError:
            return None
        stamp(F.fetch(date))
        return date

    def done(fut):
        try:
            date = fut.result()
            if date is None:
                return
        except Exception as error:  # a crash in one day must not stop the pipeline
            stamp(f"FETCH ERROR {error!r}")
            return
        if (F.RAW / date / "COMPLETE").exists():
            extract_pool.submit(do_extract, date)

    with ThreadPoolExecutor(WORKERS) as fetch_pool:
        for d in todo:
            fetch_pool.submit(do_fetch, d).add_done_callback(done)
    extract_pool.shutdown(wait=True)
    stamp("ALL DONE")


if __name__ == "__main__":
    main()
