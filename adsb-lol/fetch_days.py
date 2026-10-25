"""Fetch adsb.lol daily releases, then extract airport tracks and delete the raw archive.

Source: github.com/adsblol/globe_history_YYYY, ODbL 1.0, attribution required. One release per day and instance,
about 3.3 GB split into 2 GB parts. Raw archives are deleted as soon as a day's tracks are written, so the disk
holds only the filtered tracks (a few tens of MB per day).

Phases can be run separately, because downloading is network-bound and extracting is CPU-bound:
  --phase fetch     download the parts for each date
  --phase extract   extract tracks for dates whose parts are present, then delete the parts
  --phase both      do each date end to end

Run:  python3 -u -B fetch_days.py --dates 2026-07-01..2026-07-08 --phase fetch
      python3 -u -B fetch_days.py --dates 2026-07-01..2026-07-08 --phase extract --workers 4
"""
import argparse
import concurrent.futures as futures
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
TRACKS = HERE / "tracks"
INSTANCE = "planes-readsb-prod-0"


def expand(spec):
    import datetime as dt
    out = []
    for piece in spec.split(","):
        piece = piece.strip()
        if ".." in piece:
            start, end = piece.split("..")
            a = dt.date.fromisoformat(start)
            b = dt.date.fromisoformat(end)
            while a <= b:
                out.append(a.isoformat())
                a += dt.timedelta(days=1)
        elif piece:
            out.append(piece)
    return out


def urls(date):
    year = date[:4]
    tag = f"v{date.replace('-', '.')}-{INSTANCE}"  # releases are tagged v2026.07.06-..., with dots
    base = f"https://github.com/adsblol/globe_history_{year}/releases/download/{tag}"
    # summer 2026 days need three 2 GB parts (2026-07-15: aa, ab, ac); fetch() stops at the first missing one
    return [(f"{base}/{tag}.tar.a{letter}", RAW / date / f"part.a{letter}") for letter in "abcdefghij"]


def download(url, target, attempts=5):
    """One part; returns (HTTP status, curl exit code, stderr), with status "200" for a complete part.

    The status is read with -w, because this Mac's curl reports a 404 under --fail as exit 56, the same code as a
    network failure. A stalled transfer aborts after 120 s below 50 KB/s (on 24 September one sat at 64 KB for 28
    minutes), and is then RESUMED from its partial file with -C -, up to `attempts` times: that morning the network
    paused every few minutes, and restarting a 2 GB part from zero each time never finished (2025-01-15, four tries).

    17 October 2026: a partial file left by an EARLIER call is resumed too (both callers pass a ".partial" target that belongs to
    exactly this URL; the callers' size and tar-end checks still catch a bad result), and a failed attempt waits 60 s before the
    next, because the network there dropped for minutes every 30-40 minutes and five instant retries never bridged an outage.
    """
    code, exit_code, error = "000", -1, ""
    for attempt in range(attempts):
        if attempt:
            time.sleep(60)
        resume = ["-C", "-"] if target.exists() and target.stat().st_size > 0 else []
        result = subprocess.run(["curl", "-sSL", *resume, "--connect-timeout", "30", "--speed-limit", "50000",
                                 "--speed-time", "120", "-w", "%{http_code}", "-o", str(target), url],
                                capture_output=True, text=True)
        code, exit_code, error = result.stdout.strip()[-3:], result.returncode, result.stderr.strip()
        if exit_code == 0 and code in ("200", "206"):
            return "200", 0, ""
        if code == "416" and resume:          # the partial file is already complete; the caller's size check decides
            return "200", 0, ""
        if code == "404":
            break
    return code, exit_code, error


def fetch(date):
    started = time.time()
    (RAW / date).mkdir(parents=True, exist_ok=True)
    total = 0
    for index, (url, path) in enumerate(urls(date)):
        # a part counts as downloaded only with its .ok marker: a bare file may be a truncated leftover
        # (raw/2026-07-06 and raw/2026-07-13 from 21 September are exactly that)
        done = path.with_name(path.name + ".ok")
        if path.exists() and done.exists():
            total += path.stat().st_size
            continue
        # download under a temporary name: a transfer that dies halfway must not leave a part that looks complete
        # (this happened on 23 September 2026)
        partial = path.with_name(path.name + ".partial")
        code, exit_code, error = download(url, partial)
        if exit_code != 0 or code != "200":
            partial.unlink(missing_ok=True)
            # HTTP 404 on any part after the first means the archive ended with the previous part. Any other
            # failure is the network, and must not pass for the end: on 24 September a timed-out second part of
            # 2025-12-14 was taken for a short day that way, and the day would have been extracted from its first half.
            if index > 0 and code == "404":
                break
            return f"FETCH FAIL {date}: http {code}, curl exit {exit_code}: {error[:100]}"
        partial.rename(path)
        done.write_text(str(path.stat().st_size))
        total += path.stat().st_size
    # the day is complete only here; extract() refuses a day without this marker, because the first part ends on a
    # tar block boundary and extracting it alone can end quietly and write half a day
    (RAW / date / "COMPLETE").write_text(f"{total}\n")
    return f"fetched {date}: {total / 1e9:.2f} GB in {time.time() - started:.0f}s"


def extract(date):
    started = time.time()
    if not (RAW / date / "COMPLETE").exists():
        return f"EXTRACT SKIP {date}: download not complete"
    parts = [p for _, p in urls(date) if p.exists()]
    if not parts:
        return f"EXTRACT SKIP {date}: no parts"
    out = TRACKS / f"{date}.parquet"
    if out.exists():
        return f"extract skip {date}: already done"
    command = [sys.executable, "-B", str(HERE / "extract_tracks.py"), "--parts", *map(str, parts),
               "--date", date, "--out", str(out)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        return f"EXTRACT FAIL {date}: {result.stdout[-200:]} {result.stderr[-200:]}"
    for leftover in (RAW / date).iterdir():   # parts, their .ok markers, COMPLETE, any stale .partial
        leftover.unlink()
    (RAW / date).rmdir()
    tail = [line for line in result.stdout.splitlines() if line.startswith("traces")]
    return f"extracted {date} in {time.time() - started:.0f}s | {tail[0] if tail else ''}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True)
    parser.add_argument("--phase", choices=["fetch", "extract", "both"], default="both")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    dates = expand(args.dates)
    print(f"{len(dates)} dates, phase {args.phase}, {args.workers} workers", flush=True)
    jobs = []
    if args.phase in ("fetch", "both"):
        jobs.append(("fetch", fetch, max(2, args.workers)))
    if args.phase in ("extract", "both"):
        jobs.append(("extract", extract, args.workers))
    for name, function, workers in jobs:
        with futures.ThreadPoolExecutor(workers) if name == "fetch" else futures.ProcessPoolExecutor(workers) as pool:
            for message in pool.map(function, dates):
                print(message, flush=True)
    print("DONE", flush=True)
