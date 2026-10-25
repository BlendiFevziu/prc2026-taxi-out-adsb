"""Fetch ONE day from its PREFERRED adsb.lol release, verify it, and extract it. Downloads only; no labels are read.

Mapping: PREFERRED_RELEASES.txt of the public repositories adsblol/globe_history_2025 and _2026 (their README names
it the source for selecting daily archives), copied to adsb-lol/preferred/ with its SHA-256 recorded. One line per
day, the comma-separated URLs of that day's preferred release parts: prod-0, staging-0, a temporary staging replica
or mlatonly-0; one part when the day's archive is under 2 GB (unsplit ".tar"), several otherwise.

Why a separate tool: fetch_days.py assumes prod-0 and split names (.tar.aa, .tar.ab). On 25 September that made
2025-01-21 look absent (HTTP 404), although its preferred release is a single unsplit prod-0 file.

Verification, all recorded in research_300/results/fetch_preferred_{date}.json:
  size       each part's size equals the size in the GitHub release listing (the listing has no digest to check)
  sha256     computed locally for each part and recorded
  tar end    the last part ends in the tar end-of-archive marker (at least two zero 512-byte blocks)
  extract    fetch_days.extract reads every member to the end (a truncated archive fails there, it never writes)
Extraction runs under the shared heavy-work lock.

With --suffix (26 September 2026) the day is extracted to tracks/{date}{suffix}.parquet instead, so a day already
extracted from another release stays untouched; the raw parts are deleted after a successful extraction either way.
When the GitHub API quota is exhausted, the part sizes are checked against the size the download link itself serves
(HTTP HEAD, final Content-Length); the report records which source was used.

Run from the repository root:  python3 -u -B research_300/fetch_preferred.py 2025-01-21 [--suffix=-preferred]
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ADSB = Path("adsb-lol")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ADSB))
import fetch_days as FD  # noqa: E402
from research_300.heavy_lock import heavy_work  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def preferred_urls(date):
    mapping = ADSB / f"preferred/PREFERRED_RELEASES_{date[:4]}.txt"
    tag = "v" + date.replace("-", ".") + "-"
    lines = [l for l in mapping.read_text().split("\n") if f"/{tag}" in l]
    if len(lines) != 1:
        raise SystemExit(f"{date}: {len(lines)} lines in {mapping.name}")
    return [u for u in lines[0].strip().split(",") if u], mapping


def listed_sizes(url):
    repo, tag = url.split("/releases/download/")[0].split("github.com/")[1], url.split("/releases/download/")[1].split("/")[0]
    with urllib.request.urlopen(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", timeout=60) as r:
        release = json.loads(r.read().decode())
    return {a["name"]: a["size"] for a in release["assets"]}, tag


def served_sizes(urls):
    """Fallback when the API quota is exhausted: the size each download link serves (HEAD, after redirects)."""
    from research_300.preferred_release_sizes import head_size
    return {u.rsplit("/", 1)[1]: head_size(u) for u in urls}


def extract_to(date, out):
    """fetch_days.extract with a chosen output file; deletes the raw parts only after a successful extraction."""
    folder = FD.RAW / date
    parts = sorted(folder.glob("part.a?"))
    command = [sys.executable, "-B", str(ADSB / "extract_tracks.py"), "--parts", *map(str, parts),
               "--date", date, "--out", str(out)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        return f"EXTRACT FAIL {date}: {result.stdout[-200:]} {result.stderr[-200:]}"
    for leftover in folder.iterdir():
        leftover.unlink()
    folder.rmdir()
    tail = [line for line in result.stdout.splitlines() if line.startswith("traces")]
    return f"extracted {date} -> {out.name} | {tail[0] if tail else ''}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    date, suffix = args.date, args.suffix
    out = FD.TRACKS / f"{date}{suffix}.parquet"
    if suffix and out.exists():
        raise SystemExit(f"{out.name} exists already")
    urls, mapping = preferred_urls(date)
    tag = urls[0].split("/releases/download/")[1].split("/")[0]
    try:
        sizes, tag = listed_sizes(urls[0])
        size_source = "github release api"
    except Exception:
        sizes, size_source = served_sizes(urls), "http head (api quota exhausted)"
    folder = FD.RAW / date
    folder.mkdir(parents=True, exist_ok=True)
    report = {"date": date, "mapping_file": str(mapping), "mapping_sha256": sha256(mapping), "release": tag,
              "size_source": size_source, "parts": [], "tracks": None}
    for i, url in enumerate(urls):
        name = url.rsplit("/", 1)[1]
        part = folder / f"part.a{'abcdefghij'[i]}"
        code, exit_code, error = FD.download(url, part.with_name(part.name + ".partial"))
        if exit_code != 0 or code != "200":
            raise SystemExit(f"{name}: download failed (http {code}, curl {exit_code}) {error[:100]}")
        part.with_name(part.name + ".partial").rename(part)
        entry = {"asset": name, "listed_size": sizes.get(name), "size": part.stat().st_size, "sha256": sha256(part)}
        entry["size_matches_listing"] = entry["size"] == entry["listed_size"]
        report["parts"].append(entry)
        print(json.dumps(entry), flush=True)
        if not entry["size_matches_listing"]:
            raise SystemExit(f"{name}: size {entry['size']} differs from the listing {entry['listed_size']}")
        part.with_name(part.name + ".ok").write_text(str(entry["size"]))
    with open(folder / f"part.a{'abcdefghij'[len(urls) - 1]}", "rb") as f:     # the last part holds the archive's end
        f.seek(-1024, 2)
        tail = f.read()
    report["tar_end_marker"] = tail == b"\0" * 1024
    if not report["tar_end_marker"]:
        raise SystemExit("the last part does not end in the tar end-of-archive marker")
    (folder / "COMPLETE").write_text(f"{sum(p['size'] for p in report['parts'])}\n")
    with heavy_work(f"adsb extraction {date} (preferred release)"):
        message = extract_to(date, out) if suffix else FD.extract(date)
    print(message, flush=True)
    report["extract"] = message
    report["tracks"] = str(out) if out.exists() else None
    (ROOT / f"research_300/results/fetch_preferred_{date}{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["tracks"] is None:
        raise SystemExit("extraction failed: " + message)


if __name__ == "__main__":
    main()
