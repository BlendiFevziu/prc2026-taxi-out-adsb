"""Label-free, no download: for every scored day whose preferred adsb.lol release is not prod-0 (the feed all 62 scored
days were fetched from), compare the preferred release's listed size with prod-0's. A near-equal size predicts a
near-equal track file, as on 13 July 2025, where both feeds were extracted: 896,041 against 895,934 points inside the
airport boxes, ground aircraft per airport within 1 % (research_290 era, adsb-lol/tracks/2025-07-13*.parquet).
Sizes come from HTTP HEAD requests on the release download links (final Content-Length after the redirect), which do
not count against GitHub's API quota. Output: research_300/results/preferred_release_sizes.json.
Run from the repository root:  python3 -u -B research_300/preferred_release_sizes.py
"""
import json
from pathlib import Path
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAPPING = Path("adsb-lol/preferred/PREFERRED_RELEASES_2026.txt")
OUT = ROOT / "research_300/results/preferred_release_sizes.json"


def head_size(url):
    """Final Content-Length of a download link, or None when it does not exist."""
    out = subprocess.run(["curl", "-sIL", "--max-time", "60", url], capture_output=True, text=True).stdout
    blocks = [b for b in out.replace("\r", "").split("\n\n") if b.strip()]
    if not blocks or " 200" not in blocks[-1].split("\n")[0]:
        return None
    lengths = [l.split(":", 1)[1].strip() for l in blocks[-1].split("\n") if l.lower().startswith("content-length")]
    return int(lengths[0]) if lengths else None


def prod0_size(day):
    """Sum over prod-0's parts .tar.aa, .tar.ab, ... until the first missing one; an unsplit .tar otherwise."""
    tag = "v" + day.replace("-", ".") + "-planes-readsb-prod-0"
    base = f"https://github.com/adsblol/globe_history_2026/releases/download/{tag}/{tag}.tar"
    total, parts = 0, 0
    for suffix in "abcdefghij":
        size = head_size(f"{base}.a{suffix}")
        if size is None:
            break
        total, parts = total + size, parts + 1
    if not parts:
        size = head_size(base)
        return (size, 1) if size else (None, 0)
    return total, parts


def main():
    frame = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["PHASE_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"])
    frame = frame[frame.PHASE_mvt.eq("DEP") & frame.TAXITIME_SEC_mvt.isna()]
    days = sorted(pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").unique())
    lines = MAPPING.read_text().split("\n")
    report = {}
    for day in days:
        tag = "v" + day.replace("-", ".") + "-planes-readsb-"
        line = [l for l in lines if f"/{tag}" in l][0]
        preferred = line.split(tag)[1].split("/")[0]      # the tag appears twice per URL: path and file name
        if preferred == "prod-0":
            continue
        urls = [u for u in line.strip().split(",") if u]
        sizes = [head_size(u) for u in urls]
        pref = sum(sizes) if all(sizes) else None
        prod, n_prod = prod0_size(day)
        report[day] = {"preferred": preferred, "preferred_parts": len(urls), "preferred_bytes": pref,
                       "prod0_parts": n_prod, "prod0_bytes": prod,
                       "ratio": round(pref / prod, 4) if pref and prod else None}
        print(day, report[day], flush=True)
    ratios = [v["ratio"] for v in report.values() if v["ratio"]]
    summary = {"days": len(report), "median_ratio": float(pd.Series(ratios).median()),
               "min_ratio": min(ratios), "max_ratio": max(ratios)}
    OUT.write_text(json.dumps({"summary": summary, "days": report}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
