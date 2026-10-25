"""Download public METAR observations for the ten competition airports.

Source: Iowa Environmental Mesonet ASOS/METAR archive,
https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py (public, no key). Routine METAR and
SPECI reports in UTC with temperature and dewpoint (F), wind direction (degrees true) and
speed (kt), gust (kt), visibility (statute miles) and present-weather codes.

Files go to .cache/research_240/weather/metar_<ICAO>.csv and are never committed. A station
whose file already passes the coverage check is not downloaded again. Each request is retried
three times; nothing is retried after that.

Run from the repository root:
  python3 research_240/fetch_metar.py
"""
import argparse
import csv
import io
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
FIELDS = ["tmpf", "dwpf", "drct", "sknt", "gust", "vsby", "wxcodes"]
HEADER = ["station", "valid"] + FIELDS
SOURCE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
# End dates are exclusive in the IEM interface (checked on 13.09.2026: 3-4 January gave one day).
CHUNKS = [((2024, 12, 31), (2026, 1, 1)), ((2026, 1, 1), (2026, 8, 1))]
MONTHS = [f"2025-{m:02d}" for m in range(1, 13)] + [f"2026-{m:02d}" for m in range(1, 8)]


def request_url(station, start, end):
    params = [("station", station)] + [("data", f) for f in FIELDS] + [
        ("year1", start[0]), ("month1", start[1]), ("day1", start[2]),
        ("year2", end[0]), ("month2", end[1]), ("day2", end[2]),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
        ("missing", "M"), ("trace", "T"), ("direct", "no"), ("report_type", 3), ("report_type", 4)]
    return SOURCE + "?" + urllib.parse.urlencode(params)


def fetch(url):
    last = None
    for attempt, pause in enumerate((0, 15, 45)):
        time.sleep(pause)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "the project PRC research (weather features)"})
            with urllib.request.urlopen(req, timeout=300) as response:
                return response.read().decode("utf-8")
        except Exception as error:  # network errors are reported after the last attempt
            last = error
            print(f"  attempt {attempt+1} failed: {error}", flush=True)
    raise RuntimeError(f"download failed after three attempts: {last}")


def coverage(rows):
    """Share of hours with at least one report, per month."""
    hours = {}
    for row in rows:
        stamp = row["valid"]
        hours.setdefault(stamp[:7], set()).add(stamp[:13])
    out = {}
    for month in MONTHS:
        year, mon = int(month[:4]), int(month[5:])
        days = 31 if mon in (1, 3, 5, 7, 8, 10, 12) else 30 if mon != 2 else 28
        out[month] = round(len(hours.get(month, ())) / (days * 24), 4)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".cache/research_240/weather"))
    parser.add_argument("--min-coverage", type=float, default=0.9)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"source": SOURCE, "fields": FIELDS, "stations": {}}
    for station in AIRPORTS:
        path = args.output / f"metar_{station}.csv"
        if path.exists():
            rows = list(csv.DictReader(path.open(newline="")))
            cov = coverage(rows)
            if min(cov.values()) >= args.min_coverage:
                summary["stations"][station] = {"rows": len(rows), "kept_existing": True, "min_month_coverage": min(cov.values())}
                print(station, "kept", len(rows), "rows", flush=True)
                continue
        rows = []
        for start, end in CHUNKS:
            text = fetch(request_url(station, start, end))
            reader = csv.DictReader(io.StringIO(text))
            if reader.fieldnames != HEADER:
                raise RuntimeError(f"{station}: unexpected header {reader.fieldnames}; first bytes {text[:200]!r}")
            rows.extend(reader)
            time.sleep(3)
        rows.sort(key=lambda r: r["valid"])
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(rows)
        cov = coverage(rows)
        summary["stations"][station] = {"rows": len(rows), "first": rows[0]["valid"] if rows else None,
                                         "last": rows[-1]["valid"] if rows else None,
                                         "min_month_coverage": min(cov.values()),
                                         "months_below_min": {m: c for m, c in cov.items() if c < args.min_coverage}}
        print(station, json.dumps(summary["stations"][station]), flush=True)
    (args.output / "fetch_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
