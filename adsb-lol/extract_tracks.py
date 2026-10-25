"""Extract ADS-B track points near the ten competition airports from one adsb.lol daily archive.

Why this exists. The organisers ruled on 19 September that open trajectory data may be used and that off-block times
may be extracted from it: "if you can find open trajectory data with good ground coverage and extract off-block
times, we are ok". That is the field the repository's own analysis said was missing, a physical off-block with broad
coverage, and the target is the airport's reported block time, which for ordinary rows is close to it.

Source: adsb.lol daily releases (github.com/adsblol/globe_history_YYYY), ODbL 1.0, attribution required. Each release
is a tar of one gzipped JSON trace per aircraft for that day, about 3.3 GB per instance split into 2 GB parts.

Method. The tar is streamed, never unpacked to disk. Each trace is decompressed and tested against cheap byte
patterns (two-decimal latitude and longitude prefixes of each airport's bounding box, taken from the repository's own
OSM geometry files, padded by 0.01 degrees). Only traces that hit both a latitude and a longitude pattern of the same
airport are parsed as JSON, which is a small fraction of the roughly 300,000 aircraft in a day.

readsb trace entries are [offset_seconds, lat, lon, altitude, ground_speed, track, flags, vertical_rate, aircraft,
source, ...], where altitude is the string "ground" while on the surface and `aircraft` carries the callsign under
"flight". Points outside every box are dropped.

Output: one parquet per day with hex, callsign, timestamp, latitude, longitude, on_ground, ground_speed, airport.

Run:  python3 -B extract_tracks.py --parts raw/part.aa raw/part.ab --date 2025-07-13 --out tracks/2025-07-13.parquet
"""
import argparse
import gzip
import io
import json
import re
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import pandas as pd

GEOMETRY = Path("geometry")
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
PAD = 0.01


class Chained(io.RawIOBase):
    """Read several files as one stream, so the split parts need no 3.3 GB concatenation on disk."""

    def __init__(self, paths):
        self.paths, self.index = [str(p) for p in paths], 0
        self.handle = open(self.paths[0], "rb")

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            count = self.handle.readinto(buffer)
            if count:
                return count
            self.handle.close()
            self.index += 1
            if self.index >= len(self.paths):
                return 0
            self.handle = open(self.paths[self.index], "rb")


def boxes():
    out = {}
    for path in sorted(GEOMETRY.glob("*_osm.json")):
        data = json.loads(path.read_text())
        icao = data["icao"].upper()
        if icao not in AIRPORTS:
            continue
        south, west, north, east = data["bbox"]
        out[icao] = (south - PAD, west - PAD, north + PAD, east + PAD)
    missing = set(AIRPORTS) - set(out)
    if missing:
        raise SystemExit(f"geometry missing for {sorted(missing)}")
    return out


def patterns(box):
    """One regex per airport, matching a latitude and longitude that stand NEXT TO EACH OTHER in a trace entry.

    readsb writes each point as [offset, lat, lon, altitude, ...], so a real hit looks like ",48.3512,11.7863,".
    Testing the two prefixes separately anywhere in the file passes four traces in five, because any long flight
    carries many coordinates; requiring adjacency is what makes the pre-filter selective.
    """
    def span(low, high):
        values = np.arange(np.floor(low * 100) / 100, high + 0.01, 0.01)
        return sorted({f"{v:.2f}" for v in values if low - 0.01 <= v <= high + 0.01})
    south, west, north, east = box
    lats = "|".join(re.escape(p) for p in span(south, north))
    lons = "|".join(re.escape(p) for p in span(west, east))
    return re.compile((rf",\s*(?:{lats})\d*,\s*(?:{lons})\d*,").encode())


def scan(parts, out_path, limit=None):
    box = boxes()
    filters = {icao: patterns(b) for icao, b in box.items()}
    rows, seen, parsed, started = [], 0, 0, time.time()
    stream = Chained(parts)
    with tarfile.open(fileobj=io.BufferedReader(stream, 1 << 22), mode="r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            seen += 1
            if limit and seen > limit:
                break
            raw = archive.extractfile(member).read()
            if raw[:2] == b"\x1f\x8b":
                try:
                    raw = gzip.decompress(raw)
                except (OSError, EOFError):
                    continue
            hits = [icao for icao, rx in filters.items() if rx.search(raw)]
            if not hits:
                continue
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            parsed += 1
            trace = record.get("trace") or []
            base = record.get("timestamp")
            if base is None or not trace:
                continue
            hexid = record.get("icao")
            callsign = None
            for entry in trace:
                if len(entry) > 8 and isinstance(entry[8], dict):
                    flight = entry[8].get("flight")
                    if flight:
                        callsign = flight.strip()
                latitude, longitude = entry[1], entry[2]
                if latitude is None or longitude is None:
                    continue
                for icao in hits:
                    south, west, north, east = box[icao]
                    if south <= latitude <= north and west <= longitude <= east:
                        rows.append((hexid, callsign, base + entry[0], latitude, longitude,
                                     entry[3] == "ground", entry[4], icao))
                        break
            if seen % 50000 == 0:
                print(f"  {seen:,} traces, {parsed:,} parsed, {len(rows):,} points, "
                      f"{time.time() - started:.0f}s", flush=True)
    frame = pd.DataFrame(rows, columns=["hex", "callsign", "ts", "lat", "lon", "on_ground", "gs", "airport"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    print(f"traces {seen:,}, parsed {parsed:,} ({parsed / max(seen, 1):.1%}), points {len(frame):,}, "
          f"{time.time() - started:.0f}s -> {out_path}", flush=True)
    if len(frame):
        print(frame.groupby("airport").agg(points=("ts", "size"), aircraft=("hex", "nunique"),
                                           on_ground=("on_ground", "mean")).round(3).to_string())
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    scan([Path(p) for p in args.parts], Path(args.out), args.limit)
