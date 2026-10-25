"""Gate 0 for the Munich stand detector: is there ADS-B ground coverage on the SCORED dates at all?

Why this runs first. Codex's diagnostic on 13 December 2025 (private-nm-archive/adsb_munich_audit, reported in
full-month-research/CLAUDE_LUIS_REVIEW_2026-09-23.md) found 344 matched Munich departures, 234 with a ground point
within 60 s of the airport's block time. But on 13 July 2025 the same adsb.lol feed held 17,380 Munich points and
ZERO on the ground (adsb-lol/offblock_from_tracks.py docstring; STATUS.md 21 September). Coverage therefore moves
over time, and a detector can change the score only where the scored dates, January and July 2026, have it.

What it reads. Two scored dates, 2026-07-06 and 2026-07-13, of which only the first part of each adsb.lol release
was ever downloaded (1.43 and 1.38 GB of about 3.3 GB). A release is a tar of one trace per aircraft in an order
unrelated to airports, so a truncated tar is a sample of aircraft, not of airports. The scan stops cleanly at the
truncation and records how many traces it read. The complete 2025-07-13 extraction is summarised alongside as the
known-zero reference. No download is made.

Reuses boxes(), patterns() and Chained from adsb-lol/extract_tracks.py unchanged; that file is not modified.
Row-level tracks are written to adsb-lol/tracks/ (outside this repository); only aggregates are written here.
Source: adsb.lol daily releases, ODbL 1.0, attribution required.

Run from the repository root:  python3 -u -B research_290/adsb_coverage_gate.py
"""
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
import time

import pandas as pd

ADSB = Path("adsb-lol")
sys.path.insert(0, str(ADSB))
import extract_tracks as ET  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "results/adsb_coverage_gate.json"
PARTIAL = {"2026-07-06": ADSB / "raw/2026-07-06/part.aa", "2026-07-13": ADSB / "raw/2026-07-13/part.aa"}
REFERENCE = {"2025-07-13": ADSB / "tracks/2025-07-13.parquet"}
SLOW_KT = 5.0


def tolerant_scan(part, out_path):
    """extract_tracks.scan, but a truncated archive ends the loop instead of discarding everything read."""
    box = ET.boxes()
    filters = {icao: ET.patterns(b) for icao, b in box.items()}
    rows, seen, parsed, truncated, started = [], 0, 0, False, time.time()
    # A plain file, not ET.Chained: Chained closes its handle at the end of the last part and then fails on the
    # extra read tarfile makes when a member is cut short ("readinto of closed file"). A plain file returns b""
    # at the end, so tarfile raises its own truncation error, which is caught below.
    try:
        with open(part, "rb", buffering=1 << 22) as handle, tarfile.open(fileobj=handle, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                seen += 1
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
                base, trace, callsign = record.get("timestamp"), record.get("trace") or [], None
                if base is None:
                    continue
                for entry in trace:
                    if len(entry) > 8 and isinstance(entry[8], dict) and entry[8].get("flight"):
                        callsign = entry[8]["flight"].strip()
                    lat, lon = entry[1], entry[2]
                    if lat is None or lon is None:
                        continue
                    for icao in hits:
                        s, w, n, e = box[icao]
                        if s <= lat <= n and w <= lon <= e:
                            rows.append((record.get("icao"), callsign, base + entry[0], lat, lon,
                                         entry[3] == "ground", entry[4], icao))
                            break
                if seen % 50000 == 0:
                    print(f"  {seen:,} traces, {len(rows):,} points, {time.time() - started:.0f}s", flush=True)
    except (tarfile.ReadError, EOFError, OSError) as error:
        truncated = True
        print(f"  archive ends after {seen:,} traces ({type(error).__name__}), keeping what was read", flush=True)
    frame = pd.DataFrame(rows, columns=["hex", "callsign", "ts", "lat", "lon", "on_ground", "gs", "airport"])
    frame.to_parquet(out_path, index=False)
    return frame, {"traces_read": seen, "traces_parsed": parsed, "truncated": truncated,
                   "seconds": round(time.time() - started, 1)}


def summary(frame):
    out = {}
    for airport, g in frame.groupby("airport"):
        slow = g.on_ground & (pd.to_numeric(g.gs, errors="coerce") < SLOW_KT)
        out[airport] = {"points": int(len(g)), "aircraft": int(g.hex.nunique()),
                        "on_ground_points": int(g.on_ground.sum()), "on_ground_share": round(float(g.on_ground.mean()), 4),
                        "aircraft_with_ground_points": int(g.loc[g.on_ground, "hex"].nunique()),
                        "aircraft_seen_slow_on_ground": int(g.loc[slow, "hex"].nunique())}
    return out


def main():
    result = {"note": "ADS-B ground coverage on scored July 2026 dates (truncated archives) and the complete "
                      "2025-07-13 reference; aggregates only", "slow_threshold_kt": SLOW_KT, "dates": {}}
    for date, path in REFERENCE.items():
        result["dates"][date] = {"source": "complete extraction", "airports": summary(pd.read_parquet(path))}
    for date, part in PARTIAL.items():
        print(f"{date}: scanning {part.stat().st_size / 1e9:.2f} GB", flush=True)
        frame, info = tolerant_scan(part, ADSB / f"tracks/{date}-partial.parquet")
        result["dates"][date] = {"source": "truncated first part", "bytes": part.stat().st_size, **info,
                                 "airports": summary(frame)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n{'airport':<6}" + "".join(f"{d:>26}" for d in result["dates"]))
    print(f"{'':<6}" + "".join(f"{'aircraft / on ground':>26}" for _ in result["dates"]))
    for airport in ET.AIRPORTS:
        cells = []
        for d in result["dates"]:
            a = result["dates"][d]["airports"].get(airport)
            cells.append(f"{a['aircraft']:>6} / {a['aircraft_with_ground_points']:>5}" if a else "-")
        print(f"{airport:<6}" + "".join(f"{c:>26}" for c in cells))
    print(f"written {OUTPUT}")


if __name__ == "__main__":
    main()
