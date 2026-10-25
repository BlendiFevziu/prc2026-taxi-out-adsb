"""Stand positions from the X-Plane Scenery Gateway: the recommended (latest approved) apt.dat of each airport.

For every airport: GET /apiv1/airport/{ICAO} for recommendedSceneryId, require that scenery's status to be Approved,
GET /apiv1/scenery/{id}, base64-decode masterZipBlob, keep the zip in .cache/xplane_gateway/ (never committed), and
parse the apt.dat rows 1300 (startup location: lat, lon, heading, type, aircraft classes, name) and 1301 (ramp
metadata that follows its 1300: ICAO size class, operation type, airlines). The parsed tables go to
research_300/geometry/xplane_{ICAO}.json with their provenance.

Run from the repository root:  python3 -u -B research_300/xplane_stands.py [ICAO ...]
"""
import base64
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/xplane_gateway"
OUT = ROOT / "research_300/geometry"
API = "https://gateway.x-plane.com/apiv1"
AIRPORTS = ("EDDM", "EHAM", "EDDF")


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode())


def parse(text):
    stands = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "1300" and len(parts) >= 6:
            stands.append({"lat": float(parts[1]), "lon": float(parts[2]), "heading": float(parts[3]),
                           "type": parts[4], "classes": parts[5], "name": " ".join(parts[6:])})
        elif parts[0] == "1301" and stands:
            stands[-1].update({"icao_size": parts[1] if len(parts) > 1 else None,
                               "operation": parts[2] if len(parts) > 2 else None,
                               "airlines": parts[3:] if len(parts) > 3 else []})
    return stands


def fetch(icao):
    airport = get(f"{API}/airport/{icao}")["airport"]
    sid = airport["recommendedSceneryId"]
    entry = next((s for s in airport["scenery"] if s.get("sceneryId") == sid), {})
    if entry.get("Status") != "Approved":
        raise SystemExit(f"{icao}: recommended scenery {sid} is not Approved ({entry.get('Status')})")
    scenery = get(f"{API}/scenery/{sid}")["scenery"]
    blob = base64.b64decode(scenery["masterZipBlob"])
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{icao}_{sid}.zip").write_bytes(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".dat"))
        text = z.read(name).decode("utf-8", errors="replace")
    stands = parse(text)
    meta = {k: v for k, v in scenery.items() if k != "masterZipBlob" and not isinstance(v, (dict, list))}
    record = {"icao": icao, "scenery_id": sid, "status": entry.get("Status"),
              "date_accepted": entry.get("dateAccepted"), "dat_file": name, "stands": stands,
              "scenery_meta": meta, "source": f"{API}/scenery/{sid}"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"xplane_{icao}.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def main():
    for icao in (sys.argv[1:] or AIRPORTS):
        r = fetch(icao)
        types = {}
        for s in r["stands"]:
            types[s["type"]] = types.get(s["type"], 0) + 1
        print(icao, "scenery", r["scenery_id"], r["status"], r["date_accepted"], "| stands", len(r["stands"]), types,
              "| meta keys", sorted(r["scenery_meta"])[:12], flush=True)


if __name__ == "__main__":
    main()
