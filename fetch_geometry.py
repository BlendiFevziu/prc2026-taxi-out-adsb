"""fetch_geometry.py - Flughafengeometrie der elf Wettbewerbsflughaefen von OpenStreetMap.

Holt je Flughafen Bahnen, Rollwege, Vorfelder und Abstellpositionen ueber die
Overpass-API.

Zwei Schritte je Flughafen, weil das direkte Umwandeln des Flughafenumrisses in
ein Suchgebiet bei einigen Flughaefen fehlschlaegt:
  1. Umgrenzung des Flughafens ueber den ICAO-Code holen
  2. innerhalb dieser Umgrenzung die Bewegungsflaechen abfragen

Aufruf:
    python3 fetch_geometry.py            alle elf
    python3 fetch_geometry.py EDDF LSZH  nur einzelne
"""
import json, os, sys, time, urllib.request, urllib.parse

URL = "https://overpass-api.de/api/interpreter"
ZIEL = "geometry"
PAUSE = 8          # Sekunden zwischen Abfragen, hoeflich gegenueber der freien API

FLUGHAEFEN = {
    "EDDF": "Frankfurt am Main",
    "EDDM": "Muenchen",
    "EGLL": "London Heathrow",
    "EHAM": "Amsterdam Schiphol",
    "LEBL": "Barcelona El Prat",
    "LEMD": "Madrid Barajas",
    "LFPG": "Paris Charles de Gaulle",
    "LIRF": "Rom Fiumicino",
    "LTAI": "Antalya",
    "LTFM": "Istanbul",
    "LSZH": "Zuerich",
}


def frage(ql, versuche=3):
    for n in range(versuche):
        try:
            daten = urllib.parse.urlencode({"data": ql}).encode()
            # Overpass antwortet ohne eigenen User-Agent mit HTTP 406.
            kopf = {"User-Agent": "upbeat-goblin/1.0 (PRC Data Challenge 2026)"}
            with urllib.request.urlopen(
                    urllib.request.Request(URL, daten, headers=kopf), timeout=300) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if n == versuche - 1:
                raise
            print(f"      Versuch {n+1} fehlgeschlagen ({e}), warte 30 s", flush=True)
            time.sleep(30)


def umgrenzung(icao):
    """Bounding Box des Flughafens ueber den ICAO-Code."""
    d = frage(f'[out:json][timeout:120];'
              f'nwr["aeroway"="aerodrome"]["icao"="{icao}"];out bb;')
    boxen = [e["bounds"] for e in d.get("elements", []) if "bounds" in e]
    if not boxen:
        return None
    return (min(b["minlat"] for b in boxen), min(b["minlon"] for b in boxen),
            max(b["maxlat"] for b in boxen), max(b["maxlon"] for b in boxen))


def flaechen(bbox):
    s, w, n, e = bbox
    # Abstellpositionen sind je nach Flughafen als Weg ODER als Knoten erfasst.
    # Ohne die Knoten fehlten am 02.09.2026 in EGLL 68, in EHAM 71 und in LEBL 40 Staende.
    ql = (f'[out:json][timeout:300];'
          f'( way["aeroway"~"^(runway|taxiway|apron|parking_position|terminal)$"]({s},{w},{n},{e});'
          f'  node["aeroway"="parking_position"]({s},{w},{n},{e});'
          f'  relation["aeroway"~"^(apron|terminal)$"]({s},{w},{n},{e}); );'
          f'out geom;')
    return frage(ql)


def main(codes):
    os.makedirs(ZIEL, exist_ok=True)
    zusammen = {}
    for i, icao in enumerate(codes, 1):
        pfad = os.path.join(ZIEL, f"{icao.lower()}_osm.json")
        if os.path.exists(pfad):
            print(f"[{i}/{len(codes)}] {icao} liegt schon vor, uebersprungen")
            continue
        print(f"[{i}/{len(codes)}] {icao} {FLUGHAEFEN.get(icao,'')}", flush=True)
        bb = umgrenzung(icao)
        if bb is None:
            print("      KEINE Umgrenzung gefunden, uebersprungen")
            continue
        print(f"      Umgrenzung {bb[0]:.4f},{bb[1]:.4f} bis {bb[2]:.4f},{bb[3]:.4f}", flush=True)
        time.sleep(PAUSE)
        d = flaechen(bb)
        el = d.get("elements", [])
        json.dump({"icao": icao, "name": FLUGHAEFEN.get(icao), "bbox": bb, **d},
                  open(pfad, "w"))
        arten = {}
        for x in el:
            arten[x.get("tags", {}).get("aeroway", "?")] = \
                arten.get(x.get("tags", {}).get("aeroway", "?"), 0) + 1
        zusammen[icao] = arten
        print(f"      {len(el)} Objekte: " +
              ", ".join(f"{k} {v}" for k, v in sorted(arten.items())), flush=True)
        time.sleep(PAUSE)
    return zusammen


if __name__ == "__main__":
    codes = [c.upper() for c in sys.argv[1:]] or list(FLUGHAEFEN)
    main(codes)
