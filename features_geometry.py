"""features_geometry.py - Rollstrecke Standplatz zu Bahnschwelle aus OpenStreetMap.

Baut aus den Dateien in geometry/ zwei Nachschlagewerke je Flughafen:

  1. Standplatz -> Mittelpunkt. Die Kennungen werden normalisiert, weil OSM und
     der Wettbewerbsdatensatz verschieden schreiben. Gemessen am 02.09.2026 hebt
     das den Abgleich ueber alle Flughaefen von 87 auf 94 Prozent:
       EDDF 70 -> 96 Prozent, weil OSM "V 106" schreibt und der Datensatz "V106"
       EHAM 69 -> 95 Prozent, weil OSM "H1" schreibt und der Datensatz "H01"

  2. Bahnrichtung -> Schwelle. OSM fuehrt eine Bahn als Paar "08R/26L". Welches
     Ende die Schwelle fuer eine Richtung ist, ergibt sich aus dem Kurs: von der
     Schwelle 26L aus zeigt die Bahn nach rund 260 Grad.

Die Rollstrecke ist die Grosskreisentfernung zwischen beiden Punkten, also die
Luftlinie. Der tatsaechliche Rollweg ist laenger, aber am BER folgte die Rolldauer
der Entfernung bei praktisch konstanter Geschwindigkeit von 7,1 m/s. Fuer ein
Merkmal zaehlt der monotone Zusammenhang, nicht der Absolutwert.

Quelle der Geometrie: OpenStreetMap-Mitwirkende, ODbL.

Aufruf:
    python3 features_geometry.py            Selbsttest gegen die Lerndaten
"""
import json, glob, math, os, re, sys

GEOM = "geometry"
ERDRADIUS_M = 6371008.8


# ---------------------------------------------------------------- Kennungen

def stand_schluessel(s):
    """Normalform einer Standplatzkennung: Grossbuchstaben, ohne Trenner,
    ohne fuehrende Nullen in Zifferngruppen. H01 -> H1, V 106 -> V106."""
    if not s:
        return ""
    s = s.strip().upper().replace(" ", "").replace("-", "")
    return re.sub(r"(?<=\D)0+(\d)", r"\1", s)


def stand_kandidaten(ref):
    """Alle Normalformen, unter denen eine OSM-Kennung auffindbar sein soll.
    OSM fasst Doppelstaende zusammen, als "G1A/G1F" oder als "1 1A"."""
    r = ref.strip().upper()
    k = {stand_schluessel(r)}
    for teil in r.split("/"):
        k.add(stand_schluessel(teil))
    # Nur trennen, wenn beide Teile eine Ziffer tragen. Sonst zerfiele
    # "V 106" faelschlich in "V" und "106".
    if " " in r and all(any(c.isdigit() for c in t) for t in r.split()):
        for teil in r.split():
            k.add(stand_schluessel(teil))
    return {x for x in k if x}


# ---------------------------------------------------------------- Geometrie

def mittelpunkt(el):
    """Mittelpunkt eines OSM-Objekts. Knoten tragen lat/lon direkt,
    Wege tragen nach 'out geom' eine Punktliste."""
    if el.get("type") == "node" and "lat" in el:
        return el["lat"], el["lon"]
    g = el.get("geometry") or []
    if not g:
        return None
    return sum(p["lat"] for p in g) / len(g), sum(p["lon"] for p in g) / len(g)


def entfernung_m(a, b):
    """Grosskreisentfernung in Metern, Haversine."""
    p1, l1, p2, l2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2
    return 2 * ERDRADIUS_M * math.asin(math.sqrt(h))


def kurs_grad(a, b):
    """Rechtweisender Kurs von a nach b in Grad."""
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def winkelabstand(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


# ---------------------------------------------------------------- Aufbau

def lade(icao):
    """Nachschlagewerke fuer einen Flughafen: Staende und Bahnschwellen."""
    pfad = os.path.join(GEOM, f"{icao.lower()}_osm.json")
    if not os.path.exists(pfad):
        return None
    d = json.load(open(pfad))
    staende, bahnen = {}, {}
    for el in d.get("elements", []):
        t = el.get("tags", {})
        art = t.get("aeroway")
        if art == "parking_position" and t.get("ref"):
            m = mittelpunkt(el)
            if m:
                for k in stand_kandidaten(t["ref"]):
                    staende.setdefault(k, m)
        elif art == "runway" and t.get("ref"):
            g = el.get("geometry") or []
            if len(g) < 2:
                continue
            enden = ((g[0]["lat"], g[0]["lon"]), (g[-1]["lat"], g[-1]["lon"]))
            for richtung in t["ref"].split("/"):
                richtung = richtung.strip().upper()
                zahl = re.match(r"(\d{1,2})", richtung)
                if not zahl:
                    continue
                soll = int(zahl.group(1)) * 10
                # Schwelle ist das Ende, von dem aus die Bahn in Sollrichtung zeigt.
                a_b = kurs_grad(enden[0], enden[1])
                schwelle = enden[0] if winkelabstand(a_b, soll) < 90 else enden[1]
                # Bei mehreren Wegteilen die aeussere Schwelle behalten.
                if richtung in bahnen:
                    gegen = enden[1] if schwelle is enden[0] else enden[0]
                    if entfernung_m(schwelle, gegen) < entfernung_m(bahnen[richtung], gegen):
                        continue
                bahnen[richtung] = schwelle
    return {"icao": icao, "staende": staende, "bahnen": bahnen}


def alle():
    werke = {}
    for p in sorted(glob.glob(os.path.join(GEOM, "*_osm.json"))):
        icao = os.path.basename(p)[:4].upper()
        w = lade(icao)
        if w:
            werke[icao] = w
    return werke


def rollstrecke_m(werk, stand, bahn):
    """Luftlinie Standplatz zu Bahnschwelle in Metern, None wenn unbekannt."""
    s = werk["staende"].get(stand_schluessel(stand or ""))
    b = werk["bahnen"].get((bahn or "").strip().upper())
    return entfernung_m(s, b) if s and b else None


# ---------------------------------------------------------------- Selbsttest

def selbsttest():
    import pyarrow.parquet as pq
    werke = alle()
    print(f"{'ICAO':6} {'Staende':>8} {'Bahnen':>7}   Bahnrichtungen")
    print("-" * 74)
    for a, w in sorted(werke.items()):
        print(f"{a:6} {len(w['staende']):>8} {len(w['bahnen']):>7}   "
              f"{', '.join(sorted(w['bahnen']))[:44]}")

    dateien = sorted(glob.glob("data/training_*.parquet"))
    if not dateien:
        print("\nKeine Lerndaten gefunden, Abgleich uebersprungen.")
        return
    t = pq.read_table(dateien[5], columns=["ADEP_mvt", "PHASE_mvt", "STAND_mvt",
                                           "RUNWAY_mvt", "TAXITIME_SEC_mvt"]).to_pydict()
    treffer = {}
    for i in range(len(t["PHASE_mvt"])):
        if t["PHASE_mvt"][i] != "DEP":
            continue
        a = t["ADEP_mvt"][i]
        w = werke.get(a)
        if not w:
            continue
        d = rollstrecke_m(w, t["STAND_mvt"][i], t["RUNWAY_mvt"][i])
        e = treffer.setdefault(a, {"n": 0, "ok": 0, "summe": 0.0, "paare": []})
        e["n"] += 1
        if d is not None:
            e["ok"] += 1
            e["summe"] += d
            e["paare"].append((d, t["TAXITIME_SEC_mvt"][i]))

    print(f"\n{os.path.basename(dateien[5])}")
    print(f"{'ICAO':6} {'Abfluege':>9} {'mit Strecke':>12} {'Median m':>9} "
          f"{'Median s':>9} {'m/s':>6} {'Korrel.':>8}")
    print("-" * 68)
    for a in sorted(treffer):
        e = treffer[a]
        if not e["paare"]:
            print(f"{a:6} {e['n']:>9} {'0':>12}")
            continue
        ds = sorted(p[0] for p in e["paare"])
        ts = sorted(p[1] for p in e["paare"])
        md, mt = ds[len(ds) // 2], ts[len(ts) // 2]
        n = len(e["paare"])
        mx = sum(p[0] for p in e["paare"]) / n
        my = sum(p[1] for p in e["paare"]) / n
        sx = math.sqrt(sum((p[0] - mx) ** 2 for p in e["paare"]) / n)
        sy = math.sqrt(sum((p[1] - my) ** 2 for p in e["paare"]) / n)
        r = (sum((p[0] - mx) * (p[1] - my) for p in e["paare"]) / n / (sx * sy)) if sx and sy else 0
        print(f"{a:6} {e['n']:>9} {e['ok']/e['n']*100:>11.0f}% {md:>9.0f} "
              f"{mt:>9.0f} {md/mt if mt else 0:>6.1f} {r:>8.2f}")


if __name__ == "__main__":
    selbsttest()
