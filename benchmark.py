"""benchmark.py - Einfache Vergleichswerte fuer die Taxi-out-Vorhersage.

Ohne Messlatte laesst sich kein Modellgewinn einordnen. Dieses Skript rechnet
den RMSE mehrerer trivialer Vorhersagen.

Aufteilung: Bewertet wird beim Wettbewerb ueber Januar und Juli 2026. Als Nachbau
dienen hier Januar und Juli 2025 als Pruefmonate, die uebrigen zehn Monate als
Lernmenge. Damit bleibt die jahreszeitliche Struktur erhalten und es wird nie
mit Wissen aus dem Pruefzeitraum gelernt.

Wichtig: Der RMSE wird vom MITTELWERT minimiert, nicht vom Median. Beides steht
unten nebeneinander, weil der Median die anschaulichere, aber fuer diese
Bewertung schlechtere Wahl ist.

Aufruf:
    python3 benchmark.py
"""
import glob, math, os, statistics, sys, collections
import pyarrow.parquet as pq

SPALTEN = ["ADEP_mvt", "PHASE_mvt", "STAND_mvt", "RUNWAY_mvt",
           "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt"]
PRUEFMONATE = {"2025-01", "2025-07"}


def lade():
    lern, pruef = [], []
    for f in sorted(glob.glob("data/training_*.parquet")):
        monat = os.path.basename(f).split("_")[1][:7]
        t = pq.read_table(f, columns=SPALTEN).to_pydict()
        ziel = pruef if monat in PRUEFMONATE else lern
        for i in range(len(t["PHASE_mvt"])):
            if t["PHASE_mvt"][i] != "DEP":
                continue
            y = t["TAXITIME_SEC_mvt"][i]
            if y is None:
                continue
            ziel.append({
                "ap": t["ADEP_mvt"][i],
                "rw": (t["RUNWAY_mvt"][i] or "").strip().upper(),
                "st": (t["STAND_mvt"][i] or "").strip().upper(),
                "typ": t["AIRCRAFT_TYPE_mvt"][i] or "",
                "std": t["MVT_TIME_UTC_mvt"][i].hour,
                "y": y,
            })
    return lern, pruef


def rmse(paare):
    return math.sqrt(sum((v - y) ** 2 for v, y in paare) / len(paare))


def gruppiere(daten, schluessel, mittel):
    g = collections.defaultdict(list)
    for r in daten:
        g[schluessel(r)].append(r["y"])
    return {k: mittel(v) for k, v in g.items()}


def bewerte(name, lern, pruef, schluessel, mittel, rueckfall):
    tab = gruppiere(lern, schluessel, mittel)
    glob_wert = mittel([r["y"] for r in lern])
    paare, getroffen = [], 0
    for r in pruef:
        k = schluessel(r)
        if k in tab:
            v = tab[k]; getroffen += 1
        else:
            v = rueckfall.get(r["ap"], glob_wert) if rueckfall else glob_wert
        paare.append((v, r["y"]))
    return name, rmse(paare), getroffen / len(pruef) * 100, len(tab)


def main():
    lern, pruef = lade()
    print(f"Lernmenge {len(lern):,} Abfluege aus zehn Monaten")
    print(f"Pruefmenge {len(pruef):,} Abfluege aus Januar und Juli 2025")
    ys = [r["y"] for r in pruef]
    print(f"Zielgroesse in der Pruefmenge: Mittel {statistics.mean(ys):.0f} s, "
          f"Median {statistics.median(ys):.0f} s, "
          f"Streuung {statistics.pstdev(ys):.0f} s\n")

    for mittelname, mittel in (("Mittelwert", statistics.mean),
                               ("Median    ", statistics.median)):
        ap_tab = gruppiere(lern, lambda r: r["ap"], mittel)
        zeilen = [
            bewerte("global", lern, pruef, lambda r: "*", mittel, None),
            bewerte("Flughafen", lern, pruef, lambda r: r["ap"], mittel, None),
            bewerte("Flughafen + Bahn", lern, pruef,
                    lambda r: (r["ap"], r["rw"]), mittel, ap_tab),
            bewerte("Flughafen + Bahn + Stunde", lern, pruef,
                    lambda r: (r["ap"], r["rw"], r["std"]), mittel, ap_tab),
            bewerte("Flughafen + Bahn + Stand", lern, pruef,
                    lambda r: (r["ap"], r["rw"], r["st"]), mittel, ap_tab),
            bewerte("Flughafen + Bahn + Stand + Stunde", lern, pruef,
                    lambda r: (r["ap"], r["rw"], r["st"], r["std"]), mittel, ap_tab),
        ]
        print(f"=== Vorhersage je Gruppe ueber den {mittelname.strip()} ===")
        print(f"{'Gruppierung':34} {'RMSE s':>8} {'getroffen':>10} {'Gruppen':>9}")
        print("-" * 64)
        for n, r, g, k in zeilen:
            print(f"{n:34} {r:>8.1f} {g:>9.0f}% {k:>9,}")
        print()

    # RMSE je Flughafen fuer die beste einfache Regel
    ap_tab = gruppiere(lern, lambda r: r["ap"], statistics.mean)
    bahn = gruppiere(lern, lambda r: (r["ap"], r["rw"]), statistics.mean)
    proAP = collections.defaultdict(list)
    for r in pruef:
        v = bahn.get((r["ap"], r["rw"]), ap_tab.get(r["ap"]))
        proAP[r["ap"]].append((v, r["y"]))
    print("=== RMSE je Flughafen, Regel Flughafen + Bahn ueber den Mittelwert ===")
    print(f"{'ICAO':6} {'Abfluege':>9} {'RMSE s':>8} {'Mittel s':>9} {'Streuung s':>11}")
    print("-" * 48)
    for a in sorted(proAP, key=lambda x: -rmse(proAP[x])):
        p = proAP[a]
        ys = [y for _, y in p]
        print(f"{a:6} {len(p):>9,} {rmse(p):>8.1f} {statistics.mean(ys):>9.0f} "
              f"{statistics.pstdev(ys):>11.0f}")


if __name__ == "__main__":
    main()
