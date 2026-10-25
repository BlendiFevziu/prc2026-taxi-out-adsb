"""framework.py - Merkmalsspeicher, Vorhersagespeicher und Versuchsprotokoll.

Warum es das gibt, gemessen am 05.09.2026:

- `hauptmodell.baue()` liest vierzehn Parquet-Dateien und rechnet Umlauf, Anker und
  Laufnummern durch. Das dauert zwei bis drei Minuten und lief an einem Tag fuenfzehnmal
  fuer dasselbe Ergebnis.
- Fuer die Suche nach dem besten CatBoost-Anteil wurde CatBoost dreimal gelernt, 93 Minuten
  je Saat, obwohl die Gewichtssuche selbst Sekunden dauert. Mit gespeicherten Vorhersagen
  waere derselbe Nachmittag in zwanzig Minuten erledigt gewesen.
- Was schon gemessen wurde, stand nur in der README als Fliesstext.

Der Merkmalsspeicher wird ueber eine Pruefsumme der beteiligten Programmdateien
geschluesselt. Aendert sich eine davon, faellt der Speicher von selbst weg, es kann also
nicht passieren, dass gegen veraltete Merkmale gemessen wird.

Benutzung:

    import framework as G
    d = G.daten()                       # gebaut oder aus dem Speicher, mit Meldung
    lern, pruef = G.teile(d)
    v = G.vorhersage("lgb_5saaten", lambda: baue_und_sage_vorher(), pruef)
    G.notiere("CatBoost-Anteil 0,55", haupt=226.55, rueck=1747.1)
"""
import hashlib
import os
import pickle
import time

import numpy as np
import pandas as pd

SPEICHER = ".cache"
PROTOKOLL = "experiments.csv"
# Aendert sich eine dieser Dateien, ist der Merkmalsspeicher ungueltig.
BAUDATEIEN = ["main_model.py", "features_geometry.py", "features_turnaround.py",
              "features_arrivals.py", "features_sequence.py", "features_flight_sequence.py",
              "features_stand_rate.py"]


def _pruefsumme(dateien):
    h = hashlib.sha256()
    for f in sorted(dateien):
        if os.path.exists(f):
            with open(f, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def daten(neu=False):
    """Die gebaute Merkmalstabelle, aus dem Speicher wenn die Programme unveraendert sind."""
    import glob
    import main_model as H
    import features_geometry as MG
    schluessel = _pruefsumme(BAUDATEIEN)
    pfad = os.path.join(SPEICHER, f"data_{schluessel}.pkl")
    if os.path.exists(pfad) and not neu:
        t0 = time.time()
        d = pickle.load(open(pfad, "rb"))
        print(f"  Merkmale aus dem Speicher, {len(d):,} Zeilen ({time.time()-t0:.0f} s)")
        return d
    t0 = time.time()
    d = H.baue(sorted(glob.glob("data/training_2025-*.parquet")), MG.alle())
    d = d[d.y.notna()].copy()
    os.makedirs(SPEICHER, exist_ok=True)
    pickle.dump(d, open(pfad, "wb"))
    print(f"  Merkmale neu gebaut, {len(d):,} Zeilen ({time.time()-t0:.0f} s), "
          f"Speicher {schluessel}")
    # aeltere Staende wegraeumen, sie sind nach einer Programmaenderung wertlos
    for alt in glob.glob(os.path.join(SPEICHER, "data_*.pkl")):
        if alt != pfad:
            os.remove(alt)
    return d


def teile(d):
    """Lern- und Pruefmenge nach den Pruefmonaten aus hauptmodell."""
    import main_model as H
    return d[~d.monat.isin(H.PRUEFMONATE)].copy(), d[d.monat.isin(H.PRUEFMONATE)].copy()


def vorhersage(name, bauen, neu=False):
    """Vorhersage eines Modells auf der Pruefmenge, gespeichert unter `name`.

    `bauen` ist eine Funktion ohne Argumente, die das Feld zurueckgibt. Sie wird nur
    aufgerufen, wenn nichts gespeichert ist. Der Name muss den Merkmalssatz mit
    beschreiben, sonst vergleicht man spaeter Ungleiches.
    """
    schluessel = _pruefsumme(BAUDATEIEN)
    pfad = os.path.join(SPEICHER, f"v_{schluessel}_{name}.npy")
    if os.path.exists(pfad) and not neu:
        print(f"  '{name}' aus dem Speicher")
        return np.load(pfad)
    t0 = time.time()
    v = np.asarray(bauen(), dtype=float)
    os.makedirs(SPEICHER, exist_ok=True)
    np.save(pfad, v)
    print(f"  '{name}' gerechnet und gespeichert ({time.time()-t0:.0f} s)")
    return v


def rmse(p, y):
    return float(np.sqrt(((np.asarray(p, float) - np.asarray(y, float)) ** 2).mean()))


def gesamt(haupt, rueck, n_haupt=339046, n_rueck=5373):
    """Zusammengesetzter Wert aus den beiden Gruppenfehlern.

    ACHTUNG, 06.09.2026: dieser Wert ist als Vorhersage der Rangliste UNBRAUCHBAR,
    solange die Rueckfallgruppe ungetrimmt eingeht. Siehe `rmse_getrimmt`.
    """
    return float(np.sqrt((n_haupt * haupt ** 2 + n_rueck * rueck ** 2) / (n_haupt + n_rueck)))


def rmse_getrimmt(p, y, k=1):
    """RMSE ohne die k groessten Einzelfehler.

    **Warum das noetig ist, gemessen am 06.09.2026.** Die Halteprobe enthaelt EINE Zeile
    (LFPG, y = 84.240 s, also fast ein ganzer Tag) mit 42,5 % des Fehlers der
    Rueckfallgruppe. Sie allein hebt den zusammengesetzten Wert um 34 Sekunden:

        Halteprobe ungetrimmt        312,89
        Halteprobe ohne eine Zeile   278,70
        echtes Ergebnis v17          277,02

    Ungetrimmt misst man also vor allem, ob diese eine Zeile zufaellig etwas besser
    getroffen wurde. Getrimmt sagt die Halteprobe die Rangliste auf 1,7 s genau voraus.
    Fuer JEDE Entscheidung den getrimmten Wert nehmen, den ungetrimmten hoechstens als
    Fussnote.
    """
    e = np.sort((np.asarray(p, float) - np.asarray(y, float)) ** 2)[::-1]
    return float(np.sqrt(e[k:].mean()))


def notiere(was, haupt=None, rueck=None, gesamtwert=None, bemerkung=""):
    """Eine Zeile ins Versuchsprotokoll. Doppelte Beschreibungen werden nicht ersetzt,
    damit die Geschichte lesbar bleibt."""
    if gesamtwert is None and haupt is not None and rueck is not None:
        gesamtwert = gesamt(haupt, rueck)
    zeile = pd.DataFrame([{
        "zeitpunkt": time.strftime("%Y-%m-%d %H:%M"),
        "was": was,
        "haupt": None if haupt is None else round(haupt, 2),
        "rueck": None if rueck is None else round(rueck, 1),
        "gesamt": None if gesamtwert is None else round(gesamtwert, 2),
        "bemerkung": bemerkung,
    }])
    zeile.to_csv(PROTOKOLL, mode="a", header=not os.path.exists(PROTOKOLL), index=False)
    print(f"  notiert: {was} -> Gesamt {gesamtwert:.2f}" if gesamtwert else f"  notiert: {was}")


def protokoll(n=25):
    if not os.path.exists(PROTOKOLL):
        print("  noch kein Protokoll")
        return
    t = pd.read_csv(PROTOKOLL)
    print(t.tail(n).to_string(index=False))
    return t
