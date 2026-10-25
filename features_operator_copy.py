"""features_operator_copy.py - Kopierquote je Betreiber fuer den Rueckfallweg.

Auf Rueckfallzeilen fehlt jede `_flt`-Spalte, also auch `AIRCRAFT_OPERATOR_flt`. Der
Betreiber steckt aber weiter im Bewegungssatz: `FLIGHT_mvt` beginnt mit dem Praefix der
Fluggesellschaft. Gemessen am 16.09.2026 traegt genau dieses Praefix die Information,
die dem Modell bisher fehlt.

**Der Befund.** In Rom sind 722 von 1.488 Rueckfallzeilen des Jahres 2025 Kopien, die
Blockzeit ist also die Planzeit und das Ziel exakt die Verspaetung. Eine auf den Monaten
1 bis 6 gelernte Kopierquote je Praefix trennt die Monate 7 bis 12 mit AUC 0,807, in der
Gegenrichtung 0,790. Zum Vergleich auf denselben Zeilen: Verspaetung allein 0,656 und
0,684, Stunde allein 0,570 und 0,558. Im obersten Fuenftel der vorhergesagten Quote sind
84 % der Zeilen Kopien, im untersten 5,7 %. Nur etwa jede zehnte Zeile traegt ein
Praefix, das die Lernhaelfte nie gesehen hat.

**Die Quote muss ausserhalb der eigenen Faltung gerechnet werden**, wie bei
`features_stand_rate.py`: wird sie auf denselben Zeilen gebildet, auf denen das Modell
lernt, sieht es dort seine eigene Antwort. `ausser_faltung` rechnet sie je Lernmonat aus
den uebrigen Lernmonaten, `wende_an` traegt die volle Lerntabelle auf die Haltemonate.

Das Kopierkennzeichen ist `|y - Verspaetung| <= 60 s`. Es wird vom Aufrufer als Spalte
`zk` gesetzt, damit `z` weiter das Extremkennzeichen von `features_stand_rate.py` bleibt.

Aufruf: aus den Rueckfall-Screens, spaeter aus main_model.py und submission.py.
"""
import re

import numpy as np
import pandas as pd

SCHLUESSEL = ["ADEP_mvt", "op_praefix"]
GLAETTUNG = 20
MERKMALE = ["q_opcopy", "q_opcopy_n"]
_PRAEFIX = re.compile(r"^[A-Za-z]+")


def praefix(fluege):
    """Buchstabenvorspann von FLIGHT_mvt, etwa 'AZ' aus 'AZ1234'. Leer, wenn keiner da ist."""
    werte = pd.Series(fluege, dtype="object").fillna("").astype(str).str.strip().str.upper()
    return werte.map(lambda v: (_PRAEFIX.match(v).group(0) if _PRAEFIX.match(v) else ""))


def _idx(z):
    return pd.MultiIndex.from_arrays([z[c].astype(str) for c in SCHLUESSEL])


def _tabelle(q, grund):
    g = q.groupby([q[c].astype(str) for c in SCHLUESSEL]).zk.agg(["size", "sum"])
    return (g["sum"] + GLAETTUNG * grund) / (g["size"] + GLAETTUNG), g["size"]


def passe_an(lern):
    """Liefert (Tabelle, Fallzahlen, Grundquote) fuer die spaetere Anwendung."""
    grund = float(lern.zk.mean())
    t, n = _tabelle(lern, grund)
    return {"quote": t, "anzahl": n, "grund": grund}


def wende_an(mod, z):
    i = _idx(z)
    z = z.copy()
    z["q_opcopy"] = mod["quote"].reindex(i).fillna(mod["grund"]).to_numpy()
    z["q_opcopy_n"] = mod["anzahl"].reindex(i).fillna(0).to_numpy()
    return z


def ausser_faltung(lern, gruppe="monat"):
    """Fuer die Lernzeilen selbst: je Monat aus den uebrigen Monaten gerechnet."""
    grund = float(lern.zk.mean())
    a, b = np.zeros(len(lern)), np.zeros(len(lern))
    for g in lern[gruppe].unique():
        m = (lern[gruppe] == g).to_numpy()
        t, n = _tabelle(lern[~m], grund)
        i = _idx(lern[m])
        a[m] = t.reindex(i).fillna(grund).to_numpy()
        b[m] = n.reindex(i).fillna(0).to_numpy()
    lern = lern.copy()
    lern["q_opcopy"], lern["q_opcopy_n"] = a, b
    return lern
