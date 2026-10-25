"""features_stand_rate.py - Erfindungsquote je Standplatz fuer den Rueckfallweg.

Der Rueckfallweg lernt aus 17.097 Zeilen mit gut 600 Extremfaellen und bekommt
`STAND_mvt` mit 1.631 Auspraegungen als kategoriales Merkmal. Das ist fuer ihn
Rauschen. Dieselbe Information passt in eine einzige Zahl: den Anteil der
Extremfaelle an diesem Standplatz, geglaettet gegen die Grundquote.

**Die Quote muss ausserhalb der eigenen Faltung gerechnet werden.** Wird sie auf
denselben Zeilen gebildet, auf denen das Modell lernt, sieht es dort seine eigene
Antwort: gemessen 04.09.2026 steigt der Gruppenfehler dann von 1.777,6 auf 2.169,2.
Je Lernmonat aus den uebrigen Lernmonaten gerechnet traegt sie dagegen, und am besten
**anstelle** von `STAND_mvt`, nicht zusaetzlich:

    STAND_mvt, keine Quote            1.777,6 s
    STAND_mvt und Quote               1.759,2 s
    Quote statt STAND_mvt             1.752,6 s
    Quote plus Rufzeichenquote        1.780,9 s   (die Gruppe vertraegt nur eine)

Aufruf: von main_model.py und submission.py.
"""
import numpy as np
import pandas as pd

SCHLUESSEL = ["ADEP_mvt", "STAND_mvt"]
GLAETTUNG = 20
MERKMALE = ["q_stand", "q_stand_n"]


def _idx(z):
    return pd.MultiIndex.from_arrays([z[c].astype(str) for c in SCHLUESSEL])


def _tabelle(q, grund):
    g = q.groupby([q[c].astype(str) for c in SCHLUESSEL]).z.agg(["size", "sum"])
    return (g["sum"] + GLAETTUNG * grund) / (g["size"] + GLAETTUNG), g["size"]


def passe_an(lern):
    """Liefert (Tabelle, Fallzahlen, Grundquote) fuer die spaetere Anwendung."""
    grund = float(lern.z.mean())
    t, n = _tabelle(lern, grund)
    return {"quote": t, "anzahl": n, "grund": grund}


def wende_an(mod, z):
    i = _idx(z)
    z = z.copy()
    z["q_stand"] = mod["quote"].reindex(i).fillna(mod["grund"]).to_numpy()
    z["q_stand_n"] = mod["anzahl"].reindex(i).fillna(0).to_numpy()
    return z


def ausser_faltung(lern, gruppe="monat"):
    """Fuer die Lernzeilen selbst: je Monat aus den uebrigen Monaten gerechnet."""
    grund = float(lern.z.mean())
    a = np.zeros(len(lern))
    b = np.zeros(len(lern))
    for g in lern[gruppe].unique():
        m = (lern[gruppe] == g).to_numpy()
        t, n = _tabelle(lern[~m], grund)
        i = _idx(lern[m])
        a[m] = t.reindex(i).fillna(grund).to_numpy()
        b[m] = n.reindex(i).fillna(0).to_numpy()
    lern = lern.copy()
    lern["q_stand"], lern["q_stand_n"] = a, b
    return lern
