"""features_callsign.py - Was sich ueber das Rufzeichen zurueckholen laesst.

**Warum es die Rueckfallgruppe ueberhaupt gibt** (gemessen 06.09.2026 auf drei
Lernmonaten, 461.887 Abfluege, davon 3.627 ohne AOBT_3):

    FLIGHT_ID_mvt gefuellt    Hauptgruppe 100,00 %   Rueckfallgruppe 0,14 %
    FLIGHT_mvt gefuellt       Hauptgruppe 100,00 %   Rueckfallgruppe 99,89 %

`FLIGHT_ID_mvt` ist der Schluessel, ueber den die Veranstalter die `_flt`-Spalten
angehaengt haben. Fehlt er, haengt nichts an: **alle sechzehn `_flt`-Spalten sind in
der Rueckfallgruppe zu 0,00 % gefuellt.** Die Gruppe ist also kein Zufall und kein
eigener Flugtyp, sondern ein fehlgeschlagener Verbund.

**Das Rufzeichen ueberlebt aber.** Und 73,9 % der Rueckfallzeilen tragen ein
Rufzeichen, das anderswo in den Daten MIT Betreiber vorkommt. Was der Verbund fuer
diese Zeile nicht geliefert hat, steht also fuer dieselbe Verbindung an anderen Tagen
da und laesst sich nachschlagen.

Das ist mehr als Kosmetik: der Rueckfallweg bekommt `AIRCRAFT_OPERATOR_flt`,
`MARKET_SEGMENT_flt`, `WK_TBL_CAT_flt` und `FLIGHT_TYPE_flt` als Merkmale, und alle
vier sind dort bisher durchgehend leer. Er lernt auf vier toten Spalten.

Zwei Arten von Merkmalen, mit unterschiedlicher Leckgefahr:

1. **Eigenschaften** (Betreiber, Marktsegment, Wirbelschleppe, Flugart). Sie haengen
   nicht am Ziel, duerfen also aus allen Zeilen geschoepft werden.
2. **Zielstatistik je Rufzeichen** (Median von y, Extremquote). Sie haengt am Ziel und
   wird **je Lernmonat aus den uebrigen Monaten** gerechnet, wie die Standquote.

**GEMESSEN 06.09.2026 UND VERWORFEN.** Je drei Saaten, Rueckfallgruppe:

    ohne alles                1747,1 s   AUC 0,965
    Eigenschaften gefuellt    1752,0     AUC 0,963
    nur Zielstatistik         1840,8     AUC 0,969
    beides                    1781,5     AUC 0,967

**Die Zielstatistik verbessert die AUC und verschlechtert den RMSE.** Das ist kein
Widerspruch: die Orakelrechnung begrenzt den Gewinn aus perfekter Erkennung auf 286 s,
bessere Erkennung ist also nicht das, was dieser Gruppe fehlt, und der Preis dafuer sind
schlechtere Regressoren.

**Der Befund zum fehlgeschlagenen Verbund oben bleibt davon unberuehrt.** Er erklaert,
warum es die Gruppe gibt, und das gilt unabhaengig davon, dass das Nachschlagen nicht
traegt.

Aufruf: von main_model.py und submission.py.
"""
import numpy as np
import pandas as pd

EIGENSCHAFTEN = ["AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
                 "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt"]
GLAETTUNG = 5
MERKMALE = ["ruf_y_med", "ruf_y_n", "ruf_q_extrem", "ruf_y_p90"]


def _schl(z):
    return z.FLIGHT_mvt.astype(str).str.strip().str.upper()


def eigenschaften(alle):
    """Nachschlagewerk Rufzeichen -> haeufigste Auspraegung, aus den Zeilen MIT Verbund.

    Kein Zielbezug, daher ohne Faltung.
    """
    q = alle[alle.AIRCRAFT_OPERATOR_flt.notna()].copy()
    q["_r"] = _schl(q)
    tab = {}
    for c in EIGENSCHAFTEN:
        g = q.dropna(subset=[c]).groupby(["_r", c], observed=True).size()
        if len(g) == 0:
            tab[c] = pd.Series(dtype=object)
            continue
        g = g.reset_index(name="n").sort_values("n", ascending=False)
        tab[c] = g.drop_duplicates("_r").set_index("_r")[c]
    return tab


def fuelle(tab, z):
    """Leere Eigenschaften aus dem Nachschlagewerk ergaenzen. Nur wo bisher nichts steht."""
    z = z.copy()
    r = _schl(z)
    for c in EIGENSCHAFTEN:
        if c not in z:
            continue
        neu = tab[c].reindex(r).to_numpy()
        alt = z[c].to_numpy(dtype=object)
        z[c] = pd.Series(np.where(pd.isna(alt), neu, alt), index=z.index).astype("category")
    return z


def _tabelle(q, schwelle):
    r = _schl(q)
    y = q.y.to_numpy(float)
    g = pd.DataFrame({"y": y, "e": (y > schwelle).astype(float)}).groupby(r.to_numpy())
    a = g.agg(med=("y", "median"), n=("y", "size"), q=("e", "mean"),
              p90=("y", lambda s: np.percentile(s, 90)))
    return a


def _setze(z, a, grund):
    med0, q0, p900 = grund
    r = _schl(z)
    z = z.copy()
    n = a.n.reindex(r).fillna(0).to_numpy()
    w = n / (n + GLAETTUNG)
    z["ruf_y_n"] = n
    z["ruf_y_med"] = w * a.med.reindex(r).fillna(med0).to_numpy() + (1 - w) * med0
    z["ruf_q_extrem"] = w * a["q"].reindex(r).fillna(q0).to_numpy() + (1 - w) * q0
    z["ruf_y_p90"] = w * a.p90.reindex(r).fillna(p900).to_numpy() + (1 - w) * p900
    return z


def passe_an(lern, schwelle):
    a = _tabelle(lern, schwelle)
    grund = (float(lern.y.median()), float((lern.y > schwelle).mean()),
             float(np.percentile(lern.y, 90)))
    return {"tab": a, "grund": grund}


def wende_an(mod, z):
    return _setze(z, mod["tab"], mod["grund"])


def ausser_faltung(lern, schwelle, gruppe="monat"):
    teile = []
    for g in lern[gruppe].unique():
        m = (lern[gruppe] == g).to_numpy()
        a = _tabelle(lern[~m], schwelle)
        grund = (float(lern[~m].y.median()), float((lern[~m].y > schwelle).mean()),
                 float(np.percentile(lern[~m].y, 90)))
        teile.append(_setze(lern[m], a, grund))
    return pd.concat(teile).loc[lern.index]
