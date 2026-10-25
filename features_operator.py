"""features_operator.py - Betreibereigene Neigung des Rests gegen die Verspaetung.

Herkunft der Idee: Assaia misst an echten Abfluegen, dass die AOBT-Ungenauigkeit
**betreiberabhaengig** ist, weil jede Fluggesellschaft den Meilenstein anders setzt
(alle Tueren zu, Bremsen geloest, Pushback begonnen), und dass sie **bei verspaeteten
Fluegen groesser** ist als bei puenktlichen. Beides zusammen ist keine Verschiebung,
sondern eine betreibereigene NEIGUNG des Rests gegen die Verspaetung.

Unser Ziel ist genau so ein Meldeunterschied: rest = AOBT_3_flt - BLOCK_TIME_UTC_mvt,
zwei Systeme, die dasselbe Ereignis melden. Die Vermutung ist also unmittelbar pruefbar.

**Gemessen am 06.09.2026 auf 2.062.577 Zeilen der Hauptgruppe**, 95 Betreiber mit
mindestens 2.000 Fluegen:

    Neigung je Betreiber   Median 2,42 s/min, Spanne -1,60 bis 10,53, StdAbw 1,790
    Gegenprobe, Namen vertauscht                              StdAbw 0,437

Das 4,1-fache der Zufallsstreuung, der Effekt ist also da. Ein Baum kaeme nur ueber
die Wechselwirkung Betreiber x versp_min daran, und die ist bei dieser Kardinalitaet
teuer. Als fertige Zahl ist sie ein Merkmal.

**Die Neigung muss ausserhalb der eigenen Faltung gerechnet werden**, aus demselben
Grund wie die Standquote: sonst sieht das Modell auf den Lernzeilen seine eigene
Antwort. Je Lernmonat aus den uebrigen Lernmonaten.

Kleine Betreiber bekommen eine verrauschte Neigung. Deshalb wird sie gegen die
Gesamtneigung geschrumpft, Gewicht GLAETTUNG Fluege.

**GEMESSEN 06.09.2026 UND VERWORFEN.** Der Effekt ist echt, das Merkmal traegt trotzdem
nicht:

    ohne                     230,50 s
    alle vier Merkmale       231,02        (-0,52)
    nur bt_erwartet          231,19        (-0,69)
    nur Neigung und Achse    230,57        (-0,07)

`bt_erwartet` steht dabei auf Rang 11 von 73 in der Merkmalswichtigkeit, das Modell
greift also danach. Erklaerung: mit 255 Blaettern, 890 Runden und 1,7 Millionen Zeilen
findet es die Wechselwirkung Betreiber x Verspaetung selbst, und die fertige Zahl
verdraengt nur bessere Schnitte. **Eine hohe Einzelkorrelation sagt nichts darueber, ob
ein Merkmal dem Modell etwas Neues gibt.**

Aufruf: von main_model.py und submission.py, wie merkmale_standquote.
"""
import numpy as np
import pandas as pd

# Der Standplatz ist nicht dabei: die Konvention gehoert zum Betreiber und zum
# Flughafen, an dem gemeldet wird, nicht zur einzelnen Position.
SCHLUESSEL = ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"]
GLAETTUNG = 400          # Fluege, ab denen die eigene Neigung voll zaehlt
GRENZE_MIN = 240         # Verspaetungen darueber sind Ausreisser und verbiegen die Gerade
MERKMALE = ["bt_neigung", "bt_achse", "bt_erwartet", "bt_n"]


def _idx(z):
    return pd.MultiIndex.from_arrays([z[c].astype(str) for c in SCHLUESSEL])


def _tabelle(q):
    """Neigung und Achsenabschnitt je Schluessel, geschrumpft gegen das Ganze.

    Gerechnet ueber die geschlossene Form der kleinsten Quadrate, damit ein
    Gruppenaufruf je Betreiber entfaellt: b = Cov(v, r) / Var(v).
    """
    v = q.versp_min.to_numpy(float)
    r = q.rest.to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(r) & (np.abs(v) < GRENZE_MIN)
    q = q[ok]
    v, r = v[ok], r[ok]

    g = pd.DataFrame({"v": v, "r": r, "vv": v * v, "vr": v * r}).groupby(
        [q[c].astype(str).to_numpy() for c in SCHLUESSEL]).agg(
        n=("v", "size"), sv=("v", "sum"), sr=("r", "sum"),
        svv=("vv", "sum"), svr=("vr", "sum"))

    # Gesamtgerade als Ziel der Schrumpfung
    n0 = len(v)
    kov0 = (v * r).sum() / n0 - v.mean() * r.mean()
    var0 = (v * v).sum() / n0 - v.mean() ** 2
    b0 = kov0 / var0 if var0 > 0 else 0.0
    a0 = r.mean() - b0 * v.mean()

    n = g.n.to_numpy(float)
    mv = g.sv.to_numpy() / n
    mr = g.sr.to_numpy() / n
    kov = g.svr.to_numpy() / n - mv * mr
    var = g.svv.to_numpy() / n - mv * mv
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(var > 1e-9, kov / var, b0)
    b = np.where(np.isfinite(b), b, b0)

    w = n / (n + GLAETTUNG)
    b = w * b + (1 - w) * b0
    a = mr - b * mv                      # der Achsenabschnitt folgt der geschrumpften Neigung
    a = w * a + (1 - w) * a0

    t = pd.DataFrame({"neigung": b, "achse": a, "anzahl": n}, index=g.index)
    return t, (b0, a0)


def _setze(z, t, grund):
    b0, a0 = grund
    i = _idx(z)
    z = z.copy()
    z["bt_neigung"] = t.neigung.reindex(i).fillna(b0).to_numpy()
    z["bt_achse"] = t.achse.reindex(i).fillna(a0).to_numpy()
    z["bt_n"] = t.anzahl.reindex(i).fillna(0).to_numpy()
    # Die eigentliche Aussage: der aus der Konvention erwartete Meldeunterschied.
    # Ohne dieses Produkt muesste der Baum die Wechselwirkung selbst finden.
    v = z.versp_min.to_numpy(float)
    z["bt_erwartet"] = z.bt_achse.to_numpy() + z.bt_neigung.to_numpy() * np.clip(v, -GRENZE_MIN, GRENZE_MIN)
    return z


def passe_an(lern):
    t, grund = _tabelle(lern)
    return {"tab": t, "grund": grund}


def wende_an(mod, z):
    return _setze(z, mod["tab"], mod["grund"])


def ausser_faltung(lern, gruppe="monat"):
    """Fuer die Lernzeilen selbst: je Monat aus den uebrigen Monaten gerechnet."""
    teile = []
    for g in lern[gruppe].unique():
        m = (lern[gruppe] == g).to_numpy()
        t, grund = _tabelle(lern[~m])
        teile.append(_setze(lern[m], t, grund))
    return pd.concat(teile).loc[lern.index]
