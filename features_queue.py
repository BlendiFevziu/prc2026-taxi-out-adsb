"""features_queue.py - Bahnbelegung und Abflugfolge.

Bisher kennt das Modell nur `auslast`, die Zahl der Bewegungen am Flughafen im
Fenster von plus minus 30 Minuten. Das ist grob. Die Rollzeit haengt aber an der
Warteschlange **vor der einzelnen Bahn**, und die laesst sich genau bauen: in der
Rangdatei sind MVT_TIME_UTC_mvt, RUNWAY_mvt und STAND_mvt zu 100 Prozent belegt,
fuer Abfluege wie fuer Ankuenfte. Nur BLOCK_TIME und TAXITIME fehlen.

Gebaut werden je Abflug:
  - Zahl der Abfluege derselben Bahn in den 10 und 30 Minuten davor und danach,
  - Abstand in Sekunden zum vorigen und naechsten Abflug derselben Bahn,
  - Zahl der Ankuenfte derselben Bahn im Fenster davor (Bahnkreuzung, Belegung),
  - Anteil dieser Bahn an allen Abfluegen des Flughafens im Fenster von einer
    Stunde, was die Betriebsrichtung und ihren Wechsel abbildet.

Vorwaerts gerichtete Zaehler sind zulaessig, weil MVT_TIME fuer jede Zeile der
Rangdatei bekannt ist. Sie benutzen keine Zielgroesse.

Aufruf: von main_model.py und submission.py.
"""
import numpy as np
import pandas as pd

MERKMALE = ["bahn_vor_10", "bahn_nach_10", "bahn_vor_30", "bahn_nach_30",
            "bahn_luecke_vor", "bahn_luecke_nach", "bahn_ank_vor_10",
            "bahn_anteil_60", "flh_vor_10", "bahn_rang_10"]


def _zaehle(t_ziel, t_quelle, sek, vorwaerts):
    """Zahl der Elemente aus t_quelle im Fenster von sek Sekunden um t_ziel."""
    if len(t_quelle) == 0:
        return np.zeros(len(t_ziel))
    if vorwaerts:
        return (np.searchsorted(t_quelle, t_ziel + sek, "right")
                - np.searchsorted(t_quelle, t_ziel, "left"))
    return (np.searchsorted(t_quelle, t_ziel, "right")
            - np.searchsorted(t_quelle, t_ziel - sek, "left"))


def baue(roh, dep):
    r = roh.copy()
    r["t"] = pd.to_datetime(r.MVT_TIME_UTC_mvt).astype("int64") // 10**9
    ph = r.PHASE_mvt.astype(str).str.upper()
    r["bahn"] = r.RUNWAY_mvt.astype(str)
    ab = r[ph.str.startswith("DEP")]
    an = r[ph.str.startswith("ARR")]
    ab_flh = {k: np.sort(v) for k, v in ab.groupby(ab.ADEP_mvt.astype(str)).t}
    ab_bahn = {k: np.sort(v) for k, v in ab.groupby([ab.ADEP_mvt.astype(str), ab.bahn]).t}
    an_bahn = {k: np.sort(v) for k, v in an.groupby([an.ADES_mvt.astype(str), an.bahn]).t}

    dep = dep.copy()
    t = pd.to_datetime(dep.MVT_TIME_UTC_mvt).astype("int64").to_numpy() // 10**9
    flh = dep.ADEP_mvt.astype(str).to_numpy()
    bahn = dep.RUNWAY_mvt.astype(str).to_numpy()
    n = len(dep)
    aus = {m: np.zeros(n) for m in MERKMALE}
    for (f, b), idx in pd.DataFrame({"f": flh, "b": bahn}).groupby(["f", "b"]).groups.items():
        i = np.asarray(idx)
        ti = t[i]
        qb = ab_bahn.get((f, b), np.array([], dtype="int64"))
        qa = an_bahn.get((f, b), np.array([], dtype="int64"))
        qf = ab_flh.get(f, np.array([], dtype="int64"))
        aus["bahn_vor_10"][i]  = _zaehle(ti, qb,  600, False)
        aus["bahn_nach_10"][i] = _zaehle(ti, qb,  600, True)
        aus["bahn_vor_30"][i]  = _zaehle(ti, qb, 1800, False)
        aus["bahn_nach_30"][i] = _zaehle(ti, qb, 1800, True)
        aus["bahn_ank_vor_10"][i] = _zaehle(ti, qa, 600, False)
        aus["flh_vor_10"][i]   = _zaehle(ti, qf,  600, False)
        p = np.searchsorted(qb, ti, "left")
        aus["bahn_luecke_vor"][i]  = np.where(p > 0, ti - qb[np.maximum(p - 1, 0)], -1)
        q = np.searchsorted(qb, ti, "right")
        aus["bahn_luecke_nach"][i] = np.where(q < len(qb), qb[np.minimum(q, len(qb) - 1)] - ti, -1)
        g60 = _zaehle(ti, qf, 3600, False) + _zaehle(ti, qf, 3600, True)
        b60 = _zaehle(ti, qb, 3600, False) + _zaehle(ti, qb, 3600, True)
        aus["bahn_anteil_60"][i] = np.where(g60 > 0, b60 / np.maximum(g60, 1), np.nan)
        aus["bahn_rang_10"][i] = aus["bahn_vor_10"][i] / np.maximum(aus["flh_vor_10"][i], 1)
    for m in MERKMALE:
        dep[m] = aus[m]
    return dep
