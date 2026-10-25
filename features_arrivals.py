"""features_arrivals.py - Auslastung aus den BESCHRIFTETEN Ankuenften.

Der Wettbewerbsdatensatz enthaelt nicht nur Abfluege. In ranking.parquet stehen
**344.693 Ankuenfte** (die 215.967 galten vor dem Juli-Neuexport vom 04.09.2026,
nachgezaehlt 06.09.2026), und deren TAXITIME_SEC_mvt ist zu 100 Prozent belegt, obwohl
die der Abfluege leer ist. Ebenso BLOCK_TIME_UTC_mvt: auf der Ankunftsseite zu 100 %
da, auf der Abflugseite zu 0 %. Es liegt also beschriftete Wirklichkeit aus Januar und Juli 2026
vor, aus genau den Monaten, auf die bewertet wird.

Daraus wird ein direktes Auslastungsmass gebaut: die mittlere Rollzeit der Ankuenfte
desselben Flughafens im Zeitfenster vor dem Abflug. Das ist gemessene Behinderung auf
den Rollwegen, nicht ein Stellvertreter wie die Verspaetung.

Rueckwaerts gerichtet, es gehen nur Ankuenfte ein, die vor dem Abflug gelandet sind.

Aufruf: wird von main_model.py und submission.py benutzt.
"""
import numpy as np
import pandas as pd

FENSTER = ("30min", "90min")


def baue(roh, dep):
    """roh: alle Zeilen inklusive Ankuenfte. dep: die Abflugtabelle aus hauptmodell.baue."""
    roh = roh.copy()
    roh["MVT_TIME_UTC_mvt"] = pd.to_datetime(roh.MVT_TIME_UTC_mvt)
    ank = roh[roh.PHASE_mvt.astype(str).str.upper().str.startswith("ARR")].copy()
    ank["taxi_in"] = pd.to_numeric(ank.TAXITIME_SEC_mvt, errors="coerce")
    ank = ank.dropna(subset=["taxi_in", "ADES_mvt"])
    ank = ank[["ADES_mvt", "MVT_TIME_UTC_mvt", "taxi_in"]].rename(
        columns={"ADES_mvt": "flh", "MVT_TIME_UTC_mvt": "t"})
    ank["flh"] = ank.flh.astype(str)          # sonst kollidiert category mit object
    ank = ank.sort_values(["flh", "t"])

    for w in FENSTER:
        g = ank.set_index("t").groupby("flh", observed=True).taxi_in
        ank[f"med_{w}"] = g.rolling(w, closed="both").median().to_numpy()
        ank[f"anz_{w}"] = g.rolling(w, closed="both").count().to_numpy()

    dep = dep.copy()
    dep["_i"] = np.arange(len(dep))
    links = dep[["ADEP_mvt", "MVT_TIME_UTC_mvt", "_i"]].rename(
        columns={"ADEP_mvt": "flh", "MVT_TIME_UTC_mvt": "t"})
    links["flh"] = links.flh.astype(str)
    links = links.sort_values("t")
    sp = [f"{p}_{w}" for w in FENSTER for p in ("med", "anz")]
    j = pd.merge_asof(links, ank.sort_values("t")[["flh", "t"] + sp],
                      on="t", by="flh", direction="backward")
    j = j.sort_values("_i")
    for c in sp:
        dep["ank_" + c] = j[c].to_numpy()
    return dep.drop(columns="_i")


MERKMALE = [f"ank_{p}_{w}" for w in FENSTER for p in ("med", "anz")]
