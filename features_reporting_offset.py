"""features_reporting_offset.py - Der Meldeversatz, gemessen an den Ankuenften.

Das Hauptmodell sagt nicht die Rollzeit voraus, sondern den Rest
`y - ersatz_s = AOBT_3_flt - BLOCK_TIME_UTC_mvt`, also die Uneinigkeit zweier
Systeme ueber dieselbe Off-Block-Zeit. Das ist keine physikalische Groesse, sondern
eine Eigenschaft der Meldewege.

Dieselben zwei Systeme melden auch die Ankuenfte, und dort ist der Versatz
`ARVT_3_flt - BLOCK_TIME_UTC_mvt` (In-Block) messbar, auch in der Rangdatei: deren
Ankunftszeilen tragen BLOCK_TIME_UTC_mvt zu 100 Prozent, waehrend die Abflugzeilen
leer sind. Ueber die zehn Flughaefen laufen die Mediane beider Versaetze mit
Pearson 0,67 zusammen, LTFM liegt auf beiden Seiten am weitesten daneben.

Gebaut wird der gleitende Median der letzten Ankuenfte desselben Flughafens vor dem
Abflug, rueckwaerts gerichtet.

Aufruf: von main_model.py und submission.py.
"""
import numpy as np
import pandas as pd

FENSTER = ("60min", "360min")


def baue(roh, dep):
    """roh: alle Zeilen inklusive Ankuenfte. dep: die Abflugtabelle aus hauptmodell.baue."""
    a = roh[roh.PHASE_mvt.astype(str).str.upper().str.startswith("ARR")].copy()
    a["t"] = pd.to_datetime(a.MVT_TIME_UTC_mvt)
    a["ver"] = (pd.to_datetime(a.ARVT_3_flt) - pd.to_datetime(a.BLOCK_TIME_UTC_mvt)).dt.total_seconds()
    a["flh"] = a.ADES_mvt.astype(str)
    a = a.dropna(subset=["ver", "t"]).sort_values(["flh", "t"])

    sp = []
    for w in FENSTER:
        g = a.set_index("t").groupby("flh", observed=True).ver
        a[f"mv_med_{w}"] = g.rolling(w, closed="both").median().to_numpy()
        a[f"mv_std_{w}"] = g.rolling(w, closed="both").std().to_numpy()
        sp += [f"mv_med_{w}", f"mv_std_{w}"]

    dep = dep.copy()
    dep["_i"] = np.arange(len(dep))
    li = dep[["ADEP_mvt", "MVT_TIME_UTC_mvt", "_i"]].rename(
        columns={"ADEP_mvt": "flh", "MVT_TIME_UTC_mvt": "t"})
    li["flh"] = li.flh.astype(str)
    j = pd.merge_asof(li.sort_values("t"), a.sort_values("t")[["flh", "t"] + sp],
                      on="t", by="flh", direction="backward").sort_values("_i")
    for c in sp:
        dep[c] = j[c].to_numpy()
    return dep.drop(columns="_i")


MERKMALE = [f"mv_{p}_{w}" for w in FENSTER for p in ("med", "std")]
