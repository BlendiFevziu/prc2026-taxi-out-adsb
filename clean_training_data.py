"""clean_training_data.py - Ausreisser der Zielgroesse erkennen und ihren Einfluss messen.

Die Zielgroesse TAXITIME_SEC_mvt ist exakt MVT_TIME_UTC_mvt minus BLOCK_TIME_UTC_mvt.
Extremwerte und negative Werte sind daher Zeitstempelfehler, keine Rollvorgaenge.

Geprueft und VERWORFEN: Der Versuch, verschobene Tagesgrenzen an der Ranking-Datei zu
erkennen. Dort fehlt BLOCK_TIME, es bleibt nur MVT_TIME gegen SCHED_TIME. Diese Regel
trennt nicht: Bei Schwelle 1300 min erkennt sie im Training 207 Zeilen, von denen nur
5 eine Rollzeit ueber 300 min tragen. Ein spaeter Abflug ist fast immer ein echter
Verspaetungsfall, bei dem die Block-Zeit mitwandert.

Was bleibt, ist die Bereinigung der LERNMENGE. Dieses Werkzeug misst, ob sie hilft.

Aufruf:
    python3 clean_training_data.py
"""
import glob
import numpy as np
import pandas as pd

DATEN = "data/training_2025-*.parquet"
PRUEFMONATE = (1, 7)           # Januar und Juli, wie im Bewertungssatz


def lade():
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DATEN))], ignore_index=True)
    d = d[d.PHASE_mvt.astype(str).str.upper().str.startswith("DEP")].reset_index(drop=True)
    for c in ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d["y"] = d.TAXITIME_SEC_mvt.astype(float)
    d["monat"] = d.MVT_TIME_UTC_mvt.dt.month
    return d


def rmse(e):
    e = pd.Series(e).dropna()
    return float(np.sqrt((e ** 2).mean())) if len(e) else float("nan")


def main():
    d = lade()
    print(f"Abfluege {len(d):,}\n")

    print("=== Auffaellige Zielwerte ===")
    for lab, m in [("negativ", d.y < 0),
                   ("ueber 2 h", d.y > 7200),
                   ("ueber 5 h", d.y > 18000),
                   ("ueber 24 h", d.y > 86400)]:
        print(f"  {lab:12s} {m.sum():5d}  ({100*m.mean():.4f} %)")

    # Anteil an der gesamten quadrierten Abweichung gegen den Gesamtmittelwert
    ab = (d.y - d.y.mean()) ** 2
    for lab, m in [("negativ", d.y < 0), ("ueber 2 h", d.y > 7200)]:
        print(f"  Anteil von '{lab}' an der quadrierten Gesamtabweichung: "
              f"{100*ab[m].sum()/ab.sum():.1f} %")

    lern = ~d.monat.isin(PRUEFMONATE)
    pruef = d.monat.isin(PRUEFMONATE)
    print(f"\nLernmenge {lern.sum():,} | Pruefmenge {pruef.sum():,}"
          "   (Januar und Juli, wie im Bewertungssatz)\n")

    print("=== Hilft es, die Lernmenge zu kappen? Bewertet auf der UNGEKAPPTEN Pruefmenge ===")
    schl = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]
    for grenze in (None, 86400, 18000, 7200, 3600):
        w = lern & (d.y.between(0, grenze) if grenze else lern)
        tab = d[w].groupby(schl).y.mean()
        glob_ = d.loc[w, "y"].mean()
        p = d.loc[pruef, schl].merge(tab.rename("p"), left_on=schl, right_index=True,
                                     how="left").p.fillna(glob_)
        lab = "ohne Kappung" if grenze is None else f"gekappt bei {grenze//60:5d} min"
        print(f"  {lab:22s} n_lern {w.sum():9,}   RMSE {rmse(p.values - d.loc[pruef,'y'].values):7.1f} s")

    print("\n=== Dasselbe auf dem AOBT_3-Ersatzwert ===")
    d["px"] = (d.MVT_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()
    d["res"] = d.px - d.y
    hat = d.px.notna()
    for grenze in (None, 86400, 7200, 3600):
        w = lern & hat & (d.y.between(0, grenze) if grenze else True)
        korr = d[w].groupby(schl).res.mean()
        g = d.loc[w, "res"].mean()
        k = d.loc[pruef & hat, schl].merge(korr.rename("k"), left_on=schl, right_index=True,
                                           how="left").k.fillna(g)
        vor = d.loc[pruef & hat, "px"].values - k.values
        lab = "ohne Kappung" if grenze is None else f"gekappt bei {grenze//60:5d} min"
        print(f"  {lab:22s} RMSE {rmse(vor - d.loc[pruef & hat,'y'].values):7.1f} s "
              f"auf {(pruef & hat).sum():,} Fluegen")


if __name__ == "__main__":
    main()
