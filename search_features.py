"""search_features.py - Systematische Merkmalssuche mit dem Geruest.

Statt einzelner Einfaelle werden ganze Familien erzeugt und gemeinsam gesiebt:

  A  alle paarweisen Differenzen der acht Zeitfelder (28 Paare, zehn davon schon benutzt)
  B  Sekundenanteil und Rundungskennzeichen je Zeitfeld. Ein Feld, das immer auf 00
     Sekunden endet, ist gerundet und damit geschaetzt statt gemessen. Genau diese Bauart
     war der Rom-Fund (BLOCK gleich SCHED, dort mit Sekundenzittern).
  C  Kalendermerkmale, die noch fehlen (Tag im Monat, Woche, Abstand zur Tagesmitte)

Gesiebt wird schnell auf einer Stichprobe, bestaetigt wird nur, was durchkommt.
"""
import numpy as np
import pandas as pd

ZEITEN = ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt",
          "LOBT_flt", "IOBT_flt", "ARVT_1_flt", "ARVT_3_flt"]
# Differenzen, die das Modell schon kennt, in Sekunden ausgedrueckt
SCHON = {("MVT_TIME_UTC_mvt", "AOBT_3_flt"), ("MVT_TIME_UTC_mvt", "EOBT_1_flt"),
         ("MVT_TIME_UTC_mvt", "LOBT_flt"), ("MVT_TIME_UTC_mvt", "IOBT_flt"),
         ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"), ("EOBT_1_flt", "SCHED_TIME_UTC_mvt"),
         ("LOBT_flt", "SCHED_TIME_UTC_mvt"), ("IOBT_flt", "EOBT_1_flt"),
         ("ARVT_3_flt", "MVT_TIME_UTC_mvt"), ("ARVT_1_flt", "EOBT_1_flt"),
         ("AOBT_3_flt", "SCHED_TIME_UTC_mvt")}


def _kurz(c):
    return c.replace("_TIME_UTC_mvt", "").replace("_flt", "").lower()


def baue(d):
    """Fuegt die drei Familien an und liefert (Tabelle, {Familie: [Spalten]})."""
    d = d.copy()
    t = {c: pd.to_datetime(d[c], errors="coerce") for c in ZEITEN}
    fam = {"A_differenzen": [], "B_sekunden": [], "C_kalender": []}

    for i, a in enumerate(ZEITEN):
        for b in ZEITEN[i + 1:]:
            if (a, b) in SCHON or (b, a) in SCHON:
                continue
            nm = f"dt_{_kurz(a)}_{_kurz(b)}"
            d[nm] = (t[a] - t[b]).dt.total_seconds()
            fam["A_differenzen"].append(nm)

    for c in ZEITEN:
        s = t[c].dt.second
        nm = f"sek_{_kurz(c)}"
        d[nm] = s
        d[nm + "_null"] = (s == 0).astype(float)
        fam["B_sekunden"] += [nm, nm + "_null"]

    m = t["MVT_TIME_UTC_mvt"]
    d["tag_im_monat"] = m.dt.day
    d["woche"] = m.dt.isocalendar().week.astype(float)
    d["abstand_mittag"] = (m.dt.hour * 60 + m.dt.minute - 720).abs()
    d["ist_wochenende"] = (m.dt.dayofweek >= 5).astype(float)
    fam["C_kalender"] = ["tag_im_monat", "woche", "abstand_mittag", "ist_wochenende"]
    return d, fam
