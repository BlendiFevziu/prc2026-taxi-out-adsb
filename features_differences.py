"""features_differences.py - Alle paarweisen Differenzen der Zeitfelder.

Die Zeilen tragen acht Zeitstempel, also 28 moegliche Differenzen. Zehn davon waren im
Modell (die Ersatzwerte, versp_min, eobt_sched_min und so fort), die uebrigen 17 nicht.
Systematisch nachgeholt am 06.09.2026: Hauptgruppe **230,71 statt 231,49 s**.

**Als Familie einbauen, nicht einzeln aussuchen.** Die Einzelsiebung auf einer Stichprobe
rangiert falsch: `dt_aobt_3_eobt_1` fiel dort mit 0,00 s durch, steht im vollen Modell aber
auf Platz 11 der Wichtigkeit. Die acht bestgesiebten zusammen bringen 0,36 s, alle 17
bringen 0,78.

Im Rueckfallweg wirkungslos (1.747,1 mit und ohne), weil dort saemtliche `_flt`-Felder
leer sind und die Differenzen damit undefiniert.

Aufruf: von hauptmodell.baue().
"""
import pandas as pd

ZEITEN = ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt",
          "LOBT_flt", "IOBT_flt", "ARVT_1_flt", "ARVT_3_flt"]
# Differenzen, die hauptmodell.baue() bereits unter eigenem Namen fuehrt
SCHON = {("MVT_TIME_UTC_mvt", "AOBT_3_flt"), ("MVT_TIME_UTC_mvt", "EOBT_1_flt"),
         ("MVT_TIME_UTC_mvt", "LOBT_flt"), ("MVT_TIME_UTC_mvt", "IOBT_flt"),
         ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"), ("EOBT_1_flt", "SCHED_TIME_UTC_mvt"),
         ("LOBT_flt", "SCHED_TIME_UTC_mvt"), ("IOBT_flt", "EOBT_1_flt"),
         ("ARVT_3_flt", "MVT_TIME_UTC_mvt"), ("ARVT_1_flt", "EOBT_1_flt"),
         ("AOBT_3_flt", "SCHED_TIME_UTC_mvt")}


def _kurz(c):
    return c.replace("_TIME_UTC_mvt", "").replace("_flt", "").lower()


MERKMALE = [f"dt_{_kurz(a)}_{_kurz(b)}"
            for i, a in enumerate(ZEITEN) for b in ZEITEN[i + 1:]
            if (a, b) not in SCHON and (b, a) not in SCHON]


def baue(d):
    t = {c: pd.to_datetime(d[c], errors="coerce") for c in ZEITEN}
    for i, a in enumerate(ZEITEN):
        for b in ZEITEN[i + 1:]:
            if (a, b) in SCHON or (b, a) in SCHON:
                continue
            d[f"dt_{_kurz(a)}_{_kurz(b)}"] = (t[a] - t[b]).dt.total_seconds()
    return d
