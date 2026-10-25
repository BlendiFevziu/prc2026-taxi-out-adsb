"""fallback_path.py - Vorhersage der Taxi-out-Zeit fuer Zeilen OHNE das Feld AOBT_3.

AUSGANGSLAGE. Der Ersatzwert MVT_TIME minus AOBT_3 erreicht auf den Zeilen, die
AOBT_3 tragen, einen RMSE von 428 s gegen 625 s der besten einfachen Messlatte. Er
fehlt aber auf 1,482 Prozent der Abfluege der Ranking-Datei.

DER BEFUND, DER DIE ARBEIT LENKT. Fehlendes AOBT_3 und gestoerte Zeitstempel haengen
eng zusammen. In den Lerndaten liegen ueber zwei Stunden:
    Zeilen mit AOBT_3    0,005 Prozent
    Zeilen ohne AOBT_3   2,141 Prozent
Von 584 Zeilen ueber zwei Stunden haben 481 kein AOBT_3, erwartet waeren bei Zufall
sechs. Diese 1,5 Prozent tragen deshalb ueber die Haelfte der gesamten quadrierten
Abweichung, und der Gesamtwert haengt fast nur an ihnen.

WAS TRENNT. Ueber alle Zeilen taugt MVT_TIME minus SCHED_TIME nicht zur Erkennung
gestoerter Werte, die Trefferquote liegt bei 2,4 Prozent. Innerhalb der Zeilen ohne
AOBT_3 trennt dieselbe Groesse dagegen deutlich, und sie steht in der Ranking-Datei
zur Verfuegung:
    bis  60 min Verspaetung   0,9 Prozent gestoert, Mittel  15 min
    180 bis 600 min          10,6 Prozent gestoert, Mittel  49 min
    ueber   600 min          53,8 Prozent gestoert, Mittel 563 min

VERFAHREN. Geschrumpfte Mittelwerte je Verspaetungsklasse und Flughafen, gelernt
ausschliesslich auf Zeilen ohne AOBT_3. Bewusst ohne Fremdpakete, lightgbm ist auf
diesem Rechner nicht installiert und waere fuer diese Teilmenge auch ueberzogen.

Bewertet auf Januar und Juli 2025, weil der Bewertungssatz diese Monate umfasst.

Aufruf:
    python3 fallback_path.py
"""
import glob

import numpy as np
import pandas as pd

PRUEFMONATE = (1, 7)
KANTEN = [-np.inf, 30, 60, 120, 180, 300, 600, 1200, np.inf]   # Minuten Verspaetung
SCHRUMPF = 20


def lade():
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/training_2025-*.parquet"))],
                  ignore_index=True)
    d = d[d.PHASE_mvt.astype(str).str.upper().str.startswith("DEP")].reset_index(drop=True)
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d["y"] = d.TAXITIME_SEC_mvt.astype(float)
    d["hat_aobt"] = d.AOBT_3_flt.notna()
    d["monat"] = d.MVT_TIME_UTC_mvt.dt.month
    d["versp_min"] = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds() / 60
    d["vkl"] = pd.cut(d.versp_min, KANTEN, labels=False)
    return d


def rmse(p, y):
    return float(np.sqrt(((np.asarray(p, float) - np.asarray(y, float)) ** 2).mean()))


def lerne(lern):
    """Zwei Nachschlagewerke, grob nach fein."""
    grob = lern.groupby("vkl").y.agg(["mean", "count"])
    fein = lern.groupby(["vkl", "ADEP_mvt"]).y.agg(["mean", "count"])
    return lern.y.mean(), grob, fein


def sage_vorher(z, gesamt, grob, fein):
    v = z.vkl.map(grob["mean"]).fillna(gesamt).to_numpy()
    idx = list(zip(z.vkl, z.ADEP_mvt))
    m = np.array([fein["mean"].get(k, np.nan) for k in idx], float)
    c = np.array([fein["count"].get(k, 0) for k in idx], float)
    w = c / (c + SCHRUMPF)
    return np.where(np.isnan(m), v, w * np.nan_to_num(m) + (1 - w) * v)


def main():
    d = lade()
    lern = d[~d.monat.isin(PRUEFMONATE)]
    pruef = d[d.monat.isin(PRUEFMONATE)]
    lo, po = lern[~lern.hat_aobt], pruef[~pruef.hat_aobt]
    ph = pruef[pruef.hat_aobt]

    print(f"Abfluege {len(d):,}, ohne AOBT_3 {(~d.hat_aobt).sum():,} "
          f"({100*(~d.hat_aobt).mean():.3f} %)")
    print(f"Lernen auf {len(lo):,} Zeilen ohne AOBT_3, pruefen auf {len(po):,}\n")

    gesamt, grob, fein = lerne(lo)
    stufen = {
        "Mittel aller Zeilen": np.full(len(po), lern.y.mean()),
        "Mittel nur der Zeilen ohne AOBT_3": np.full(len(po), gesamt),
        "je Verspaetungsklasse": po.vkl.map(grob["mean"]).fillna(gesamt).to_numpy(),
        "je Verspaetungsklasse und Flughafen": sage_vorher(po, gesamt, grob, fein),
    }
    print("=== RMSE auf den Pruefzeilen ohne AOBT_3 ===")
    for k, v in stufen.items():
        print(f"  {k:38s} {rmse(v, po.y):8.1f} s")

    ersatz = (ph.MVT_TIME_UTC_mvt - ph.AOBT_3_flt).dt.total_seconds()
    r_hat = rmse(ersatz, ph.y)
    print(f"\n=== Zusammengesetzt ueber alle {len(pruef):,} Pruefzeilen ===")
    print(f"  Zeilen mit AOBT_3, Ersatzwert:     n={len(ph):7,}  RMSE {r_hat:7.1f} s")
    for k, v in stufen.items():
        r_ohne = rmse(v, po.y)
        ges = np.sqrt((len(ph)*r_hat**2 + len(po)*r_ohne**2) / len(pruef))
        anteil = 100*len(po)*r_ohne**2 / (len(ph)*r_hat**2 + len(po)*r_ohne**2)
        print(f"  mit Rueckfallweg '{k[:34]:34s}' -> {ges:7.1f} s "
              f"(Rueckfall traegt {anteil:4.1f} % des Fehlers)")

    print("\n  Zum Vergleich, beste einfache Messlatte ueber alles: 624,9 s")
    print("\nHINWEIS: Der Ersatzwert allein sieht mit 428 s stark aus, gilt aber nur fuer")
    print("98,5 Prozent der Zeilen. Erst zusammengerechnet zeigt sich, dass der Rueckfallweg")
    print("den Gesamtwert bestimmt.")


if __name__ == "__main__":
    main()
