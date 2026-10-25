"""fallback_cv.py - Messgeraet fuer die Rueckfallgruppe, zweite Fassung.

## ACHTUNG, 23.09.2026: DIESES WERKZEUG MISST SEIT DEM 13.09.2026 GEGEN DIE FALSCHE GRUNDLINIE

`verfahren()` bildet den Rueckfallweg nach, wie er VOR v46 aussah. v46 hat die Gruppe am
13.09.2026 auf das verankerte Modell aus `research_240/anchor_preview_2026_production.py`
umgestellt. Gemessen: `models.pkl` reproduziert die abgegebenen Rueckfallzeilen NICHT, im
Median 74 s daneben, bei 97,8 Prozent der Zeilen, hoechstens 4.536 s.

Was sich geaendert hat: Klassifikator und Hoehenmodell bleiben auf den 35 Merkmalen aus
`H.MERKMALE_RUECK`, nur der NORMALZWEIG wird durch `nt_anchor + offset` ersetzt, der Offset
auf 41 Merkmalen gelernt. Das Gewicht `q = p_z ** HOCH` ist unveraendert.

**Vorlage fuer neue Messungen: `research_280/kopie_anchored_sixpair.py`.** Wer dieses Werkzeug
hier trotzdem benutzt, misst eine Konfiguration, die nicht abgegeben wird.

## UND: DIE EICHPROBE IST DURCHGEFALLEN, ZWEIMAL

Am 08.09.2026 (experiments.csv Zeile 71) und erneut am 23.09.2026: das Geraet haelt HOCH 0,8
fuer besser als 1,6, der Bewertungssatz sagt das Gegenteil. Am 23.09.2026 kam dazu, dass die
KORRIGIERTE, verankerte Fassung fuer v53 +1,219 s vorhersagte und die Rangliste -0,021 s
lieferte, also eine Ueberschaetzung um rund das Sechzigfache. **Auf diesem Weg laesst sich
keine Aenderung der Rueckfallgruppe der Groesse nach beziffern.** Zaehlbare Groessen wie die
Treffergenauigkeit eines Gates sind hier die einzige belastbare Evidenz.


## Warum die erste Fassung falsch war

Die erste Fassung (08.09.2026 vormittags) faltete ZUFAELLIG ueber alle zwoelf Monate,
geschichtet nach Monat und Extremfall. Sie sagte fuer HOCH 0,8 gegen 1,6 einen Gewinn von
+1,077 s voraus, Bootstrap-Bereich [+0,262, +1,915], die Null also klar ausserhalb.
Der Bewertungssatz lieferte 277,0198 gegen 276,5108, also 0,509 s SCHLECHTER.
Das Vorzeichen war falsch, nicht die Groesse.

Die Ursache steckt in der Faltungsform. Bei einer zufaelligen Faltung enthaelt jede
Lernmenge Zeilen aus DEMSELBEN Monat wie die bewerteten Zeilen, oft aus demselben Tag.
Der Klassifikator sieht also die Meldelage des Bewertungszeitraums bereits beim Lernen.
Im Ernstfall lernt er auf 2025 und trifft auf Januar und Juli 2026, wo er diese Lage nicht
kennt. HOCH steuert genau die Schaerfe dieser Wahrscheinlichkeiten, also misst eine
zufaellige Faltung fuer diese Groesse etwas anderes als den Ernstfall.

## Was diese Fassung anders macht

**Die Faltung hat die Form des Ernstfalls.** Gelernt wird auf zehn Monaten, bewertet wird
ein zurueckgehaltenes Monatspaar, sechs Paare insgesamt: (1,7), (2,8) bis (6,12). Das
Paar (1,7) ist die Form der Abgabe selbst, die uebrigen fuenf liefern die Streuung.

**Verglichen wird gepaart.** Die alte Kreuzvalidierung ueber Monatspaare (04.09.2026)
verglich absolute RMSE-Werte zwischen Faltungen und kam auf einen Standardfehler von
rund 92 s, weshalb sie nichts unter 180 s aufloeste. Zwei Verfahren auf DENSELBEN Zeilen
und DENSELBEN Faltungen zu vergleichen, hebt den groessten Teil dieser Streuung heraus.

**Bewertet wird der Beitrag zur Rangliste**, nicht der gruppeneigene RMSE.

**Die grosse Merkmalstabelle wird nie geladen.** Die 22.470 Rueckfallzeilen liegen einmal
als eigene Datei, danach braucht ein Versuch rund 200 MB statt 6 GB. Die erste Fassung
hielt die volle Tabelle waehrend der ganzen Schleife und liess den Rechner auslagern, eine
Variante brauchte dadurch 80 Minuten statt 3.

## Eichprobe

Der Bewertungssatz kennt die Antwort fuer eine Frage bereits: HOCH 1,6 ist um 0,509 s
besser als 0,8. Ein Messgeraet, das diese Reihenfolge nicht reproduziert, wird nicht
benutzt. `python3 fallback_cv.py` fuehrt genau diese Probe aus.

Aufruf:
    python3 fallback_cv.py            Eichprobe: HOCH 1,6 gegen 0,8
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import neural_model as NM        # zuerst, OpenMP, siehe neural_model.py
import lightgbm as lgb
import main_model as H
import features_stand_rate as MSQ

ZEILEN = ".cache/fallback_rows.parquet"
PAARE = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]

N_TEST = 344_841
N_TEST_RUECK = 5_290
RMSE_BEST = 276.5108
# Die Rueckfallgruppe des Bewertungssatzes ist extremdichter als die der Lerndaten,
# gemessen 06.09.2026: 3,63 Prozent gegen 2,87.
DICHTEFAKTOR = 1.27
BOOTSTRAP = 2000


def ranglistenwirkung(mse_alt, mse_neu):
    """Aenderung des mittleren Fehlerquadrats je Zeile -> Sekunden auf der Rangliste.

    RMSE = sqrt(SSE / N). Aendert sich nur die Rueckfallgruppe, ist
    dRMSE = n_rueck * dMSE / (2 * RMSE * N).
    """
    return N_TEST_RUECK * (mse_alt - mse_neu) * DICHTEFAKTOR / (2 * RMSE_BEST * N_TEST)


def vorbereiten(neu=False):
    """Schneidet die Rueckfallzeilen einmal heraus und legt sie als eigene Datei ab.

    Danach laedt kein Versuch mehr die volle Merkmalstabelle.
    """
    if os.path.exists(ZEILEN) and not neu:
        lo = pd.read_parquet(ZEILEN)
        print(f"  {len(lo):,} Rueckfallzeilen aus {ZEILEN}")
        return lo
    import framework as G
    print("  einmalig: volle Merkmalstabelle bauen ...", flush=True)
    d = G.daten()
    lo = d[d.hat_aobt == 0].copy()
    lo["monat"] = pd.to_datetime(lo.MVT_TIME_UTC_mvt).dt.month
    del d
    os.makedirs(".cache", exist_ok=True)
    lo.to_parquet(ZEILEN, index=False)
    print(f"  {len(lo):,} Zeilen abgelegt in {ZEILEN} "
          f"({os.path.getsize(ZEILEN)/1e6:.0f} MB)")
    return lo


def verfahren(hoch=H.HOCH, anteil_hm=H.ANTEIL_HM):
    """Der Rueckfallweg aus main_model.endmodell, mit einstellbarer Schaerfe."""
    def f(lern, pruef):
        lern = lern.copy()
        lern["z"] = (lern.y > H.SCHWELLE_EXTREM).astype(int)
        quo = MSQ.passe_an(lern)
        lern = MSQ.ausser_faltung(lern)
        pruef = MSQ.wende_an(quo, pruef.copy())
        m2c = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.05,
                                 num_leaves=31, min_child_samples=20, verbose=-1, n_jobs=-1)
        m2c.fit(lern[H.MERKMALE_RUECK], lern.z)
        m2 = lgb.LGBMRegressor(objective="l2", n_estimators=400, learning_rate=0.05,
                               num_leaves=31, min_child_samples=20, verbose=-1, n_jobs=-1)
        m2.fit(lern[lern.z == 0][H.MERKMALE_RUECK], lern[lern.z == 0].y)
        m2h = lgb.LGBMRegressor(objective="l2", n_estimators=300, learning_rate=0.05,
                                num_leaves=15, min_child_samples=10, verbose=-1, n_jobs=-1)
        m2h.fit(lern[lern.z == 1][H.MERKMALE_RUECK], lern[lern.z == 1].y)
        x = pruef[H.MERKMALE_RUECK]
        pw = m2c.predict_proba(x)[:, 1] ** hoch
        hoehe = ((1 - anteil_hm) * pruef.versp_min.to_numpy() * 60
                 + anteil_hm * m2h.predict(x))
        return np.clip((1 - pw) * m2.predict(x) + pw * hoehe, 0, None)
    return f


def messe(f, lo, name="", laut=True):
    """Sechs Faltungen in der Form des Ernstfalls: zehn Monate lernen, ein Paar bewerten.

    Gibt die Fehlerquadrate je Zeile zurueck, in der Reihenfolge von `lo`.
    """
    q = np.full(len(lo), np.nan)
    for paar in PAARE:
        te = lo.monat.isin(paar).to_numpy()
        v = f(lo[~te], lo[te])
        q[te] = (v - lo.y.to_numpy()[te]) ** 2
        if laut:
            print(f"    {name} Paar {paar}: {int(te.sum()):,} Zeilen, "
                  f"MSE {q[te].mean():,.0f}", flush=True)
    assert not np.isnan(q).any(), "nicht jede Zeile wurde bewertet"
    return q


def vergleiche(q_a, q_b, name_a, name_b):
    """Gepaarter Vergleich zweier Verfahren auf denselben Zeilen."""
    d = q_a - q_b                       # positiv heisst: b ist besser
    rs = np.random.RandomState(0)
    idx = rs.randint(0, len(d), size=(BOOTSTRAP, len(d)))
    dbs = d[idx].mean(axis=1)
    s = ranglistenwirkung(q_a.mean(), q_b.mean())
    lo_ = ranglistenwirkung(0, -np.percentile(dbs, 2.5))
    hi_ = ranglistenwirkung(0, -np.percentile(dbs, 97.5))
    print(f"\n  {name_a:28s} MSE {q_a.mean():12,.0f}  RMSE {np.sqrt(q_a.mean()):9,.1f}")
    print(f"  {name_b:28s} MSE {q_b.mean():12,.0f}  RMSE {np.sqrt(q_b.mean()):9,.1f}")
    print(f"  {name_b} gegen {name_a}: {s:+.3f} s   "
          f"Bootstrap 95 % [{lo_:+.3f}, {hi_:+.3f}]")
    return s, lo_, hi_


def main():
    lo = vorbereiten()
    print(f"  Extremfaelle: {int((lo.y > H.SCHWELLE_EXTREM).sum()):,} "
          f"({(lo.y > H.SCHWELLE_EXTREM).mean():.2%})")
    print(f"  1 s auf der Rangliste = {1/ranglistenwirkung(1, 0):,.0f} MSE je Zeile\n")
    print("  EICHPROBE. Der Bewertungssatz sagt: HOCH 1,6 ist um 0,509 s besser als 0,8.")
    print("  Ein Messgeraet, das das nicht reproduziert, wird nicht benutzt.\n", flush=True)

    q16 = messe(verfahren(hoch=1.6), lo, "HOCH 1,6")
    q08 = messe(verfahren(hoch=0.8), lo, "HOCH 0,8")
    s, lo_, hi_ = vergleiche(q16, q08, "HOCH 1,6 (abgegeben)", "HOCH 0,8 (verworfen)")

    np.save(".cache/rueck_einsatzform_16.npy", q16)
    np.save(".cache/rueck_einsatzform_08.npy", q08)
    print()
    if hi_ < 0:
        print(f"  BESTANDEN: 0,8 ist schlechter, wie der Bewertungssatz sagt.")
        print(f"  Der Bewertungssatz nennt -0,509 s, das Messgeraet {s:+.3f} s.")
    elif lo_ > 0:
        print(f"  DURCHGEFALLEN: das Messgeraet haelt 0,8 fuer besser, der "
              f"Bewertungssatz sagt das Gegenteil.")
    else:
        print(f"  UNENTSCHIEDEN: der Bereich enthaelt die Null. Das Messgeraet kann "
              f"einen Unterschied dieser Groesse nicht aufloesen.")
        print(f"  Das ist ehrlicher als die erste Fassung, die sich sicher und falsch war.")


if __name__ == "__main__":
    main()
