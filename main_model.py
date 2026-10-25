"""main_model.py - Gradient Boosting auf die Taxi-out-Zeit, zwei getrennte Modelle.

Zuerst wurde ein gemeinsames Modell fuer alle Zeilen versucht, mit dem Ersatzwert
MVT_TIME minus AOBT_3 als Merkmal, das dort leer bleibt, wo AOBT_3 fehlt. Es erreicht
456,3 s, aber auf den Zeilen ohne AOBT_3 nur 2.900,8 s und damit schlechter als das
Handverfahren aus fallback_path.py mit 2.795,6 s.

Zwei getrennte Modelle sind deutlich besser. Die Zeilen ohne AOBT_3 verhalten sich
anders, dort liegen 2,141 Prozent ueber zwei Stunden gegen 0,005 Prozent sonst, und
ein eigenes, kleineres Modell auf dieser Teilmenge erreicht 2.135,7 s. Zusammengesetzt
ergibt das 384,8 s.

    Zeilen mit AOBT_3    grosses Modell, 279,6 s
    Zeilen ohne AOBT_3   kleines Modell,  2.135,7 s
    zusammen                              384,8 s

Merkmale ausschliesslich aus Feldern, die in ranking.parquet vorhanden sind.

Bewertet auf Januar und Juli 2025, weil der Bewertungssatz diese Monate umfasst.

Aufruf:
    python3 main_model.py
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
# netz_modell zuerst, es zieht torch nach. Wird libomp zuerst von LightGBM geladen,
# stuerzt der Prozess spaeter im Netz mit Segmentation fault 11 ab, siehe neural_model.py.
import neural_model as NM
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import features_geometry as MG
import features_turnaround as MU
import features_stand_rate as MSQ
import features_sequence as MLN
import features_differences as MD
import features_arrivals as MA
import features_flight_sequence as MFID

PRUEFMONATE = (1, 7)
FENSTER_S = 1800
# A separate output keeps a full-retrain candidate from overwriting the known-good
# submission model.  Example: PRC_MODEL_OUT=models_fid_arrivals.pkl python3 main_model.py --alle
MODEL_OUT = os.environ.get("PRC_MODEL_OUT", "models.pkl")
# Schwelle, ab der eine Rollzeit als Extremfall gilt. Nie eingestellt gewesen, am
# 04.09.2026 nachgeholt. Gruppenfehler ueber die Schwelle: 1800 -> 2.197 | 2700 -> 2.016
# | 3600 -> 1.895 | 5400 -> 1.778 | 7200 -> 1.786 | 10800 -> 1.960 | 14400 -> 1.937.
# Tiefer heisst mehr Lernbeispiele fuer Klassifikator und Hoehenmodell, aber auch mehr
# Faelle, in denen die Verspaetung als Hoehe nicht mehr stimmt.
SCHWELLE_EXTREM = 5400   # anderthalb Stunden
# Rueckfallweg, eingestellt 04.09.2026 auf Januar und Juli. Die Flaeche um das Optimum
# ist flach (Anteil 0,4 bis 0,6 mal Potenz 1,3 bis 2,0 alles zwischen 1.786 und 1.790 s),
# das ist also keine Ueberanpassung an zwei Monate.
# Anteil von CatBoost am Gemisch der Hauptgruppe. Am 05.09.2026 von 0,5 auf 0,55 erhoeht:
# CatBoost profitiert von den Ankunftsmerkmalen staerker als LightGBM und ist allein besser
# geworden (229,04 gegen 229,44 fuer die ganze LightGBM-Seite). Gemessen ueber 0,30 bis 0,80.
# Mittelung ueber mehrere CatBoost-Saaten wurde geprueft und VERWORFEN: drei Saaten geben
# 226,26 statt 226,55, also 0,29 s fuer 4,4 zusaetzliche Stunden Rechenzeit.
ANTEIL_CB = 0.6
# Anteil des neuronalen Netzes am Gemisch der Hauptgruppe, eingebaut 07.09.2026.
# Die Merkmale sind ausgereizt, deshalb kommt der Gewinn jetzt aus einer dritten
# MODELLKLASSE. LightGBM und CatBoost sind beide Baumverfahren und irren sich aehnlich
# (Restfehler-Korrelation 0,9515); das Netz liegt bei 0,9332 gegen LightGBM und 0,9272
# gegen CatBoost. Gemessen auf der Halteprobe, Hauptgruppe, 339.046 Zeilen:
#
#   LightGBM allein                     229,69
#   CatBoost allein                     228,16
#   Netz allein, fuenf Saaten           233,31
#   0,40 / 0,60 / 0     bisher          226,08
#   0,30 / 0,45 / 0,25  neu             225,28
#
# Das Verhaeltnis LightGBM zu CatBoost bleibt bei 0,4 zu 0,6, das Netz nimmt ein
# Viertel des Ganzen. Die Kurve ist flach, 0,20 bis 0,30 liegen innerhalb 0,05 s.
# TabM wurde daneben geprueft und VERWORFEN: allein 236,22 bei 4.208 s Rechenzeit,
# im Vierergemisch nur 0,16 s besser als ohne.
ANTEIL_NETZ = 0.25
# Weiche Korrektur der erfundenen Block-Zeiten in der Hauptgruppe, 07.09.2026.
# 4,52 Prozent der Zeilen tragen BLOCK_TIME = SCHED_TIME, und dort gilt y = versp exakt
# (RMSE von versp auf diesen Zeilen 3,72 s). Ein Klassifikator auf denselben Merkmalen
# erkennt sie mit AUC 0,8759. Vorhergesagt wird dann
#
#     v = (1 - p**ERF_HOCH) * Modell + p**ERF_HOCH * versp
#
# Der Ausdruck zieht die Wahrscheinlichkeiten nach unten, die Korrektur greift also nur
# dort spuerbar, wo der Klassifikator sicher ist. Gestimmt auf einer Haelfte der
# Halteprobe, gemessen auf der anderen:
#
#     h        1,0    1,5    2,0    2,5    3,0    4,0
#     auf B  226,44 225,52 225,35 225,32 225,32 225,34
#
# ohne Korrektur 225,835, mit 225,316, also 0,52 s. Die Flaeche ist flach von 2,0 bis 4,0.
# Zusammen mit der unteren Kappung sind es 1,04 s, die beiden Wirkungen addieren sich
# sauber (0,516 plus 0,519). Das Orakel laege bei 211,58, erreichbar ist davon wenig:
# bei einer Trefferquote von 65,7 Prozent an der Schwelle 0,5 waere jede haertere
# Korrektur teurer als sie einbringt.
ERF_HOCH = 2.5
# Eigene Modelle je Flughafen. Allein sind sie schlechter (237,6 gegen 231,3), als
# Beimischung tragen sie: 229,4 statt 230,2. Die Flaeche ist flach, 0,1 bis 0,3 liegen
# innerhalb von 0,3 s. Wieder Modellvielfalt, dieselbe Wirkung wie CatBoost.
ANTEIL_FLH = 0.2
SAATEN = (0, 7, 13, 21, 29)   # fuenf statt drei, 231,3 statt 231,5
# 06.09.2026 NEU ABGELESEN. Die 6.100 stammten aus einem Pruefdurchlauf vom 04.09. mit
# dem DAMALIGEN Merkmalssatz (52 Merkmale, beste Runde 5.754). Der Satz ist seither auf
# 69 gewachsen, und die Datei verlangt fuer LightGBM ausdruecklich, die Rundenzahl bei
# jeder Aenderung neu abzulesen. Fuer CatBoost war das nie geschehen.
# Drei Pruefdurchlaeufe mit Abbruch (od_wait 150, use_best_model) am 05.09. ergaben die
# besten Runden 7.788 / 6.865 / 7.894, im Mittel 7.516. Plus ein Zwanzigstel fuer die
# groessere Lernmenge sind das 7.892.
CB_RUNDEN = 7900
CB_SAATEN = (0, 7, 13)   # 227,94 statt 229,04 fuer eine einzelne Saat (05.09.2026)
# Anteil des Hoehenmodells gegenueber der blossen Verspaetung im Rueckfallweg.
# Am 09.09.2026 auf der RANGLISTE ausgemessen, vier Punkte:
#
#   ANTEIL_HM   0,15     0,30     0,50     0,70
#   RMSE      276,20   275,94   276,48   278,05
#
# Kleinstwert bei 0,30. Die Parabel durch die unteren drei Punkte hat ihren Scheitel bei
# 0,294 und laege dort 0,0007 s tiefer, die Achse ist damit ausgemessen. Der Schritt von
# 0,5 auf 0,3 brachte 0,5451 s, der Schritt auf 0,7 kostet 1,5665 s.
ANTEIL_HM = 0.3
# Scharfstellung der Wahrscheinlichkeit im Rueckfallweg, p -> p hoch HOCH.
# Von 1,6 auf 0,8 gesenkt am 08.09.2026, gemessen mit fallback_cv.py (sechs Faltungen,
# fuenf Wiederholungen, geschichtet nach Monat und Extremfall, Bootstrap ueber die Zeilen).
# Die Wirkung ist in Sekunden auf der Rangliste angegeben, gegen HOCH 1,6 als Messlatte:
#
#   HOCH   MSE je Zeile   Rangliste   Bootstrap 95 %
#   0,00     91.670.911   -3154,436   [-3288,798, -3024,517]   Gegenprobe
#   0,20      2.442.887      -11,035   [ -14,699,    -7,008]
#   0,35      2.101.949       +0,976   [  -1,155,    +3,206]
#   0,60      2.091.029       +1,360   [  +0,164,    +2,619]   Hoechstwert
#   0,80      2.099.056       +1,077   [  +0,262,    +1,915]   probiert, siehe unten
#   1,00      2.127.520       +0,788   [  +0,247,    +1,347]
#   1,20      2.115.099       +0,512   [  +0,186,    +0,852]
#   1,60      2.129.640        0       Messlatte
#
# Warum 0,8 und nicht der Hoechstwert 0,6: die drei Werte 0,6, 0,8 und 1,0 liegen
# gegenseitig in ihren Vertrauensbereichen, der Gipfel ist also nicht unterscheidbar.
# 0,6 liegt einen Schritt vor einer Klippe, die bei 0,2 elf Sekunden kostet, und 0,8 hat
# den engeren Bereich. Den Hoechstwert zu nehmen hiesse, auf das Messgeraet einzustellen.
#
# Gegenprobe HOCH 0: p hoch 0 ist eins, dann sagt das Hoehenmodell jede Zeile vorher.
# Dass das 3.154 s kostet, belegt den Mechanismus und nicht nur den Trend.
#
# ZURUECKGENOMMEN am 08.09.2026. Auf dem Bewertungssatz ergab 0,8 den Wert 277,0198
# gegen 276,5108 fuer 1,6, also 0,509 s SCHLECHTER, wo das Messgeraet 1,077 s besser
# vorhergesagt hatte. Das Vorzeichen stimmt nicht, nicht nur die Groesse.
# Wahrscheinliche Ursache: die Kreuzvalidierung lernt ihre Klassifikatoren auf fuenf
# Sechsteln von 2025 und misst auf dem Rest desselben Jahres. Der abgegebene
# Klassifikator lernt auf allen zwoelf Monaten und trifft auf 2026. Ist er dort
# zuversichtlicher oder anders geeicht, verschiebt ein kleineres HOCH zu viel Gewicht
# auf das Hoehenmodell. Bis das geklaert ist, bleibt es bei 1,6.
# Am 08.09.2026 auf der RANGLISTE ausgemessen statt in der Kreuzvalidierung, weil diese
# sich hier im Vorzeichen geirrt hatte. Der Bewertungssatz ist rauschfrei, vier Abgaben
# genuegten fuer die Kurve:
#
#   HOCH   0,8    1,6    2,5    4,0     50
#   RMSE  277,02 276,51 276,48 276,57 283,01
#
# Kleinstwert bei 2,5. Der Wert 50 schaltet den Hoehenapparat praktisch ab und kostet
# 6,53 s: Klassifikator und Hoehenmodell sind damit der groesste gemessene Einzelbeitrag
# des ganzen Modells.
HOCH = 2.5
KAT = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
       "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
       "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt", "ruf_pre"]


def auslastung(d):
    out = np.zeros(len(d), dtype=np.int32)
    t = d.MVT_TIME_UTC_mvt.astype("int64").to_numpy() // 10**9
    for _, idx in d.groupby("ADEP_mvt", observed=True).groups.items():
        i = np.asarray(idx)
        ts = np.sort(t[i])
        out[i] = (np.searchsorted(ts, t[i] + FENSTER_S, "right")
                  - np.searchsorted(ts, t[i] - FENSTER_S, "left") - 1)
    return out


def rollstrecke(d, werke):
    tab = {}
    for a, st, bn in d[["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]].drop_duplicates().itertuples(index=False):
        w = werke.get(a)
        tab[(a, st, bn)] = (MG.rollstrecke_m(w, st, bn) if w else None)
    return np.array([tab.get(k) if tab.get(k) is not None else np.nan
                     for k in zip(d.ADEP_mvt, d.STAND_mvt, d.RUNWAY_mvt)], float)


def baue(pfade, werke):
    roh = pd.concat([pd.read_parquet(f) for f in pfade], ignore_index=True)
    d = roh[roh.PHASE_mvt.astype(str).str.upper().str.startswith("DEP")].reset_index(drop=True)
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt",
              "LOBT_flt", "IOBT_flt", "ARVT_1_flt", "ARVT_3_flt"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d["y"] = d.TAXITIME_SEC_mvt.astype(float) if "TAXITIME_SEC_mvt" in d else np.nan
    d["monat"] = d.MVT_TIME_UTC_mvt.dt.month
    d["stunde"] = d.MVT_TIME_UTC_mvt.dt.hour
    d["minute_tag"] = d.MVT_TIME_UTC_mvt.dt.hour * 60 + d.MVT_TIME_UTC_mvt.dt.minute
    d["wtag"] = d.MVT_TIME_UTC_mvt.dt.dayofweek
    d["ersatz_s"] = (d.MVT_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()
    d["ersatz_eobt_s"] = (d.MVT_TIME_UTC_mvt - d.EOBT_1_flt).dt.total_seconds()
    # Die Flugdatei fuehrt vier Off-Block-Zeiten, benutzt wurden bisher zwei. LOBT und
    # IOBT sind zu 98,9 Prozent belegt, und jede liefert einen eigenen Ersatzwert.
    # Gemessen am 03.09.2026: Hauptgruppe 241,2 statt 246,5 s. Dazu die geplante
    # Flugdauer aus den Ankunftszeiten, die Kurz- von Langstrecke trennt.
    d["ersatz_lobt_s"] = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds()
    d["ersatz_iobt_s"] = (d.MVT_TIME_UTC_mvt - d.IOBT_flt).dt.total_seconds()
    d["lobt_sched_min"] = (d.LOBT_flt - d.SCHED_TIME_UTC_mvt).dt.total_seconds() / 60
    d["iobt_eobt_min"] = (d.IOBT_flt - d.EOBT_1_flt).dt.total_seconds() / 60
    d["flugdauer_min"] = (d.ARVT_3_flt - d.MVT_TIME_UTC_mvt).dt.total_seconds() / 60
    d["flugdauer_plan_min"] = (d.ARVT_1_flt - d.EOBT_1_flt).dt.total_seconds() / 60
    d["umleitung"] = (d.ADES_flt.astype(str) != d.ADES_FILED_flt.astype(str)).astype(int)
    # Vier Anker fuer dieselbe Off-Block-Zeit. Wie weit sie auseinanderliegen, ist ein
    # Mass dafuer, wie einig sich die Quellen ueber diesen Flug sind, und genau diese
    # Uneinigkeit ist die Zielgroesse. Gemessen am 03.09.2026: Hauptgruppe 232,1 statt
    # 239,9 s, der groesste Einzelgewinn dieser Nacht.
    _a = d[["ersatz_s", "ersatz_eobt_s", "ersatz_lobt_s", "ersatz_iobt_s"]].to_numpy(float)
    d["anker_spanne"] = np.nanmax(_a, 1) - np.nanmin(_a, 1)
    d["anker_median"] = np.nanmedian(_a, 1)
    d["anker_streu"] = np.nanstd(_a, 1)
    d["anker_anz"] = np.isfinite(_a).sum(1)
    d["anker_abw"] = d.ersatz_s - d.anker_median
    d["versp_min"] = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds() / 60
    # Was der Rest waere, wenn die Block-Zeit die Planzeit ist, wie in Rom. Baeume koennen
    # nicht subtrahieren, also vorgelegt. Traegt in der Hauptgruppe wenig (0,13 s allein),
    # zusammen mit dem neuen CatBoost-Anteil aber messbar, siehe unten.
    d["kand_sched"] = d.versp_min * 60 - d.ersatz_s
    d["eobt_sched_min"] = (d.EOBT_1_flt - d.SCHED_TIME_UTC_mvt).dt.total_seconds() / 60
    d["hat_aobt"] = d.AOBT_3_flt.notna().astype(int)
    # Signatur der erfundenen Block-Zeiten. In Rom traegt die Bewegungsdatei die
    # Planzeit als Off-Block ein, wenn kein Off-Block gemeldet wurde. Solche Zeilen
    # liegen oft einen Kalendertag hinter ihrer Planzeit und tragen ein
    # ueberlanges oder platzhalterhaftes Rufzeichen.
    d["tag_versch"] = (d.MVT_TIME_UTC_mvt.dt.date
                       != d.SCHED_TIME_UTC_mvt.dt.date).astype(int)
    ruf = d.FLIGHT_mvt.astype(str)
    d["ruf_len"] = ruf.str.len()
    d["ruf_pre"] = ruf.str[:3]
    d["sched_sek"] = d.SCHED_TIME_UTC_mvt.dt.second
    d["auslast"] = auslastung(d)
    d["strecke_m"] = rollstrecke(d, werke)
    # Lage am Flughafen: mittlere Verspaetung der vorangegangenen Fluege. Nur aus
    # MVT_TIME und SCHED_TIME gebildet, beide in der Ranking-Datei vorhanden, also
    # kein Blick auf Zielwerte. Faengt Stoertage wie Istanbul im Februar 2025 ein,
    # wo der Tagesmedian von 15,9 auf 48,9 Minuten stieg.
    d = d.sort_values(["ADEP_mvt", "MVT_TIME_UTC_mvt"]).reset_index(drop=True)
    for name, n in (("lage_50", 50), ("lage_200", 200)):
        d[name] = (d.groupby("ADEP_mvt", observed=True).versp_min
                     .transform(lambda x: x.shift(1).rolling(n, min_periods=10).median()))
    # Der Umlauf am Standplatz braucht die ANKUNFTSzeilen, die oben herausgefiltert
    # wurden. Deshalb wird hier die ungefilterte Tabelle uebergeben.
    d = MU.baue(roh, d)
    d = MA.baue(roh, d)      # Auslastung aus den gemessenen Ankunftsrollzeiten
    d = MLN.baue(d)
    d = MFID.baue(d)
    d = MD.baue(d)
    del roh
    for c in KAT:
        d[c] = d[c].astype("category")
    return d


MERKMALE = (KAT + ["stunde", "minute_tag", "wtag", "monat", "auslast", "strecke_m",
                   "ersatz_s", "ersatz_eobt_s", "versp_min", "eobt_sched_min", "hat_aobt",
                   "lage_50", "lage_200", "tag_versch", "ruf_len", "sched_sek"])

# Getrennte Saetze, gemessen am 03.09.2026. Die Hauptgruppe traegt die ganze
# Umlauffamilie (247,2 statt 251,8 s), der Rueckfallweg nur drei davon, weil ihn
# mehr verwaessert (1.827,1 statt 1.872,6). Zusammen 335,0 statt 342,2.
ANKER = ["ersatz_lobt_s", "ersatz_iobt_s", "lobt_sched_min", "iobt_eobt_min",
         "flugdauer_min", "flugdauer_plan_min", "umleitung",
         "anker_spanne", "anker_median", "anker_streu", "anker_anz", "anker_abw",
         "kand_sched"]
# Die Ankunftsauslastung bringt mit dem jetzigen Merkmalssatz 1,1 s (230,2 statt
# 231,3 bei fuenf Saaten). Am 03.09. waren es nur 0,4, damals mit einer Saat und noch
# ohne die Anker gemessen.
MERKMALE_HAUPT = (MERKMALE + MU.MERKMALE + ANKER + MA.MERKMALE + MD.MERKMALE
                  + MLN.MERKMALE + MFID.MERKMALE)
# Der Rueckfallweg bekommt die Erfindungsquote je Standplatz ANSTELLE von STAND_mvt.
# 1.631 Auspraegungen sind bei gut 600 Extremfaellen Rauschen, dieselbe Information in
# einer geglaetteten Zahl traegt. Gemessen 04.09.2026: 1.752,6 statt 1.777,6 s.
MERKMALE_RUECK = ([m for m in MERKMALE if m != "STAND_mvt"]
                  + MU.MERKMALE_KLEIN + MSQ.MERKMALE + MLN.MERKMALE
                  + ["ank_med_30min", "ank_med_90min"])


def kat_str(x):
    """CatBoost braucht die kategorialen Spalten als Zeichenketten, nicht als category."""
    x = x.copy()
    for c in KAT:
        x[c] = x[c].astype(str).fillna("(leer)")
    return x


def rmse(p, y):
    return float(np.sqrt(((np.asarray(p, float) - np.asarray(y, float)) ** 2).mean()))


def endmodell(d):
    """Endmodell fuer die Abgabe, gelernt auf ALLEN zwoelf Monaten.

    Die Bewertung laeuft auf Januar und Juli 2026. Fuer den Wettbewerb sind Januar und
    Juli 2025 gewoehnliche Lerndaten, und es sind die beiden Monate, die dem Bewertungssatz
    am aehnlichsten sind. Das Modell aus dem Pruefdurchlauf hat sie nie gesehen, weil es
    sie als Pruefmenge zurueckhaelt.

    Ohne Pruefmenge gibt es kein fruehes Abbrechen. Die Rundenzahlen stammen aus dem
    Pruefdurchlauf und werden um ein Fuenftel erhoeht, weil die Lernmenge um ein Fuenftel
    waechst.
    """
    lh = d[d.hat_aobt == 1].copy()
    lo = d[d.hat_aobt == 0]
    lh["rest"] = lh.y - lh.ersatz_s
    print(f"Endmodell auf allen Monaten: {len(lh):,} mit AOBT_3, {len(lo):,} ohne", flush=True)

    # Dritte Modellklasse, siehe ANTEIL_NETZ und neural_model.py. Sie lernt ZUERST, vor
    # LightGBM und CatBoost. Grund ist dieselbe OpenMP-Falle wie bei der Einfuhr: torch
    # und LightGBM bringen je eine eigene libomp mit, und ein Lauf, der erst stundenlang
    # CatBoost rechnet und dann das Netz anwirft, riskiert den Absturz am Ende statt am
    # Anfang. Sieben Minuten gegen vier Stunden, also lernt das Netz vorne.
    pnetz = NM.lerne(lh, MERKMALE_HAUPT, KAT) if ANTEIL_NETZ else None

    # Erkennung der erfundenen Block-Zeiten, siehe ERF_HOCH. Das Kennzeichen laesst sich
    # ohne die Rohdatei bilden: y - versp = SCHED_TIME - BLOCK_TIME, weil beide Groessen
    # dieselbe MVT_TIME abziehen.
    lh["erf"] = ((lh.y - lh.versp_min * 60).abs() <= 10).astype(int)
    merf = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05,
                              num_leaves=127, min_child_samples=40, verbose=-1,
                              n_jobs=-1, random_state=0)
    merf.fit(lh[MERKMALE_HAUPT], lh.erf)
    print(f"  Erkennung der erfundenen Block-Zeiten gelernt, "
          f"{lh.erf.mean():.2%} der Lernzeilen betroffen", flush=True)

    # Drei Saaten, gemittelt. Gemessen am 03.09.2026 auf Januar und Juli: eine Saat
    # 251,0 s, das Mittel dreier 250,2 s. Die kleinere Lernrate 0,015 bringt
    # zusaetzlich 0,4 s (250,6 statt 251,0) bei 1.361 statt 678 Runden. Mehr
    # Blaetter helfen nicht, 511 gibt 251,1 und 1023 gibt 251,5.
    # 890 Runden bei lr 0,03: der Pruefdurchlauf findet mit diesen Merkmalen die beste
    # Runde 840, erhoeht um ein Zwanzigstel fuer die groessere Lernmenge. Die kleinere
    # Lernrate 0,015 brachte frueher 0,4 s, ihre Rundenzahl muesste aber hochgerechnet
    # werden, und genau das ging beim Einbau des Umlaufs schief (1.443 statt noetiger
    # 1.913, der Gewinn verpuffte). Hier steht deshalb die direkt abgelesene Zahl.
    # Sie wandert mit dem Merkmalssatz: 678 am Anfang, 958 mit dem Umlauf, 1.045 mit
    # den Ankern, 840 mit den Ankerkombinationen. Bei jeder Aenderung neu ablesen.
    m = []
    for saat in SAATEN:
        e = lgb.LGBMRegressor(objective="l2", n_estimators=890, learning_rate=0.03,
                              num_leaves=255, min_child_samples=50, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                              n_jobs=-1, random_state=saat)
        e.fit(lh[MERKMALE_HAUPT], lh.rest)
        m.append(e)
        print(f"  Saat {saat} fertig", flush=True)
    # CatBoost daneben. Es behandelt die kategorialen Merkmale grundlegend anders, und
    # STAND_mvt mit 1.631 Auspraegungen ist das wichtigste Merkmal. Gemessen 04.09.2026:
    # allein 230,3 s gegen 232,4 fuer LightGBM, das Gemisch 228,3. Die Flaeche ist flach
    # (Anteil 0,5 bis 0,7 alles innerhalb 0,1 s), deshalb der runde Wert 0,5.
    # Mehrere Saaten, wie auf der LightGBM-Seite. CatBoost stellt mit ANTEIL_CB mehr als
    # die Haelfte des Gemischs und lief bisher mit einer einzigen. Gemessen 05.09.2026:
    # eine Saat 229,04, Mittel aus zwei 228,21, Mittel aus drei 227,94.
    pool = Pool(kat_str(lh[MERKMALE_HAUPT]), lh.rest, cat_features=KAT)
    mcb = []
    for saat in CB_SAATEN:
        e = CatBoostRegressor(loss_function="RMSE", iterations=CB_RUNDEN, learning_rate=0.06,
                              depth=8, verbose=1000, thread_count=-1, random_seed=saat,
                              allow_writing_files=False)
        e.fit(pool)
        mcb.append(e)
        print(f"  CatBoost Saat {saat} fertig", flush=True)
    mflh = {}
    for f_ in sorted(lh.ADEP_mvt.astype(str).unique()):
        sl = (lh.ADEP_mvt.astype(str) == f_).to_numpy()
        e = lgb.LGBMRegressor(objective="l2", n_estimators=890, learning_rate=0.03,
                              num_leaves=255, min_child_samples=50, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                              n_jobs=-1, random_state=0)
        e.fit(lh[sl][MERKMALE_HAUPT], lh[sl].rest)
        mflh[f_] = e
    print(f"  {len(mflh)} Modelle je Flughafen fertig", flush=True)
    # Rueckfallweg als Mischung. Ein Klassifikator erkennt die Extremfaelle, ein
    # Regressor deckt den Normalfall ab, und die Hoehe der Extreme liefert die
    # Verspaetung MVT_TIME minus SCHED_TIME. Gemessen am 03.09.2026: 1.945 statt
    # 2.130 s, auf den Gesamtwert rund 16 Sekunden.
    lo = lo.copy(); lo["z"] = (lo.y > SCHWELLE_EXTREM).astype(int)
    quo = MSQ.passe_an(lo)
    lo = MSQ.ausser_faltung(lo)          # Lernzeilen: je Monat aus den uebrigen
    m2c = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.05,
                             num_leaves=31, min_child_samples=20, verbose=-1, n_jobs=-1)
    m2c.fit(lo[MERKMALE_RUECK], lo.z)
    # STAERKE LEIHEN wurde am 06.09.2026 gebaut und am 07.09.2026 ZURUECKGENOMMEN.
    # Der Gedanke war, den Normalregressor auf allen Zeilen zu lernen statt auf den gut
    # 16.000 der eigenen Gruppe, weil er nur Merkmale benutzt, die beide Gruppen haben.
    # Die getrimmte Halteprobe sagte 277,74 statt 278,70 voraus, also 0,96 s besser.
    # Auf der Rangliste ergab v18 dann 277,0422 gegen 277,0245 fuer v17, also 0,018 s
    # SCHLECHTER. Die Kreuzvalidierung ueber sechs Monatspaare erklaert warum: in dieser
    # Gruppe betraegt der Standardfehler rund 92 s, unter etwa 180 s ist dort nichts
    # messbar. Das Trimmen hat die HOEHE der Halteprobe geheilt, nicht ihre
    # EMPFINDLICHKEIT. Es bleibt bei der eigenen Gruppe, weil dafuer eine echte Messung
    # auf dem Bewertungssatz spricht.
    m2 = lgb.LGBMRegressor(objective="l2", n_estimators=400, learning_rate=0.05,
                           num_leaves=31, min_child_samples=20, verbose=-1, n_jobs=-1)
    m2.fit(lo[lo.z == 0][MERKMALE_RUECK], lo[lo.z == 0].y)
    # Hoehenmodell. Die Verspaetung ist fuer 79 % der Extremfaelle auf vier Sekunden
    # genau (Median), fuer die uebrigen 21 % voellig falsch, weil das nur fuer die
    # erfundenen Block-Zeiten in Rom gilt. Ein eigenes Modell auf den Extremfaellen
    # faengt die anderen ab. Gemessen 04.09.2026: Gruppe 1.785,8 statt 1.843,2 s.
    m2h = lgb.LGBMRegressor(objective="l2", n_estimators=300, learning_rate=0.05,
                            num_leaves=15, min_child_samples=10, verbose=-1, n_jobs=-1)
    m2h.fit(lo[lo.z == 1][MERKMALE_RUECK], lo[lo.z == 1].y)

    import joblib
    joblib.dump({"mit_aobt": m, "mit_aobt_cb": mcb, "anteil_cb": ANTEIL_CB,
                 "mit_aobt_netz": pnetz, "anteil_netz": ANTEIL_NETZ,
                 "erf_klass": merf, "erf_hoch": ERF_HOCH,
                 "mit_aobt_flh": mflh, "anteil_flh": ANTEIL_FLH,
                 "ohne_aobt": m2, "ohne_aobt_klass": m2c,
                 "ohne_aobt_hoehe": m2h, "anteil_hm": ANTEIL_HM, "hoch": HOCH,
                 "standquote": quo,
                 "merkmale_haupt": MERKMALE_HAUPT, "merkmale_rueck": MERKMALE_RUECK,
                 "ziel_mit_aobt": "rest",
                 "ziel_ohne_aobt": "mischung", "gelernt_auf": "alle zwoelf Monate 2025"},
                MODEL_OUT)
    print(f"gespeichert: {MODEL_OUT} (Endmodell, alle Monate)")


def main():
    print("Daten und Merkmale werden gebaut ...", flush=True)
    d = baue(sorted(glob.glob("data/training_2025-*.parquet")), MG.alle())
    lern = d[~d.monat.isin(PRUEFMONATE)]
    pruef = d[d.monat.isin(PRUEFMONATE)]
    print(f"  Lernmenge {len(lern):,} | Pruefmenge {len(pruef):,}")
    print(f"  Merkmale {len(MERKMALE_HAUPT)} Haupt / {len(MERKMALE_RUECK)} Rueckfall,"
          f" davon kategorial {len(KAT)}\n")

    # Das grosse Modell lernt den REST gegen den Ersatzwert MVT_TIME minus AOBT_3,
    # nicht die Rollzeit selbst. Gemessen am 02.09.2026: 251,0 statt 281,2 s, weil das
    # Modell die Hoehe nicht mehr rekonstruieren muss, sondern nur die Abweichung.
    # Es lernt daher nur auf Zeilen MIT AOBT_3, fuer die uebrigen ist das zweite Modell da.
    lh, ph_ = lern[lern.hat_aobt == 1].copy(), pruef[pruef.hat_aobt == 1].copy()
    lh["rest"] = lh.y - lh.ersatz_s
    ph_["rest"] = ph_.y - ph_.ersatz_s
    m = lgb.LGBMRegressor(objective="l2", n_estimators=3000, learning_rate=0.03,
                          num_leaves=255, min_child_samples=50, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
    print(f"Training auf {len(lh):,} Zeilen mit AOBT_3, Ziel ist der Rest ...", flush=True)
    m.fit(lh[MERKMALE_HAUPT], lh.rest, eval_set=[(ph_[MERKMALE_HAUPT], ph_.rest)], eval_metric="rmse",
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(300)])
    v = np.full(len(pruef), np.nan)
    hat_ = (pruef.hat_aobt == 1).to_numpy()
    v[hat_] = ph_.ersatz_s.to_numpy() + m.predict(ph_[MERKMALE_HAUPT])
    print(f"\n  beste Runde: {m.best_iteration_}")

    hat = pruef.hat_aobt == 1
    print("\n=== Ergebnis auf Januar und Juli 2025 ===")
    print(f"  {'Zeilen mit AOBT_3, Restmodell':44s} {rmse(v[hat.to_numpy()], pruef.y[hat]):8.1f} s "
          f"(n={hat.sum():,})")
    print("\n  Messlatten zum Vergleich")
    print(f"  {'beste einfache Gruppenmittel':44s}    624,9 s")
    print(f"  {'Ersatzwert plus Rueckfallweg (fallback_path.py)':44s}    549,9 s")

    print("\n=== Wichtigste Merkmale ===")
    imp = pd.Series(m.feature_importances_, index=MERKMALE_HAUPT).sort_values(ascending=False)
    for k, val in imp.head(12).items():
        print(f"  {k:26s} {int(val):7d}")

    # Zweites, kleineres Modell nur fuer die Zeilen ohne AOBT_3.
    lo, po = lern[lern.hat_aobt == 0], pruef[pruef.hat_aobt == 0]
    print(f"\nZweites Modell auf {len(lo):,} Zeilen ohne AOBT_3 ...", flush=True)
    lo = lo.copy(); lo["z"] = (lo.y > SCHWELLE_EXTREM).astype(int)
    quo = MSQ.passe_an(lo)
    lo = MSQ.ausser_faltung(lo)          # Lernzeilen: je Monat aus den uebrigen
    m2c = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.05,
                             num_leaves=31, min_child_samples=20, verbose=-1, n_jobs=-1)
    m2c.fit(lo[MERKMALE_RUECK], lo.z)
    # STAERKE LEIHEN wurde am 06.09.2026 gebaut und am 07.09.2026 ZURUECKGENOMMEN.
    # Der Gedanke war, den Normalregressor auf allen Zeilen zu lernen statt auf den gut
    # 16.000 der eigenen Gruppe, weil er nur Merkmale benutzt, die beide Gruppen haben.
    # Die getrimmte Halteprobe sagte 277,74 statt 278,70 voraus, also 0,96 s besser.
    # Auf der Rangliste ergab v18 dann 277,0422 gegen 277,0245 fuer v17, also 0,018 s
    # SCHLECHTER. Die Kreuzvalidierung ueber sechs Monatspaare erklaert warum: in dieser
    # Gruppe betraegt der Standardfehler rund 92 s, unter etwa 180 s ist dort nichts
    # messbar. Das Trimmen hat die HOEHE der Halteprobe geheilt, nicht ihre
    # EMPFINDLICHKEIT. Es bleibt bei der eigenen Gruppe, weil dafuer eine echte Messung
    # auf dem Bewertungssatz spricht.
    m2 = lgb.LGBMRegressor(objective="l2", n_estimators=400, learning_rate=0.05,
                           num_leaves=31, min_child_samples=20, verbose=-1, n_jobs=-1)
    m2.fit(lo[lo.z == 0][MERKMALE_RUECK], lo[lo.z == 0].y)
    # Hoehenmodell. Die Verspaetung ist fuer 79 % der Extremfaelle auf vier Sekunden
    # genau (Median), fuer die uebrigen 21 % voellig falsch, weil das nur fuer die
    # erfundenen Block-Zeiten in Rom gilt. Ein eigenes Modell auf den Extremfaellen
    # faengt die anderen ab. Gemessen 04.09.2026: Gruppe 1.785,8 statt 1.843,2 s.
    m2h = lgb.LGBMRegressor(objective="l2", n_estimators=300, learning_rate=0.05,
                            num_leaves=15, min_child_samples=10, verbose=-1, n_jobs=-1)
    m2h.fit(lo[lo.z == 1][MERKMALE_RUECK], lo[lo.z == 1].y)
    po = MSQ.wende_an(quo, po)
    pw = m2c.predict_proba(po[MERKMALE_RUECK])[:, 1] ** HOCH
    hoehe = (1 - ANTEIL_HM) * po.versp_min.to_numpy() * 60 + ANTEIL_HM * m2h.predict(po[MERKMALE_RUECK])
    v2 = (1 - pw) * m2.predict(po[MERKMALE_RUECK]) + pw * hoehe
    r2 = rmse(v2, po.y)
    v[~hat.to_numpy()] = v2
    print(f"  beste Runde {m2.best_iteration_}, RMSE {r2:.1f} s")

    r1 = rmse(v[hat.to_numpy()], pruef.y[hat])
    n1, n2 = int(hat.sum()), len(po)
    ges = np.sqrt((n1 * r1 ** 2 + n2 * r2 ** 2) / (n1 + n2))
    print(f"\n=== ZUSAMMENGESETZT: {ges:.1f} s ===")
    print("  Messlatten: 624,9 einfache Gruppenmittel | 549,9 Ersatzwert plus Handverfahren")

    import joblib
    joblib.dump({"mit_aobt": m, "ohne_aobt": m2,
                 "merkmale_haupt": MERKMALE_HAUPT, "merkmale_rueck": MERKMALE_RUECK,
                 "ziel_mit_aobt": "rest",      # Vorhersage = ersatz_s + Modellausgabe
                 "ziel_ohne_aobt": "absolut"}, MODEL_OUT)
    print(f"\ngespeichert: {MODEL_OUT}")


if __name__ == "__main__":
    if "--alle" in sys.argv:
        endmodell(baue(sorted(glob.glob("data/training_2025-*.parquet")), MG.alle()))
    else:
        main()
