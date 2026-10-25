"""features_reporting_state.py - wie der Flughafen gerade Zeitstempel schreibt, gemessen an seinen Ankuenften.

Das Ziel ist keine Rolldauer, sondern ein **geschriebener Zeitstempel**: y ist MVT_TIME minus BLOCK_TIME, und die
Hauptbahn sagt genau die Differenz zwischen der Uhr des Network Managers und dem Blockeintrag des Flughafens vorher
(rest = AOBT_3 - BLOCK_TIME). Kein Merkmal misst bisher, wie der Flughafen im Moment des Abflugs schreibt.

Die Ankunftsseite macht das sichtbar, und zwar auch zur Bewertungszeit: in `ranking.parquet` sind BLOCK_TIME,
TAXITIME und SCHED_TIME der **Ankuenfte** zu 100 Prozent belegt, die der Abfluege zu 0 Prozent.
`features_arrivals.py` nutzt davon die Rollzeit als Auslastungsmass; hier wird dieselbe Zeile anders befragt.

Gemessen 16.09.2026 ueber alle 2.085.047 Abfluege des Jahres 2025, Ankuenfte desselben Flughafens im
Zweistundenfenster:

- **Sekundenanteil** (Blockzeit der Ankunft traegt eine Sekundenstelle ungleich null). Von Fuenftel zu Fuenftel
  0,877 bis 1,000 faellt die Streuung des Ziels der Hauptbahn von **415,5 auf 310,1 Sekunden**, der Median von
  |rest| von 179 auf 146. Auf der Rueckfallbahn faellt die Kopierquote von 8,67 auf 4,90 Prozent und die
  Extremquote von 6,14 auf 1,37 Prozent. Ein maschinell gesetzter Stempel traegt Sekunden, ein getippter oder
  abgeleiteter ist rund.
- **Kopierquote der Ankuenfte** (Blockzeit gleich Planzeit, Toleranz 60 s). Ueber die Fuenftel steigt die
  Kopierquote der Rueckfallabfluege von 2,93 auf **14,33** Prozent und die Extremquote von 1,94 auf **10,42**
  Prozent. Rom liegt mit 12,1 Prozent Ankunftskopien weit vor allen anderen (4 bis 5,5 Prozent).
- **Fehlender NM-Satz der Ankuenfte** wirkt schwaecher und bleibt als drittes Mass dabei.

Nicht zu verwechseln mit dem verworfenen Merkmal aus research_270: dort ging es um die Ankunft **desselben Fluges**
(tot, 5,87 gegen 5,60 Prozent). Hier ist es der Zustand des Flughafens im Fenster, nicht der einzelne Flug.

Vorwaerts gerichtete Fenster sind zulaessig, MVT_TIME steht fuer jede Zeile der Rangdatei, und the organiser hat
Merkmale aus anderen Bewegungszeilen ausdruecklich erlaubt (09.09.2026). Es geht keine Zielgroesse ein: auf der
Abflugseite werden BLOCK_TIME und TAXITIME nie gelesen.

Aufruf: aus den Screens, spaeter aus main_model.py und submission.py.
"""
import numpy as np
import pandas as pd

RAW_COLUMNS = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
               "SCHED_TIME_UTC_mvt", "ARVT_3_flt"]
DEP_COLUMNS = ["ADEP_mvt", "MVT_TIME_UTC_mvt"]
FENSTER = 5400  # Sekunden, plus minus
MERKMALE = ["rs_sek_90", "rs_kopie_90", "rs_nmfehlt_90", "rs_n_90", "rs_sek_tag", "rs_kopie_tag"]


def _fenstermittel(ank_zeit, werte, ab_zeit):
    """Mittelwerte der Ankunftsgroessen im Fenster um jede Abflugzeit, ueber Praefixsummen."""
    links = np.searchsorted(ank_zeit, ab_zeit - FENSTER, side="left")
    rechts = np.searchsorted(ank_zeit, ab_zeit + FENSTER, side="right")
    anzahl = (rechts - links).astype(float)
    mittel = {}
    for name, wert in werte.items():
        summe = np.concatenate([[0.0], np.cumsum(wert.astype(float))])
        mittel[name] = np.where(anzahl > 0, (summe[rechts] - summe[links]) / np.maximum(anzahl, 1), np.nan)
    return mittel, anzahl


def baue(roh, dep):
    """roh: alle Zeilen einschliesslich Ankuenfte. dep: die Abflugtabelle, Index wird erhalten."""
    r = roh[[c for c in RAW_COLUMNS if c in roh.columns]].copy()
    for spalte in ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"):
        if spalte in r:
            r[spalte] = pd.to_datetime(r[spalte], errors="coerce", utc=True)
    ank = r[r.PHASE_mvt.astype(str).str.upper().str.startswith("ARR")].copy()
    ank = ank[ank.MVT_TIME_UTC_mvt.notna() & ank.BLOCK_TIME_UTC_mvt.notna()]
    ank["flughafen"] = ank.ADES_mvt.astype(str)
    ank["sek"] = ank.BLOCK_TIME_UTC_mvt.dt.second.ne(0).to_numpy(float)
    ank["kopie"] = (ank.BLOCK_TIME_UTC_mvt - ank.SCHED_TIME_UTC_mvt).dt.total_seconds().abs().le(60).to_numpy(float)
    ank["nmfehlt"] = ank.ARVT_3_flt.isna().to_numpy(float) if "ARVT_3_flt" in ank else 0.0
    ank["tag"] = ank.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")

    ab = dep[DEP_COLUMNS].copy()
    ab["MVT_TIME_UTC_mvt"] = pd.to_datetime(ab.MVT_TIME_UTC_mvt, errors="coerce", utc=True)
    ab["flughafen"] = ab.ADEP_mvt.astype(str)
    ab["tag"] = ab.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")

    ergebnis = pd.DataFrame(index=dep.index, columns=MERKMALE, dtype=float)
    for flughafen, gruppe in ab.groupby("flughafen", observed=True):
        quelle = ank[ank.flughafen == flughafen]
        if quelle.empty:
            continue
        quelle = quelle.sort_values("MVT_TIME_UTC_mvt")
        ank_zeit = quelle.MVT_TIME_UTC_mvt.to_numpy("datetime64[s]").astype("int64")
        ab_zeit = gruppe.MVT_TIME_UTC_mvt.to_numpy("datetime64[s]").astype("int64")
        reihe = np.argsort(ab_zeit, kind="stable")
        mittel, anzahl = _fenstermittel(ank_zeit, {"rs_sek_90": quelle.sek.to_numpy(),
                                                   "rs_kopie_90": quelle.kopie.to_numpy(),
                                                   "rs_nmfehlt_90": quelle.nmfehlt.to_numpy()}, ab_zeit[reihe])
        zurueck = np.empty_like(reihe)
        zurueck[reihe] = np.arange(len(reihe))
        for name, werte in mittel.items():
            ergebnis.loc[gruppe.index, name] = werte[zurueck]
        ergebnis.loc[gruppe.index, "rs_n_90"] = anzahl[zurueck]
    tag = ank.groupby(["flughafen", "tag"], observed=True).agg(rs_sek_tag=("sek", "mean"),
                                                               rs_kopie_tag=("kopie", "mean"))
    schluessel = pd.MultiIndex.from_arrays([ab.flughafen, ab.tag])
    for name in ("rs_sek_tag", "rs_kopie_tag"):
        ergebnis[name] = tag[name].reindex(schluessel).to_numpy()
    return ergebnis


def _selbsttest():
    t = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    ank = pd.DataFrame({
        "PHASE_mvt": ["ARR"] * 4, "ADEP_mvt": ["XXXX"] * 4, "ADES_mvt": ["TEST"] * 4,
        "MVT_TIME_UTC_mvt": [t, t + pd.Timedelta(minutes=10), t + pd.Timedelta(minutes=20), t + pd.Timedelta(hours=5)],
        "BLOCK_TIME_UTC_mvt": [t + pd.Timedelta(seconds=17), t + pd.Timedelta(minutes=10),
                               t + pd.Timedelta(minutes=20, seconds=5), t + pd.Timedelta(hours=5)],
        "SCHED_TIME_UTC_mvt": [t, t + pd.Timedelta(minutes=10), t + pd.Timedelta(minutes=30), t + pd.Timedelta(hours=5)],
        "ARVT_3_flt": [t, None, t, t]})
    dep = pd.DataFrame({"PHASE_mvt": ["DEP"], "ADEP_mvt": ["TEST"], "ADES_mvt": ["YYYY"],
                        "MVT_TIME_UTC_mvt": [t + pd.Timedelta(minutes=15)],
                        "BLOCK_TIME_UTC_mvt": [pd.NaT], "SCHED_TIME_UTC_mvt": [t], "ARVT_3_flt": [None]})
    gebaut = baue(pd.concat([ank, dep], ignore_index=True), dep)
    assert gebaut.rs_n_90.iloc[0] == 3, gebaut.rs_n_90.iloc[0]
    assert abs(gebaut.rs_sek_90.iloc[0] - 2 / 3) < 1e-9, gebaut.rs_sek_90.iloc[0]
    assert abs(gebaut.rs_kopie_90.iloc[0] - 2 / 3) < 1e-9, gebaut.rs_kopie_90.iloc[0]
    assert abs(gebaut.rs_nmfehlt_90.iloc[0] - 1 / 3) < 1e-9, gebaut.rs_nmfehlt_90.iloc[0]
    # Die Abflugseite darf nichts beisteuern: Ziel und Blockzeit werden nie gelesen.
    andere = baue(pd.concat([ank, dep.assign(TAXITIME_SEC_mvt=-999)], ignore_index=True), dep)
    pd.testing.assert_frame_equal(gebaut, andere)
    print("Berichtszustand geprueft: Fenster, Tageswerte, Sekunden- und Kopienanteil, Zielunabhaengigkeit.")


if __name__ == "__main__":
    _selbsttest()
