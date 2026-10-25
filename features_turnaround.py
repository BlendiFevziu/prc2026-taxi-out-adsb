"""features_turnaround.py - Der Umlauf am Standplatz.

Ein Flugzeug kann nicht abrollen, bevor es angekommen ist. Der Abstand zwischen dem
In-Block der letzten Ankunft auf demselben Standplatz und der eigenen Startzeit ist
damit eine obere Schranke fuer die Rollzeit, und ausserdem ein Mass fuer die Enge
des Umlaufs.

Als harte Kappung ist sie unbrauchbar, sie wird in 9,8 Prozent der Faelle verletzt
(Standplatzwechsel, Schleppen, eine fremde Ankunft auf demselben Platz). Gemessen am
03.09.2026 verschlechtert Kappen die Hauptgruppe von 251,8 auf 384,4 s. **Als
Merkmal dagegen traegt sie**: Hauptgruppe 248,2 statt 251,8, Rueckfallweg 1.857,5
statt 1.872,6.

In der Rangdatei ist alles davon vorhanden, denn deren Ankunftszeilen fuehren
BLOCK_TIME_UTC_mvt und TAXITIME_SEC_mvt zu 100 Prozent, waehrend nur die
Abflugzeilen leer sind.

Aufruf: von main_model.py und submission.py.
"""
import numpy as np
import pandas as pd

MERKMALE = ["um_seit_ank", "um_taxi_ank", "um_bis_naechste_ank",
            "um_seit_abflug", "um_bis_naechster_abflug", "um_dichte_stand_3h",
            "um_ver_ank", "um_versp_ank"]

# Der Rueckfallweg lernt aus 22.470 Zeilen mit nur 370 Extremfaellen und
# verschlechtert sich, sobald er zu viele Merkmale bekommt. Gemessen am 03.09.2026,
# Fehler der Gruppe: ganze Familie 1.863,0 | nur die Schranke 1.857,5 | diese drei
# 1.827,1 | nur der Vorflugversatz 1.878,8 | ohne alles 1.872,6.
MERKMALE_KLEIN = ["um_seit_ank", "um_ver_ank", "um_versp_ank"]


def _sek(a, b):
    """Differenz zweier Zeitfelder in Sekunden, NaT wird zu NaN.

    Die Zeiten tragen eine Zeitzone, deshalb liefert to_numpy() ein Objektfeld und
    weder np.timedelta64 noch der .dt-Zugriff funktionieren. Ueber DatetimeIndex
    geht es, und NaT wandert sauber nach NaN.
    """
    a = pd.DatetimeIndex(pd.to_datetime(a, utc=True))
    b = pd.DatetimeIndex(pd.to_datetime(b, utc=True))
    return (a - b).total_seconds().to_numpy()


def baue(roh, dep):
    r = roh.copy()
    ph = r.PHASE_mvt.astype(str).str.upper()
    an = r[ph.str.startswith("ARR")].copy()
    an["sch"] = an.ADES_mvt.astype(str) + "|" + an.STAND_mvt.astype(str)
    an["t_block"] = pd.to_datetime(an.BLOCK_TIME_UTC_mvt)
    an["t_mvt"] = pd.to_datetime(an.MVT_TIME_UTC_mvt)
    an["taxi"] = pd.to_numeric(an.TAXITIME_SEC_mvt, errors="coerce")
    # Meldeversatz des Vorflugs: um wieviel weichen Flugdatei und Bewegungsdatei bei
    # DIESER Ankunft voneinander ab. Das Ziel des Hauptmodells ist derselbe Versatz
    # auf der Abflugseite, und beide Male melden dieselben zwei Systeme denselben
    # Umlauf. Ausserdem die Ankunftsverspaetung, die die Enge des Umlaufs misst.
    an["ver"] = _sek(pd.to_datetime(an.ARVT_3_flt, utc=True).to_numpy(),
                     an.t_block.to_numpy())
    an["versp"] = _sek(an.t_mvt.to_numpy(),
                       pd.to_datetime(an.SCHED_TIME_UTC_mvt, utc=True).to_numpy())
    ab = r[ph.str.startswith("DEP")].copy()
    ab["sch"] = ab.ADEP_mvt.astype(str) + "|" + ab.STAND_mvt.astype(str)
    ab["t_mvt"] = pd.to_datetime(ab.MVT_TIME_UTC_mvt)

    dep = dep.copy()
    dep["sch"] = dep.ADEP_mvt.astype(str) + "|" + dep.STAND_mvt.astype(str)
    dep["t_mvt"] = pd.to_datetime(dep.MVT_TIME_UTC_mvt)
    dep["_i"] = np.arange(len(dep))
    li = dep[["sch", "t_mvt", "_i"]].sort_values("t_mvt")

    # letzte Ankunft davor: In-Block und ihre Rollzeit
    q = an.dropna(subset=["t_block"]).sort_values("t_block")[
        ["sch", "t_block", "taxi", "ver", "versp"]]
    j = pd.merge_asof(li, q, left_on="t_mvt", right_on="t_block", by="sch",
                      direction="backward").sort_values("_i")
    dep["um_seit_ank"] = _sek(dep.t_mvt.to_numpy(), j.t_block.to_numpy())
    dep["um_taxi_ank"] = j.taxi.to_numpy()
    dep["um_ver_ank"] = j.ver.to_numpy()
    dep["um_versp_ank"] = j.versp.to_numpy()

    # naechste Ankunft danach: wie sehr draengt der Nachfolger auf den Platz
    q = an.dropna(subset=["t_mvt"]).sort_values("t_mvt")[["sch", "t_mvt"]].rename(
        columns={"t_mvt": "t_na"})
    j = pd.merge_asof(li, q, left_on="t_mvt", right_on="t_na", by="sch",
                      direction="forward").sort_values("_i")
    dep["um_bis_naechste_ank"] = _sek(j.t_na.to_numpy(), dep.t_mvt.to_numpy())

    # voriger und naechster Abflug vom selben Platz
    q = ab.dropna(subset=["t_mvt"]).sort_values("t_mvt")[["sch", "t_mvt"]].rename(
        columns={"t_mvt": "t_ab"})
    for nm, ri in (("um_seit_abflug", "backward"), ("um_bis_naechster_abflug", "forward")):
        j = pd.merge_asof(li, q, left_on="t_mvt", right_on="t_ab", by="sch",
                          direction=ri, allow_exact_matches=False).sort_values("_i")
        v = _sek(dep.t_mvt.to_numpy(), j.t_ab.to_numpy())
        dep[nm] = v if ri == "backward" else -v

    # Belegungsdichte des Platzes: Bewegungen im Fenster von drei Stunden
    alle = pd.concat([an[["sch", "t_mvt"]], ab[["sch", "t_mvt"]]]).dropna()
    alle = alle.sort_values("t_mvt")
    ts = {k: np.sort(v.astype("int64") // 10**9) for k, v in alle.groupby("sch").t_mvt}
    t = dep.t_mvt.astype("int64").to_numpy() // 10**9
    out = np.zeros(len(dep))
    for k, idx in dep.groupby("sch").groups.items():
        i = np.asarray(idx)
        a = ts.get(k)
        if a is None:
            continue
        out[i] = (np.searchsorted(a, t[i] + 5400, "right")
                  - np.searchsorted(a, t[i] - 5400, "left"))
    dep["um_dichte_stand_3h"] = out
    return dep.drop(columns=["sch", "t_mvt", "_i"])
