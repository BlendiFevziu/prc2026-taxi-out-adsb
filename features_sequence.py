"""features_sequence.py - Was die laufende Nummer ueber erfundene Zeiten verraet.

`MVT_ID_mvt` ist eine aufsteigende Nummer, ueber alle Zeilen mit Spearman 0,9929 an
die Bewegungszeit gekoppelt. Wo sie davon abweicht, ist etwas an der Entstehung des
Datensatzes ungewoehnlich.

**Die erfundenen Zeilen tragen eine ZU NIEDRIGE Nummer** gemessen an ihren zeitlichen
Nachbarn, und das passt zum Mechanismus: Der Datensatz entsteht nahe der Planzeit und
bekommt dort seine Nummer, wird nie mit einer echten Off-Block-Zeit nachgefuehrt, und
die Bewegungszeit landet Stunden spaeter. Gemessen 04.09.2026 ueber die Sechstel der
Abweichung, LIRF ohne AOBT_3: Extremquote 64,3 % im untersten gegen 24,2 % im
vorletzten Sechstel.

Im Modell traegt es wenig (Gruppe 1.741,2 statt 1.752,6), und die Trennschaerfe sinkt
dabei sogar leicht. Der Grund steht in der Lernkurve: die Rueckfallgruppe hat 623
Extremfaelle und ist nicht durch Merkmale, sondern durch Beispiele begrenzt.

Aufruf: von main_model.py und submission.py.
"""
import numpy as np
import pandas as pd

MERKMALE = ["id_abw", "id_abw2"]


def baue(dep):
    d = dep.copy()
    d["_i"] = np.arange(len(d))
    d["_id"] = d.MVT_ID_mvt.astype("int64")
    d = d.sort_values(["ADEP_mvt", "MVT_TIME_UTC_mvt"])
    g = d.groupby("ADEP_mvt", observed=True)._id
    for nm, fen, mind in (("id_abw", 41, 5), ("id_abw2", 201, 20)):
        lok = g.transform(lambda x: x.rolling(fen, center=True, min_periods=mind).median())
        d[nm] = d._id - lok
    return d.sort_values("_i").drop(columns=["_i", "_id"])
