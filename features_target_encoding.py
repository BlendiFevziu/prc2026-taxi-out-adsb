"""features_target_encoding.py - Erlernte Rollzeit je Standplatz und Bahn.

`strecke_m` ist eine geometrische Entfernung zwischen Standplatz und Bahn. Die
tatsaechliche Rollzeit haengt aber an der Fuehrung des Rollwegs, an Kreuzungen und an
der Reihenfolge, und die stehen in den Lerndaten. LightGBM sieht STAND_mvt und
RUNWAY_mvt als getrennte kategoriale Merkmale und muesste die Verknuepfung von 1.631
Standplaetzen mit 48 Bahnen selbst finden, wofuer 255 Blaetter nicht reichen.

Hier wird sie ihm vorgelegt: der Median der Rollzeit je Schluessel, geglaettet gegen
den naechstgroeberen Schluessel, plus die Zahl der Faelle, damit das Modell die
Verlaesslichkeit selbst gewichten kann.

**Angepasst wird ausschliesslich auf den Lernzeilen**, angewandt auf Lern- und
Pruefzeilen. Im Endmodell also auf allen zwoelf Monaten 2025, angewandt auf 2026.

Aufruf: von main_model.py und submission.py.
"""
import numpy as np
import pandas as pd

SCHLUESSEL = [("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"),
              ("ADEP_mvt", "STAND_mvt"),
              ("ADEP_mvt", "RUNWAY_mvt"),
              ("ADEP_mvt", "AIRCRAFT_OPERATOR_flt")]
MIND = 20


def _nam(k):
    return "zk_" + "_".join(s.split("_")[0].lower() for s in k)


MERKMALE = [f(k) for k in SCHLUESSEL for f in (_nam, lambda k: _nam(k) + "_n")]


class Kodierung:
    def __init__(self):
        self.tab, self.flh, self.ges = {}, None, None

    def passe_an(self, lern):
        d = lern[lern.y.notna()]
        self.ges = float(d.y.median())
        self.flh = d.groupby(d.ADEP_mvt.astype(str)).y.median()
        for k in SCHLUESSEL:
            g = d.groupby([d[c].astype(str) for c in k]).y.agg(["median", "size"])
            self.tab[k] = g[g["size"] >= MIND]
        return self

    def wende_an(self, d):
        d = d.copy()
        fl = d.ADEP_mvt.astype(str)
        rueck = fl.map(self.flh).fillna(self.ges).to_numpy()
        for k in SCHLUESSEL:
            idx = pd.MultiIndex.from_arrays([d[c].astype(str) for c in k])
            g = self.tab[k].reindex(idx)
            v = g["median"].to_numpy()
            d[_nam(k)] = np.where(np.isnan(v), rueck, v)
            d[_nam(k) + "_n"] = np.nan_to_num(g["size"].to_numpy())
        return d
