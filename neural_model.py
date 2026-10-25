"""neural_model.py - Ein neuronales Netz als dritte Modellfamilie fuer die Hauptgruppe.

Warum es ueberhaupt dabei ist: die Merkmale sind ausgereizt. Am 06.09.2026 wurden fuenf
Merkmalsfamilien gemessen und verworfen, am 07.09. zwei weitere. Was bleibt, ist die
Modellklasse. LightGBM und CatBoost sind beide Baumverfahren, ihre Restfehler haengen mit
0,9515 zusammen; das Netz liegt bei 0,9332 gegen LightGBM und 0,9272 gegen CatBoost, irrt
sich also verschieden genug, um zu tragen.

Gemessen auf der Halteprobe (Januar und Juli 2025, 339.046 Zeilen der Hauptgruppe):

    LightGBM allein                    229,69 s
    CatBoost allein                    228,16 s
    Netz allein, fuenf Saaten          233,31 s
    Gemisch 0,40 / 0,60 / 0    (bisher) 226,08 s
    Gemisch 0,30 / 0,45 / 0,25 (neu)    225,28 s

TabM wurde daneben geprueft und VERWORFEN: allein 236,22 s bei 4.208 s Rechenzeit, und im
Vierergemisch nur 0,16 s besser als ohne.

Bauform: die elf kategorialen Merkmale bekommen Einbettungen, die numerischen werden
robust standardisiert und bei plus/minus zehn gekappt, Fehlwerte auf null plus eigener
Anzeiger. Drei Schichten 1024-512-256. Vier Bauformen wurden am 07.09. bei gleicher
Epochenzahl verglichen, diese war die beste: schmal 238,22, jetzt 237,74, breit 239,48,
tief 238,43.

Zwoelf Epochen. Mehr schadet, das ist gemessen: 30 Epochen geben 243,21 gegen 237,72 bei
zwoelf. Das Netz ist nicht untertrainiert, es ueberpasst.

Gerechnet wird auf der Grafikeinheit (MPS auf dem M3 Pro), sonst auf der CPU.
Rund 80 Sekunden je Saat auf 1,72 Millionen Zeilen.

FALLE, gefunden am 07.09.2026 durch zwei reproduzierbare Abstuerze. torch und LightGBM
bringen je eine eigene OpenMP-Laufzeit mit. Wird libomp zuerst von LightGBM oder CatBoost
geladen und danach von torch, faellt der Prozess beim ersten Lernschritt mit
Segmentation fault 11 in `__kmp_fork_barrier` um, ohne jede Fehlermeldung und mit
Rueckgabewert 0, wenn eine Pipe dahinter haengt. **Dieses Modul muss deshalb VOR
lightgbm und catboost eingefuehrt werden.** main_model.py und submission.py tun das,
Absturzbericht Python-2026-09-07-152750.ips.

Aufruf: von main_model.py (lerne) und submission.py (sage_voraus).
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BREITEN = [1024, 512, 256]
DROP = 0.15
EPOCHEN = 12
STAPEL = 8192
LERNRATE = 2e-3
# Fuenf Saaten. Die Halteprobe zeigt den Nutzen des Mittelns deutlich: zwei Saaten
# 235,05 s, fuenf Saaten 233,31 s.
SAATEN = (13, 21, 29, 37, 45)


def geraet():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Netz(nn.Module):
    def __init__(self, kdim, n_num, breiten=BREITEN, drop=DROP):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n, d) for n, d in kdim])
        ein = sum(d for _, d in kdim) + n_num
        lagen = [nn.BatchNorm1d(ein)]
        vor = ein
        for b in breiten:
            lagen += [nn.Linear(vor, b), nn.GELU(), nn.Dropout(drop)]
            vor = b
        lagen += [nn.Linear(vor, 1)]
        self.f = nn.Sequential(*lagen)

    def forward(self, xk, xn):
        e = [m(xk[:, i]) for i, m in enumerate(self.emb)]
        return self.f(torch.cat(e + [xn], 1)).squeeze(1)


def _kat_kodieren(sp, karte):
    """Unbekannte Auspraegungen fallen auf 0, die dafuer freigehaltene Zeile."""
    return sp.astype(str).map(karte).fillna(0).to_numpy().astype(np.int64)


def _num_aufbereiten(x, num, med, iqr, spalten_mit):
    a = x[num].to_numpy(np.float32)
    fehlt = (~np.isfinite(a)).astype(np.float32)
    a = np.clip((np.nan_to_num(a, nan=0.0) - med) / iqr, -10, 10)
    return np.hstack([a, fehlt[:, spalten_mit]]).astype(np.float32)


def lerne(lh, merkmale, kat, ziel="rest", saaten=SAATEN, epochen=EPOCHEN, laut=True):
    """Lernt das Netz auf der Hauptgruppe und gibt ein Paket fuer sage_voraus zurueck.

    lh        Tabelle der Hauptgruppe mit der Zielspalte
    merkmale  vollstaendige Merkmalsliste, in der Reihenfolge des Hauptmodells
    kat       Teilmenge davon, die kategorial behandelt wird
    """
    t0 = time.time()
    dev = geraet()
    num = [m for m in merkmale if m not in kat]

    karten, kdim = {}, []
    Xk = np.zeros((len(lh), len(kat)), dtype=np.int64)
    for i, c in enumerate(kat):
        v = lh[c].astype(str)
        karte = {s: j + 1 for j, s in enumerate(pd.unique(v))}
        karten[c] = karte
        Xk[:, i] = _kat_kodieren(v, karte)
        n = len(karte) + 1
        kdim.append((n, min(64, max(4, int(round(n ** 0.4))))))

    a = lh[num].to_numpy(np.float32)
    med = np.nanmedian(a, 0)
    iqr = np.nanpercentile(a, 75, axis=0) - np.nanpercentile(a, 25, axis=0)
    iqr[~np.isfinite(iqr) | (iqr < 1e-6)] = 1.0
    spalten_mit = (~np.isfinite(a)).mean(0) > 0
    Xn = _num_aufbereiten(lh, num, med, iqr, spalten_mit)

    y = lh[ziel].to_numpy(np.float32)
    ym, ys = float(y.mean()), float(y.std())
    yl = ((y - ym) / ys).astype(np.float32)

    if laut:
        print(f"  Netz auf {dev}: {len(lh):,} Zeilen, {len(kat)} kategorial "
              f"({sum(d for _, d in kdim)} Einbettungsspalten), "
              f"{Xn.shape[1]} numerisch", flush=True)

    tk = torch.from_numpy(Xk)
    tn = torch.from_numpy(Xn)
    ty = torch.from_numpy(yl)
    n = len(yl)
    zustaende = []
    for saat in saaten:
        torch.manual_seed(saat)
        net = Netz(kdim, Xn.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=LERNRATE, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LERNRATE, total_steps=(n // STAPEL + 1) * epochen)
        for _ in range(epochen):
            net.train()
            idx = torch.randperm(n)
            for b in range(0, n, STAPEL):
                j = idx[b:b + STAPEL]
                opt.zero_grad()
                v = nn.functional.mse_loss(
                    net(tk[j].to(dev), tn[j].to(dev)), ty[j].to(dev))
                v.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
                sch.step()
        zustaende.append({k: t.cpu() for k, t in net.state_dict().items()})
        if laut:
            print(f"    Saat {saat} fertig ({time.time() - t0:.0f} s)", flush=True)

    return {"zustaende": zustaende, "kdim": kdim, "karten": karten,
            "kat": list(kat), "num": num, "med": med, "iqr": iqr,
            "spalten_mit": spalten_mit, "ym": ym, "ys": ys,
            "n_num": Xn.shape[1], "breiten": list(BREITEN), "drop": DROP}


def sage_voraus(paket, x, stapel=16384):
    """Mittelt die Vorhersagen aller gelernten Saaten. Gibt den Rest zurueck, nicht y."""
    dev = geraet()
    kat, num = paket["kat"], paket["num"]
    Xk = np.zeros((len(x), len(kat)), dtype=np.int64)
    for i, c in enumerate(kat):
        Xk[:, i] = _kat_kodieren(x[c], paket["karten"][c])
    Xn = _num_aufbereiten(x, num, paket["med"], paket["iqr"], paket["spalten_mit"])

    aus = []
    for zustand in paket["zustaende"]:
        net = Netz(paket["kdim"], paket["n_num"],
                   paket.get("breiten", BREITEN), paket.get("drop", DROP)).to(dev)
        net.load_state_dict(zustand)
        net.eval()
        teile = []
        with torch.no_grad():
            for b in range(0, len(Xn), stapel):
                teile.append(net(torch.from_numpy(Xk[b:b + stapel]).to(dev),
                                 torch.from_numpy(Xn[b:b + stapel]).to(dev)).cpu().numpy())
        aus.append(np.concatenate(teile) * paket["ys"] + paket["ym"])
    return np.mean(aus, axis=0)
