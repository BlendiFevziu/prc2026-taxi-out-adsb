"""Import-safe, saveable version of the team's paired FLAT/SEQ network recipe.

The original screen remains unchanged as the historical reference. This module
preserves architecture, initialization, row shuffling and training hyperparameters.
Only the neighbour grouping is an experimental variable. Predict returns RAW
residuals; apply the reporting gate and 300-second floor once, after blending.
"""
import time
import neural_model as NM  # torch before all tree libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from research_260.sequence_features import FIELDS


class SeqNet(nn.Module):
    def __init__(self, kdim, n_num, use_seq, k=24, d=64, heads=4, layers=2,
                 breiten=NM.BREITEN, drop=NM.DROP):
        super().__init__()
        self.use_seq = use_seq
        self.emb = nn.ModuleList([nn.Embedding(n, e) for n, e in kdim])
        ein = sum(e for _, e in kdim) + n_num + (d if use_seq else 0)
        if use_seq:
            self.proj = nn.Sequential(nn.Linear(len(FIELDS), d), nn.GELU())
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.pos = nn.Parameter(torch.zeros(1, k+1, d))
            self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.att = nn.ModuleList([nn.MultiheadAttention(d, heads, dropout=.1, batch_first=True) for _ in range(layers)])
            self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, d)) for _ in range(layers)])
            self.lnf = nn.LayerNorm(d)
        lagen = [nn.BatchNorm1d(ein)]
        vor = ein
        for b in breiten:
            lagen += [nn.Linear(vor, b), nn.GELU(), nn.Dropout(drop)]
            vor = b
        lagen += [nn.Linear(vor, 1)]
        self.f = nn.Sequential(*lagen)

    def forward(self, xk, xn, xs=None, ms=None):
        parts = [e(xk[:, i]) for i, e in enumerate(self.emb)] + [xn]
        if self.use_seq:
            h = self.proj(xs)
            h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], 1) + self.pos
            pad = torch.cat([torch.zeros(ms.shape[0], 1, dtype=torch.bool, device=ms.device), ~ms], 1)
            for ln1, att, ln2, ff in zip(self.ln1, self.att, self.ln2, self.ff):
                q = ln1(h)
                a, _ = att(q, q, q, key_padding_mask=pad, need_weights=False)
                h = h+a
                h = h+ff(ln2(h))
            parts.append(self.lnf(h[:, 0]))
        return self.f(torch.cat(parts, 1)).squeeze(1)


def prepare(frame, features, kat, fit=None):
    num = [m for m in features if m not in kat]
    if fit is None:
        karten, kdim = {}, []
        for col in kat:
            mapping = {s: j+1 for j, s in enumerate(pd.unique(frame[col].astype(str)))}
            karten[col] = mapping
            n = len(mapping)+1
            kdim.append((n, min(64, max(4, int(round(n**.4))))))
        a = frame[num].to_numpy(np.float32)
        med = np.nanmedian(a, 0)
        iqr = np.nanpercentile(a, 75, axis=0)-np.nanpercentile(a, 25, axis=0)
        iqr[~np.isfinite(iqr) | (iqr < 1e-6)] = 1.
        fit = dict(karten=karten, kdim=kdim, med=med, iqr=iqr,
                   spalten_mit=(~np.isfinite(a)).mean(0) > 0, num=num, kat=list(kat), features=list(features))
    Xk = np.zeros((len(frame), len(kat)), np.int64)
    for i, col in enumerate(kat):
        Xk[:, i] = NM._kat_kodieren(frame[col].astype(str), fit['karten'][col])
    Xn = NM._num_aufbereiten(frame, fit['num'], fit['med'], fit['iqr'], fit['spalten_mit'])
    if not np.isfinite(Xn).all():
        raise ValueError("nonfinite tabular features")
    return Xk, Xn, fit


def fit_net(use_seq, seed, Xk, Xn, Xs, Ms, y, kdim, log=print,
            epochs=NM.EPOCHEN, batch_size=NM.STAPEL, device=None):
    dev = device or NM.geraet()
    torch.manual_seed(seed)
    np.random.seed(seed)
    ym, ys = float(y.mean()), float(y.std())
    if not np.isfinite(y).all() or ys <= 0:
        raise ValueError("invalid residual target")
    yl = ((y-ym)/ys).astype(np.float32)
    k = Xs.shape[1] if use_seq else 24
    net = SeqNet(kdim, Xn.shape[1], use_seq, k=k).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=NM.LERNRATE, weight_decay=1e-5)
    n, B = len(yl), batch_size
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=NM.LERNRATE, total_steps=(n//B+1)*epochs)
    tk, tn, ty = torch.from_numpy(Xk), torch.from_numpy(Xn), torch.from_numpy(yl)
    ts, tm = (torch.from_numpy(Xs), torch.from_numpy(Ms)) if use_seq else (None, None)
    generator = torch.Generator().manual_seed(seed)
    started = time.time()
    for ep in range(epochs):
        net.train()
        idx, total = torch.randperm(n, generator=generator), 0.
        for b in range(0, n, B):
            j = idx[b:b+B]
            opt.zero_grad()
            output = net(tk[j].to(dev), tn[j].to(dev), ts[j].to(dev) if use_seq else None,
                         tm[j].to(dev) if use_seq else None)
            loss = nn.functional.mse_loss(output, ty[j].to(dev))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            sch.step()
            total += float(loss.detach())*len(j)
        log(f'epoch {ep+1}/{epochs}, standardized training MSE {total/n:.6f}, {time.time()-started:.0f}s')
    return dict(state={key: value.detach().cpu() for key, value in net.state_dict().items()},
                kdim=kdim, n_num=Xn.shape[1], use_seq=use_seq, k=k, ym=ym, ys=ys,
                seed=seed, epochs=epochs, seconds=time.time()-started)


def predict(package, Xk, Xn, Xs=None, Ms=None, batch_size=4096, device=None):
    dev = device or NM.geraet()
    net = SeqNet(package['kdim'], package['n_num'], package['use_seq'], k=package['k'])
    net.load_state_dict(package['state'])
    net.to(dev).eval()
    predictions = []
    with torch.no_grad():
        for b in range(0, len(Xk), batch_size):
            sl = slice(b, b+batch_size)
            output = net(torch.from_numpy(Xk[sl]).to(dev), torch.from_numpy(Xn[sl]).to(dev),
                         torch.from_numpy(Xs[sl]).to(dev) if package['use_seq'] else None,
                         torch.from_numpy(Ms[sl]).to(dev) if package['use_seq'] else None)
            predictions.append(output.cpu().numpy())
    result = np.concatenate(predictions)*package['ys']+package['ym']
    if not np.isfinite(result).all():
        raise ValueError("nonfinite prediction")
    return result
