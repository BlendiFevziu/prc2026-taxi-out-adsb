"""Ordered neighbouring-movement sequence model versus the matched flat net, main lane, January/July.

Pre-registered 14.09.2026 (the team's clone, branch claude/research-240-kevin) before any fit.
New information relative to the logged failures (larger flat MLPs, TabM, aggregate queue/arrival
medians): the model reads the ORDERED list of the K nearest movements at the same airport
(arrivals and departures, both directions in time, within +/-W minutes) with per-neighbour
observable fields, and attends over them. Aggregate windows summarise counts and medians; this
keeps who blocked in or taxied where, in order, relative to the departure. Every field exists in
ranking.parquet: neighbour movement time, phase, its own network gap (MVT-AOBT_3 for departures,
ARVT_3-MVT and BLOCK-MVT for arrivals; arrival block times are supplied in the ranking file),
schedule delay, stand/runway/operator/type equality, flight-ID offset and schedule copy flag.
No departure BLOCK_TIME or TAXITIME is read for any departure row. Built separately per source split.

Model and data: the v45 research-cache tables (core_* from research_250/fallback250_cv.py, queue
columns attached exactly as research_250/queue_screen.py), main lane, research categorical encoding,
91 features, target rest = y - ersatz_s, prediction max(ersatz_s + rest_hat, 300).
  LGBM   queue-screen recipe (seed 177, 1400 rounds, early stopping 80 on the fold) as the
         replication reference: research_250 recorded 223.698 s on this split.
  FLAT   neural_model.py recipe (embeddings, BN, 1024-512-256 GELU, dropout .15, 12 epochs,
         batch 8192, AdamW 2e-3 OneCycle, clip 2), seeds 13 and 21.
  SEQ    the same net plus a sequence branch: per-neighbour linear 16->64, two pre-norm
         attention blocks (4 heads, ff 128), a CLS token pooled to 64 dims and concatenated
         to the tabular input. Same seeds, epochs, optimiser and batches. K=24, W=90 min.
Decision rule, fixed now: advance SEQ to a six-pair screen only if, on the January/July main rows,
(1) mean-of-seeds SEQ RMSE beats mean-of-seeds FLAT RMSE by >= 0.3 s, (2) January and July both
improve, (3) the MSE gain survives removing the 100 most favourable rows, and (4) the fixed blend
0.75*LGBM + 0.25*net improves by >= 0.15 s more with SEQ than with FLAT (no weight tuning).
Otherwise the sequence line is closed at this scale. Component screen only; not full ensemble;
no leaderboard feedback; no submission.

Run (repository root, one fit at a time):
  python3 -u -B research_260/neighbour_sequence_screen.py --project . \
     --core-train .cache/rmse260/core_<train>.parquet --core-valid .cache/rmse260/core_<valid>.parquet \
     --output .cache/rmse260/seq-screen
"""
import argparse, gc, json, os, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--core-train", type=Path, required=True)
parser.add_argument("--core-valid", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seeds", default="13,21")
parser.add_argument("--k", type=int, default=24)
parser.add_argument("--window-min", type=float, default=90.0)
parser.add_argument("--jobs", type=int, default=8)
parser.add_argument("--skip-lgbm", action="store_true")
parser.add_argument("--permute-seed", type=int, default=None, help="control: permute neighbour tensors among departures within airport, separately per split")
args = parser.parse_args()
for key in ("project", "core_train", "core_valid", "output"):
    setattr(args, key, getattr(args, key).resolve())
args.output.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(args.output / "matplotlib"))
sys.path.insert(0, str(args.project)); sys.path.insert(0, str(args.project / "research_250"))
os.chdir(args.project)
import neural_model as NM          # torch before tree libraries on macOS
import torch, torch.nn as nn
import main_model as H
import lightgbm as lgb
import numpy as np, pandas as pd
import features_correct_queue as Q
import features_time_units as U

SEEDS = [int(s) for s in args.seeds.split(",")]
K, W = args.k, args.window_min
RAW = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
       "SCHED_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
       "AOBT_3_flt", "ARVT_3_flt"]
NF = 16


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def secs(s):
    return (pd.to_datetime(s, utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()


def table(path):
    core = pd.read_parquet(path)
    source = json.loads(path.with_suffix(".json").read_text())
    raw = pd.concat([pd.read_parquet(args.project / "data" / n, columns=Q.RAW_COLUMNS) for n in source["source_files"]],
                    ignore_index=True)
    fixed = Q.baue(raw, core[Q.DEP_COLUMNS]); del raw
    for col in Q.MERKMALE:
        core[col] = fixed[col].to_numpy()
    del fixed; gc.collect()
    return core.loc[core.hat_aobt.eq(1) & core.y.notna()].copy(), source


def neighbours(source, mvt_ids):
    """(N, K, NF) float32 fields and (N, K) bool mask for the given main-lane departures, from this split's raw files."""
    raw = pd.concat([pd.read_parquet(args.project / "data" / n, columns=RAW) for n in source["source_files"]], ignore_index=True)
    raw = raw.sort_values(["ADEP_mvt", "MVT_TIME_UTC_mvt"], kind="mergesort").reset_index(drop=True)
    t = secs(raw.MVT_TIME_UTC_mvt); is_dep = (raw.PHASE_mvt.to_numpy() == "DEP"); is_arr = (raw.PHASE_mvt.to_numpy() == "ARR")
    aobt = secs(raw.AOBT_3_flt); arvt = secs(raw.ARVT_3_flt); block = secs(raw.BLOCK_TIME_UTC_mvt); sched = secs(raw.SCHED_TIME_UTC_mvt)
    block = np.where(is_arr, block, np.nan)   # departure block times are labels: never read
    nm_taxi = np.where(is_dep & np.isfinite(aobt), np.clip((t - aobt) / 60, -30, 120) / 60, 0.0)
    has_aobt = (is_dep & np.isfinite(aobt)).astype(np.float32)
    taxi_in = np.where(is_arr & np.isfinite(block), np.clip((block - t) / 60, -30, 120) / 60, 0.0)
    nm_arr = np.where(is_arr & np.isfinite(arvt), np.clip((arvt - t) / 60, -60, 60) / 60, 0.0)
    has_arvt = (is_arr & np.isfinite(arvt)).astype(np.float32)
    copied = (is_arr & np.isfinite(block) & np.isfinite(sched) & (np.abs(block - sched) <= 10)).astype(np.float32)
    sdel = np.where(np.isfinite(sched), np.clip((t - sched) / 60, -180, 180) / 180, 0.0)
    fid = raw.FLIGHT_ID_mvt.to_numpy(float)
    stand = raw.STAND_mvt.astype(str).to_numpy(); rwy = raw.RUNWAY_mvt.astype(str).to_numpy()
    oper = raw.AIRCRAFT_OPERATOR_flt.astype(str).to_numpy(); typ = raw.AIRCRAFT_TYPE_mvt.astype(str).to_numpy()
    airport = raw.ADEP_mvt.to_numpy()
    pos = pd.Series(np.arange(len(raw)), index=raw.MVT_ID_mvt.to_numpy())
    own = pos.loc[mvt_ids].to_numpy()
    n = len(own); X = np.zeros((n, K, NF), np.float32); M = np.zeros((n, K), bool)
    offsets = [o for o in range(-K // 2, K // 2 + 1) if o != 0]
    for j, o in enumerate(offsets):
        nb = own + o
        ok = (nb >= 0) & (nb < len(raw))
        nbc = np.clip(nb, 0, len(raw) - 1)
        ok &= airport[nbc] == airport[own]
        dt = (t[nbc] - t[own]) / 60
        ok &= np.abs(dt) <= W
        M[:, j] = ok
        X[:, j, 0] = np.clip(dt, -W, W) / W
        X[:, j, 1] = is_dep[nbc]; X[:, j, 2] = is_arr[nbc]
        X[:, j, 3] = nm_taxi[nbc]; X[:, j, 4] = has_aobt[nbc]
        X[:, j, 5] = taxi_in[nbc]; X[:, j, 6] = nm_arr[nbc]; X[:, j, 7] = has_arvt[nbc]; X[:, j, 8] = copied[nbc]
        X[:, j, 9] = stand[nbc] == stand[own]; X[:, j, 10] = rwy[nbc] == rwy[own]
        X[:, j, 11] = (oper[nbc] == oper[own]) & (oper[own] != "None") & (oper[own] != "nan")
        X[:, j, 12] = sdel[nbc]
        fd = (fid[nbc] - fid[own]) / 1000
        okf = np.isfinite(fd)
        X[:, j, 13] = np.where(okf, np.clip(fd, -10, 10) / 10, 0.0); X[:, j, 14] = okf
        X[:, j, 15] = typ[nbc] == typ[own]
        X[:, j][~ok] = 0.0
    return X, M


class SeqNet(nn.Module):
    def __init__(self, kdim, n_num, use_seq, d=64, heads=4, layers=2, breiten=NM.BREITEN, drop=NM.DROP):
        super().__init__()
        self.use_seq = use_seq
        self.emb = nn.ModuleList([nn.Embedding(n, e) for n, e in kdim])
        ein = sum(e for _, e in kdim) + n_num + (d if use_seq else 0)
        if use_seq:
            self.proj = nn.Sequential(nn.Linear(NF, d), nn.GELU())
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.pos = nn.Parameter(torch.zeros(1, K + 1, d))
            self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.att = nn.ModuleList([nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True) for _ in range(layers)])
            self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, d)) for _ in range(layers)])
            self.lnf = nn.LayerNorm(d)
        lagen = [nn.BatchNorm1d(ein)]
        vor = ein
        for b in breiten:
            lagen += [nn.Linear(vor, b), nn.GELU(), nn.Dropout(drop)]; vor = b
        lagen += [nn.Linear(vor, 1)]
        self.f = nn.Sequential(*lagen)

    def forward(self, xk, xn, xs=None, ms=None):
        parts = [e(xk[:, i]) for i, e in enumerate(self.emb)] + [xn]
        if self.use_seq:
            h = self.proj(xs)
            h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], 1) + self.pos
            pad = torch.cat([torch.zeros(ms.shape[0], 1, dtype=torch.bool, device=ms.device), ~ms], 1)
            for ln1, att, ln2, ff in zip(self.ln1, self.att, self.ln2, self.ff):
                q = ln1(h); a, _ = att(q, q, q, key_padding_mask=pad, need_weights=False); h = h + a
                h = h + ff(ln2(h))
            parts.append(self.lnf(h[:, 0]))
        return self.f(torch.cat(parts, 1)).squeeze(1)


def prepare(lh, merkmale, kat, fit=None):
    num = [m for m in merkmale if m not in kat]
    if fit is None:
        karten, kdim = {}, []
        for c in kat:
            v = lh[c].astype(str); karte = {s: j + 1 for j, s in enumerate(pd.unique(v))}
            karten[c] = karte; n = len(karte) + 1; kdim.append((n, min(64, max(4, int(round(n ** 0.4))))))
        a = lh[num].to_numpy(np.float32)
        med = np.nanmedian(a, 0); iqr = np.nanpercentile(a, 75, axis=0) - np.nanpercentile(a, 25, axis=0)
        iqr[~np.isfinite(iqr) | (iqr < 1e-6)] = 1.0
        spalten_mit = (~np.isfinite(a)).mean(0) > 0
        fit = {"karten": karten, "kdim": kdim, "med": med, "iqr": iqr, "spalten_mit": spalten_mit, "num": num}
    Xk = np.zeros((len(lh), len(kat)), np.int64)
    for i, c in enumerate(kat):
        Xk[:, i] = NM._kat_kodieren(lh[c].astype(str), fit["karten"][c])
    Xn = NM._num_aufbereiten(lh, fit["num"], fit["med"], fit["iqr"], fit["spalten_mit"])
    return Xk, Xn, fit


def train_net(use_seq, seed, Xk, Xn, Xs, Ms, y, kdim, Xk_v, Xn_v, Xs_v, Ms_v):
    dev = NM.geraet(); torch.manual_seed(seed); np.random.seed(seed)
    ym, ys = float(y.mean()), float(y.std()); yl = ((y - ym) / ys).astype(np.float32)
    net = SeqNet(kdim, Xn.shape[1], use_seq).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=NM.LERNRATE, weight_decay=1e-5)
    n = len(yl); B = NM.STAPEL
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=NM.LERNRATE, total_steps=(n // B + 1) * NM.EPOCHEN)
    tk, tn, ty = torch.from_numpy(Xk), torch.from_numpy(Xn), torch.from_numpy(yl)
    ts_, tm = (torch.from_numpy(Xs), torch.from_numpy(Ms)) if use_seq else (None, None)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(NM.EPOCHEN):
        net.train(); idx = torch.randperm(n, generator=g); tot = 0.0
        for b in range(0, n, B):
            j = idx[b:b + B]
            opt.zero_grad()
            out = net(tk[j].to(dev), tn[j].to(dev), ts_[j].to(dev) if use_seq else None, tm[j].to(dev) if use_seq else None)
            v = nn.functional.mse_loss(out, ty[j].to(dev)); v.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sch.step(); tot += float(v) * len(j)
        log(f"   {'SEQ' if use_seq else 'FLAT'} seed {seed} epoch {ep + 1}/{NM.EPOCHEN} train mse(std) {tot / n:.4f} ({time.time() - t0:.0f} s)")
    net.eval(); preds = []
    with torch.no_grad():
        for b in range(0, len(Xk_v), 16384):
            s = slice(b, b + 16384)
            o = net(torch.from_numpy(Xk_v[s]).to(dev), torch.from_numpy(Xn_v[s]).to(dev),
                    torch.from_numpy(Xs_v[s]).to(dev) if use_seq else None, torch.from_numpy(Ms_v[s]).to(dev) if use_seq else None)
            preds.append(o.cpu().numpy())
    return np.concatenate(preds) * ys + ym, time.time() - t0


def scores(y, pred, month, airport, ref_err=None):
    e = pred - y; mse = float(np.mean(e ** 2))
    out = {"rmse": float(np.sqrt(mse)), "mse": mse,
           "by_month": {int(m): float(np.sqrt(np.mean(e[month == m] ** 2))) for m in np.unique(month)},
           "by_airport": {str(a): float(np.sqrt(np.mean(e[airport == a] ** 2))) for a in np.unique(airport)}}
    if ref_err is not None:
        worst = np.abs(ref_err) >= np.quantile(np.abs(ref_err), 0.99)
        out["worst1pct_mse"] = float(np.mean(e[worst] ** 2)); out["rest99_mse"] = float(np.mean(e[~worst] ** 2))
    return out


def gain_without_top100(y, base, cand):
    d = (base - y) ** 2 - (cand - y) ** 2
    keep = np.argsort(-d)[100:]
    return float(np.mean(d)), float(np.mean(d[keep]))


log("load tables")
train, src_tr = table(args.core_train); valid, src_va = table(args.core_valid)
assert train.MVT_ID_mvt.is_unique and valid.MVT_ID_mvt.is_unique and set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))
for col in H.KAT:
    cats = pd.unique(train[col].astype(str))
    train[col] = pd.Categorical(train[col].astype(str), categories=cats)
    valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)
features = H.MERKMALE_HAUPT + Q.MERKMALE
x, xv = train[features].copy(), valid[features].copy()
for col in U.CORRECTED_COLUMNS:
    x[col], xv[col] = train["tu_" + col].to_numpy(), valid["tu_" + col].to_numpy()
y_tr, y_va = train.y.to_numpy(float), valid.y.to_numpy(float)
ers_tr, ers_va = train.ersatz_s.to_numpy(float), valid.ersatz_s.to_numpy(float)
rest_tr = (y_tr - ers_tr).astype(np.float32)
month = pd.to_datetime(valid.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy(); airport = valid.ADEP_mvt.astype(str).to_numpy()
log(f"train {len(train):,} valid {len(valid):,} features {len(features)}")
suffix = "_".join(map(str, sorted(src_va["source_months"])))
result = {"pair": sorted(src_va["source_months"]), "permute_seed": args.permute_seed, "train_rows": int(len(train)), "valid_rows": int(len(valid)), "k": K, "window_min": W,
          "seeds": SEEDS, "recipe": "see docstring", "scores": {}}
preds = pd.DataFrame({"MVT_ID_mvt": valid.MVT_ID_mvt.to_numpy(), "y": y_va, "ersatz_s": ers_va, "month": month, "airport": airport})

if not args.skip_lgbm:
    log("FIT LGBM reference (queue-screen recipe, seed 177)")
    t0 = time.time()
    m = lgb.LGBMRegressor(objective="l2", n_estimators=1400, learning_rate=.03, num_leaves=255, min_child_samples=50,
                          subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, random_state=177, n_jobs=args.jobs)
    m.fit(x, rest_tr, eval_set=[(xv, (y_va - ers_va).astype(np.float32))], callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(300)])
    p_lgb = np.maximum(ers_va + m.predict(xv), 300)
    preds["lgbm"] = p_lgb
    result["scores"]["lgbm"] = scores(y_va, p_lgb, month, airport); result["scores"]["lgbm"]["best_iteration"] = int(m.best_iteration_)
    result["scores"]["lgbm"]["seconds"] = time.time() - t0; result["lgbm_reference_research_250"] = 223.697629244
    log("LGBM", json.dumps({k: result["scores"]["lgbm"][k] for k in ("rmse", "best_iteration", "seconds")}))
    del m; gc.collect()

log("build neighbour tensors")
t0 = time.time()
Xs_tr, Ms_tr = neighbours(src_tr, train.MVT_ID_mvt.to_numpy()); Xs_va, Ms_va = neighbours(src_va, valid.MVT_ID_mvt.to_numpy())
result["neighbour_build_seconds"] = time.time() - t0
result["neighbour_coverage"] = {"train_mean_valid_neighbours": float(Ms_tr.sum(1).mean()), "valid_mean_valid_neighbours": float(Ms_va.sum(1).mean()),
                                "train_rows_without_neighbours": int((Ms_tr.sum(1) == 0).sum()), "valid_rows_without_neighbours": int((Ms_va.sum(1) == 0).sum())}
log("neighbours", json.dumps(result["neighbour_coverage"]))
if args.permute_seed is not None:
    rng = np.random.default_rng(args.permute_seed)
    for Xs_, Ms_, ap in ((Xs_tr, Ms_tr, train.ADEP_mvt.astype(str).to_numpy()), (Xs_va, Ms_va, valid.ADEP_mvt.astype(str).to_numpy())):
        for a in np.unique(ap):
            idx = np.flatnonzero(ap == a); perm = rng.permutation(idx)
            Xs_[idx] = Xs_[perm]; Ms_[idx] = Ms_[perm]
    log("PERMUTATION CONTROL: neighbour tensors shuffled within airport, seed", args.permute_seed)
Xk, Xn, fit = prepare(x, features, H.KAT); Xk_v, Xn_v, _ = prepare(xv, features, H.KAT, fit)
del x, xv, train; valid = valid[["MVT_ID_mvt", "y", "ersatz_s", "MVT_TIME_UTC_mvt", "ADEP_mvt"]]; gc.collect()  # free the wide frames before training

for use_seq in (False, True):
    name = "seq" if use_seq else "flat"
    for seed in SEEDS:
        log(f"FIT {name} seed {seed}")
        r, secs_ = train_net(use_seq, seed, Xk, Xn, Xs_tr, Ms_tr, rest_tr, fit["kdim"], Xk_v, Xn_v, Xs_va, Ms_va)
        p = np.maximum(ers_va + r, 300); preds[f"{name}_s{seed}"] = p
        result["scores"][f"{name}_s{seed}"] = scores(y_va, p, month, airport); result["scores"][f"{name}_s{seed}"]["seconds"] = secs_
        log(name, seed, json.dumps({k: result["scores"][f"{name}_s{seed}"][k] for k in ("rmse", "by_month")}))
        preds.to_parquet(args.output / f"seq_screen_predictions_{suffix}.parquet", index=False)
        (args.output / f"seq_screen_result_{suffix}.json").write_text(json.dumps(result, indent=2))
    preds[f"{name}_mean"] = np.maximum(ers_va + np.mean([preds[f"{name}_s{s}"] - ers_va for s in SEEDS], 0), 300)

flat_err = preds["flat_mean"] - y_va
for name in ("flat", "seq"):
    result["scores"][f"{name}_mean"] = scores(y_va, preds[f"{name}_mean"].to_numpy(), month, airport, flat_err)
    if "lgbm" in preds:
        b = np.maximum(0.75 * (preds["lgbm"] - ers_va) + 0.25 * (preds[f"{name}_mean"] - ers_va) + ers_va, 300)
        preds[f"blend_{name}"] = b; result["scores"][f"blend_{name}"] = scores(y_va, b.to_numpy(), month, airport, flat_err)
g_all, g_100 = gain_without_top100(y_va, preds["flat_mean"].to_numpy(), preds["seq_mean"].to_numpy())
dec = {"seq_minus_flat_rmse_gain": result["scores"]["flat_mean"]["rmse"] - result["scores"]["seq_mean"]["rmse"],
       "mse_gain": g_all, "mse_gain_without_top100": g_100,
       "january_gain": result["scores"]["flat_mean"]["by_month"].get(1, np.nan) - result["scores"]["seq_mean"]["by_month"].get(1, np.nan),
       "july_gain": result["scores"]["flat_mean"]["by_month"].get(7, np.nan) - result["scores"]["seq_mean"]["by_month"].get(7, np.nan)}
if "lgbm" in preds:
    dec["blend_gain_flat"] = result["scores"]["lgbm"]["rmse"] - result["scores"]["blend_flat"]["rmse"]
    dec["blend_gain_seq"] = result["scores"]["lgbm"]["rmse"] - result["scores"]["blend_seq"]["rmse"]
    dec["blend_extra_with_seq"] = dec["blend_gain_seq"] - dec["blend_gain_flat"]
dec["advance"] = bool(dec["seq_minus_flat_rmse_gain"] >= 0.3 and dec["january_gain"] > 0 and dec["july_gain"] > 0 and g_100 > 0
                      and dec.get("blend_extra_with_seq", 0) >= 0.15)
result["decision"] = dec
preds.to_parquet(args.output / f"seq_screen_predictions_{suffix}.parquet", index=False)
(args.output / f"seq_screen_result_{suffix}.json").write_text(json.dumps(result, indent=2))
log("DECISION", json.dumps(dec))
