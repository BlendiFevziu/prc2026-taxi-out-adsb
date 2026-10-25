"""Count matching for Rome's copy rows in the fallback lane, six pairs (research_420_fallback/PREREG_rome_count.md).

Stage A refits each pair's production fallback exactly as research_240/fallback_anchor_production.py does (production
encoding; its research-encoding and permutation arms are not needed) and must reproduce the saved prod_anchor; it saves
the classifier probability. Stage B applies the count rule and judges it. No production change, no upload.
Run from the repository root:  python3 -u -B research_420_fallback/rome_count.py
"""
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main_model as H  # noqa: E402  torch before tree libraries on macOS
import features_stand_rate as MSQ  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from research_240 import features_neighbour_taxi as NT  # noqa: E402
from research_300.heavy_lock import heavy_work  # noqa: E402

SCREEN = ROOT / ".cache/research_240/fallback_anchor"
SAVED = ROOT / ".cache/research_240/fallback_anchor_production"
CACHE = ROOT / ".cache/research_420/rome_count"
OUT = ROOT / "research_420_fallback/results/rome_count.json"
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
BASE, WITH_NT = list(H.MERKMALE_RUECK), list(H.MERKMALE_RUECK) + NT.MERKMALE
JOBS, H2, N_FB, N, REF = 4, 7200.0, 5290, 344841, 258.0008


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def tables():
    found = {}
    for side in sorted(SCREEN.glob("fallback_nt_*.json")):
        found[tuple(sorted(json.loads(side.read_text())["source_months"]))] = side.with_suffix(".parquet")
    return found


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05, num_leaves=leaves,
                             min_child_samples=minimum, verbose=-1, n_jobs=JOBS, random_state=0)


def stage_a(key, found):
    """Production fallback of one pair (as fallback_anchor_production.parts/anchored), plus p."""
    path = CACHE / f"parts_{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    a, b = map(int, key.split("_"))
    train = pd.read_parquet(found[tuple(sorted(set(range(1, 13)) - {a, b}))]).reset_index(drop=True)
    valid = pd.read_parquet(found[(a, b)]).reset_index(drop=True)
    train["z"] = (train.y > H.SCHWELLE_EXTREM).astype(int)
    mapping = MSQ.passe_an(train)
    tr, va = MSQ.ausser_faltung(train), MSQ.wende_an(mapping, valid)
    z0, y = tr.z.eq(0).to_numpy(), tr.y.to_numpy(float)
    with heavy_work(f"research_420 rome count: fallback refit {key}", log=log):
        p = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31, min_child_samples=20,
                               verbose=-1, n_jobs=JOBS, random_state=0).fit(tr[BASE], tr.z).predict_proba(va[BASE])[:, 1]
        q = p ** H.HOCH
        height = ((1 - H.ANTEIL_HM) * va.versp_min.to_numpy(float) * 60
                  + H.ANTEIL_HM * reg(300, 15, 10).fit(tr.loc[~z0, BASE], y[~z0]).predict(va[BASE]))
        normal = reg(400, 31, 20).fit(tr.loc[z0, BASE], y[z0]).predict(va[BASE])
        a_tr, a_va = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
        rows = z0 & np.isfinite(a_tr)
        offset = reg(400, 31, 20).fit(tr.loc[rows, WITH_NT], y[rows] - a_tr[rows])
        has = np.isfinite(a_va)
        out = normal.copy()
        out[has] = a_va[has] + offset.predict(va.loc[has, WITH_NT])
        pred = np.maximum((1 - q) * out + q * height, 0)
    # learning-month copy rate among late LIRF rows (labels of the learning months only)
    lt = tr[tr.ADEP_mvt.astype(str).eq("LIRF")]
    versp_t = lt.versp_min.to_numpy(float) * 60
    late_t = versp_t > H2
    rate = float((np.abs(lt.y.to_numpy(float) - versp_t) <= 60)[late_t].mean())
    frame = pd.DataFrame({"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "airport": va.ADEP_mvt.astype(str).to_numpy(),
                          "month": pd.to_datetime(va.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy(),
                          "y": va.y.to_numpy(float), "versp": va.versp_min.to_numpy(float) * 60, "p": p, "pred": pred,
                          "rate": rate, "fold": key})
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    del train, valid, tr, va
    gc.collect()
    return frame


def count_rule(f, order=None, seed=None):
    """Top-k late LIRF rows per month by p (or at random with seed) take versp; the rest keep pred."""
    new = f.pred.to_numpy(float).copy()
    rng = np.random.default_rng(seed) if seed is not None else None
    for m in sorted(f.month.unique()):
        idx = np.flatnonzero(f.airport.eq("LIRF").to_numpy() & f.month.eq(m).to_numpy() & (f.versp.to_numpy() > H2))
        if not len(idx):
            continue
        k = int(round(f.rate.iloc[0] * len(idx)))
        ranked = idx[np.argsort(-f.p.to_numpy()[idx], kind="stable")] if rng is None else rng.permutation(idx)
        new[ranked[:k]] = f.versp.to_numpy()[ranked[:k]]
    return new


def main():
    found, frames, replication = tables(), [], {}
    for a, b in PAIRS:
        key = f"{a}_{b}"
        f = stage_a(key, found)
        saved = pd.read_parquet(SAVED / f"predictions_{key}.parquet", columns=["MVT_ID_mvt", "prod_anchor", "y"])
        s = saved.set_index("MVT_ID_mvt").loc[f.MVT_ID_mvt]
        r_new, r_old = np.sqrt(np.mean((f.pred - f.y) ** 2)), np.sqrt(np.mean((s.prod_anchor.to_numpy() - f.y.to_numpy()) ** 2))
        replication[key] = {"rmse_refit": float(r_new), "rmse_saved": float(r_old), "max_abs_diff_s": float(np.max(np.abs(f.pred.to_numpy() - s.prod_anchor.to_numpy())))}
        log(key, json.dumps(replication[key]))
        if abs(r_new - r_old) > 0.01:
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps({"replication": replication, "verdict": "gate_failed"}, indent=2) + "\n")
            raise SystemExit(f"replication gate failed on {key}")
        f["pred"] = s.prod_anchor.to_numpy(float)                 # judge against the saved shipped values themselves
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    y = d.y.to_numpy(float)
    base_err = (d.pred.to_numpy() - y) ** 2
    real = np.concatenate([count_rule(g) for _, g in d.groupby("fold", sort=False)])
    d = pd.concat([g for _, g in d.groupby("fold", sort=False)], ignore_index=True)
    y, base_err = d.y.to_numpy(float), (d.pred.to_numpy() - d.y.to_numpy()) ** 2
    gain = base_err - (real - y) ** 2
    ctrl = [base_err - (np.concatenate([count_rule(g, seed=s) for _, g in d.groupby("fold", sort=False)]) - y) ** 2 for s in (1, 2, 3, 4, 5)]
    changed = real != d.pred.to_numpy()
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-20:]] = False
    pairs = {k: float(gain[d.fold.eq(k).to_numpy()].mean()) for k in d.fold.unique()}
    per_row = float(gain.mean())
    mse = per_row * N_FB / N
    official = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    copies = np.abs(y - d.versp.to_numpy()) <= 60
    rules = {"1_pooled_positive": per_row > 0, "2_without_best_20_positive": float(gain[keep].mean()) > 0,
             "3_four_of_six_pairs": sum(v > 0 for v in pairs.values()) >= 4,
             "4_beats_mean_random_control": per_row > float(np.mean([c.mean() for c in ctrl])),
             "5_official_scale_at_least_0_1_s": official >= 0.1}
    rep = {"prereg": "research_420_fallback/PREREG_rome_count.md", "replication": replication, "rows": int(len(d)),
           "rmse_shipped": float(np.sqrt(base_err.mean())), "rmse_rule": float(np.sqrt(((real - y) ** 2).mean())),
           "gain_per_fallback_row": per_row, "without_best_20": float(gain[keep].mean()), "pairs": pairs,
           "controls": [float(c.mean()) for c in ctrl], "changed_rows": int(changed.sum()),
           "changed_rows_that_are_copies": int((changed & copies).sum()),
           "learning_rates": {k: float(d.rate[d.fold.eq(k)].iloc[0]) for k in d.fold.unique()},
           "by_month": {int(m): {"changed": int((changed & d.month.eq(m).to_numpy()).sum()), "gain_sum": float(gain[d.month.eq(m).to_numpy()].sum())}
                        for m in sorted(d.month.unique())},
           "official_scale_s": official, "rules": rules}
    rep["verdict"] = "pass" if all(rules.values()) else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
