"""Are the fallback path's three models under-fitted, as the NM add-on's residual model was?

Today's three gains all came from constants nobody had tuned: ANTEIL_NETZ, then the add-on's shrink, then the
add-on's model size. The fallback path has never been examined the same way. It fits three models on 17,097 rows:

    normal      LGBMRegressor(400 rounds, lr .05, 31 leaves, min_child 20)   serves the 92.8 % below threshold
    height      LGBMRegressor(300 rounds, lr .05, 15 leaves, min_child 10)   blended 0.3 against the schedule delay
    classifier  LGBMClassifier(600 rounds, lr .05, 31 leaves, min_child 20)  supplies p, raised to HOCH

A prior observation points at the height model: `fallback_weights_rederive.py` found ANTEIL_HM wants to rise from
0.30 to 0.45, which is what an under-fitted, under-weighted height model would look like.

Components are refitted per configuration on the ten learning months and scored on the untouched January/July
holdout. One model is varied at a time so the attribution is clean.

Rules, fixed before the run, the same four the fallback lane has used all week:
  1. the configuration beats the current one pooled
  2. BOTH January and July improve
  3. the gain survives removing its 100 most favourable rows
  4. ordinary rows (label <= 5,400 s) are not made worse
The fallback lane concentrates its error in about 150 rows, so rule 3 is demanding here; v46's neighbour-taxi
anchor nonetheless passed it in this lane, so it is achievable and a failure is informative.

  python3 -u -B research_280/fallback_capacity.py
"""
import json
import time
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import main_model as H
import features_stand_rate as MSQ
import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = PROJECT / ".cache/clock_queue_release_20260912/data"
OUTPUT = PROJECT / "research_280/results/fallback_capacity.json"
ORDINARY = 5400.0
CUR = {"normal": dict(n_estimators=400, num_leaves=31, min_child_samples=20),
       "height": dict(n_estimators=300, num_leaves=15, min_child_samples=10),
       "clf": dict(n_estimators=600, num_leaves=31, min_child_samples=20)}
VARIANTS = {
    "normal": [dict(n_estimators=1200, num_leaves=31, min_child_samples=20),
               dict(n_estimators=1200, num_leaves=63, min_child_samples=20),
               dict(n_estimators=2500, num_leaves=63, min_child_samples=10)],
    "height": [dict(n_estimators=900, num_leaves=15, min_child_samples=10),
               dict(n_estimators=900, num_leaves=31, min_child_samples=10),
               dict(n_estimators=2000, num_leaves=31, min_child_samples=5)],
    "clf":    [dict(n_estimators=1500, num_leaves=31, min_child_samples=20),
               dict(n_estimators=1500, num_leaves=63, min_child_samples=10)],
}


def log(*p):
    print(time.strftime("%H:%M:%S"), *p, flush=True)


def main():
    columns = sorted(set(H.MERKMALE_RUECK) - set(MSQ.MERKMALE)
                     | {"y", "hat_aobt", "versp_min", "ADEP_mvt", "STAND_mvt", "monat"})
    learn = pd.read_parquet(DATA / "months_2_3_4_5_6_8_9_10_11_12.parquet", columns=columns)
    hold = pd.read_parquet(DATA / "months_1_7.parquet", columns=columns)
    for f in (learn, hold):
        f["hat_aobt"] = f.hat_aobt.astype(int)
        f["z"] = (f.y > H.SCHWELLE_EXTREM).astype(int)
    learn = learn.loc[learn.hat_aobt.eq(0)].copy()
    mapping = MSQ.passe_an(learn)
    learn = MSQ.ausser_faltung(learn)
    hold = MSQ.wende_an(mapping, hold)
    hold = hold.loc[hold.hat_aobt.eq(0)].copy()
    B = list(H.MERKMALE_RUECK)
    y = hold.y.to_numpy(float)
    month = hold.monat.to_numpy()
    jan, jul, ordinary = month == 1, month == 7, y <= ORDINARY
    yl = learn.y.to_numpy(float)
    extreme = learn.z.eq(1).to_numpy()
    schedule = hold.versp_min.to_numpy(float) * 60
    log(f"learn {len(learn):,}, hold {len(hold):,} (Jan {jan.sum():,}, Jul {jul.sum():,}, ordinary {ordinary.sum():,})")

    def reg(cfg):
        return lgb.LGBMRegressor(objective="l2", learning_rate=.05, verbose=-1, n_jobs=-1, random_state=0, **cfg)

    def clf(cfg):
        return lgb.LGBMClassifier(objective="binary", learning_rate=.05, verbose=-1, n_jobs=-1, random_state=0, **cfg)

    cache = {}

    def build(which, cfg):
        key = (which, tuple(sorted(cfg.items())))
        if key in cache:
            return cache[key]
        if which == "clf":
            v = clf(cfg).fit(learn[B], learn.z).predict_proba(hold[B])[:, 1]
        elif which == "height":
            v = reg(cfg).fit(learn.loc[extreme, B], yl[extreme]).predict(hold[B])
        else:
            v = reg(cfg).fit(learn.loc[~extreme, B], yl[~extreme]).predict(hold[B])
        cache[key] = v
        return v

    def predict(cfgs):
        p = build("clf", cfgs["clf"]) ** H.HOCH
        high = (1 - H.ANTEIL_HM) * schedule + H.ANTEIL_HM * build("height", cfgs["height"])
        return np.maximum((1 - p) * build("normal", cfgs["normal"]) + p * high, 0)

    base = predict(CUR)
    base_sq = (base - y) ** 2
    log(f"current configuration reproduces fallback RMSE {np.sqrt(base_sq.mean()):.2f}")
    rows = []
    for which, options in VARIANTS.items():
        for cfg in options:
            started = time.time()
            sq = (predict({**CUR, which: cfg}) - y) ** 2
            g = base_sq - sq
            keep = np.ones(len(g), bool)
            keep[np.argsort(g)[::-1][:100]] = False
            rows.append({"model": which, "config": json.dumps(cfg), "rmse": float(np.sqrt(sq.mean())),
                         "gain": float(g.mean()), "gain_no_top100": float(g[keep].mean()),
                         "january_gain": float(g[jan].mean()), "july_gain": float(g[jul].mean()),
                         "ordinary_gain": float(g[ordinary].mean()), "seconds": time.time() - started})
            r = rows[-1]
            log(f"{which:7s} {json.dumps(cfg):58s} rmse {r['rmse']:8.1f}  gain {r['gain']:>9,.0f}  "
                f"no-top100 {r['gain_no_top100']:>9,.0f}  Jan {r['january_gain']:>8,.0f}  Jul {r['july_gain']:>8,.0f}")
    t = pd.DataFrame(rows)
    ok = t.loc[(t.gain > 0) & (t.january_gain > 0) & (t.july_gain > 0) & (t.gain_no_top100 > 0) & (t.ordinary_gain >= 0)]
    res = {"current_rmse": float(np.sqrt(base_sq.mean())), "variants": rows, "passing_all_four": ok.to_dict("records")}
    print()
    if len(ok):
        best = ok.sort_values("gain", ascending=False).iloc[0]
        V = 265.2839
        per = best.gain * 5290 / 344841
        print(f"BEST passing all four: {best.model} {best.config} -> fallback RMSE {best.rmse:.2f} "
              f"(from {np.sqrt(base_sq.mean()):.2f})")
        print(f"  mechanical official: {V:.4f} -> {np.sqrt(V**2-per):.4f} ({V-np.sqrt(V**2-per):+.3f} s), NOT a forecast")
        res["best"] = best.to_dict()
    else:
        print("No configuration passes all four rules: the fallback models stand as they are.")
    OUTPUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
