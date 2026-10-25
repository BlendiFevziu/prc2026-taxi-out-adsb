"""Train a covariate-shifted copy of only the no-AOBT fallback models.

The ranking file exposes the schedule/movement covariates for the scored months,
but not their targets.  The fallback group is small and its airport/delay/hour
mix moves substantially between months.  We therefore reweight 2025 training
rows to the covariate distribution in ``ranking.parquet`` and leave the current
main-group models untouched.  The baseline ``models.pkl`` is never overwritten;
the result is written to ``models_shifted.pkl`` and can be selected with
``PRC_MODELLE=models_shifted.pkl`` when generating a local submission.

For the arrival-traffic candidate, set
``PRC_FALLBACK_ARRIVALS=1 PRC_FALLBACK_SHIFT=0 PRC_MODEL_OUT=models_arrivals.pkl``.
"""
import os
import joblib
import neural_model  # Load torch before tree libraries (macOS OpenMP).
import lightgbm as lgb
import numpy as np
import pandas as pd

import framework as G
import main_model as H
import features_geometry as MG
import features_stand_rate as MSQ


BASE = "models.pkl"
OUT = os.environ.get("PRC_MODEL_OUT", "models_shifted.pkl")
RANKING = "data/ranking.parquet"
USE_ARRIVALS = os.environ.get("PRC_FALLBACK_ARRIVALS", "0") == "1"
USE_SHIFT = os.environ.get("PRC_FALLBACK_SHIFT", "1") != "0"


def _bins(z):
    """Small, robust covariate cells available in both train and ranking data."""
    z = z.copy()
    # Delay is observable before scoring and is highly informative for the tail.
    z["_delay_bin"] = np.clip(np.floor(z["versp_min"].fillna(0) / 15), -8, 32).astype(int)
    z["_hour_bin"] = (z["stunde"].fillna(-1).astype(int) // 4).astype(int)
    z["_adep_key"] = z["ADEP_mvt"].astype(str)
    return z


def _weights(train, target):
    """Importance weights matching target cell frequencies to training cells."""
    keys = ["_adep_key", "_delay_bin", "_hour_bin"]
    a = train.groupby(keys, observed=True).size().rename("n_train")
    b = target.groupby(keys, observed=True).size().rename("n_target")
    tab = a.to_frame().join(b, how="outer").fillna(0)
    # Smoothing avoids explosive weights for sparse airport/delay cells.  The
    # ratio is subsequently normalized, preserving the effective sample size.
    ratio = (tab.n_target + 20.0) / (tab.n_train + 20.0)
    idx = pd.MultiIndex.from_frame(train[keys])
    w = ratio.reindex(idx).fillna(1.0).to_numpy(float)
    w /= w.mean()
    return np.clip(w, 0.20, 5.0)


def main():
    if not os.path.exists(BASE):
        raise SystemExit(f"{BASE} fehlt")

    print("Lade gecachte Trainingsmerkmale ...", flush=True)
    d = G.daten()
    lo = d[d.hat_aobt == 0].copy()
    lo["z"] = (lo.y > H.SCHWELLE_EXTREM).astype(int)
    quo = MSQ.passe_an(lo)
    lo = MSQ.ausser_faltung(lo)

    print(f"Baue Ranking-Merkmale ...", flush=True)
    rk = H.baue([RANKING], MG.alle())
    rk = MSQ.wende_an(quo, rk)
    rk = rk[rk.hat_aobt == 0].copy()
    lo = _bins(lo)
    rk = _bins(rk)
    if USE_SHIFT:
        w = _weights(lo, rk)
    else:
        w = np.ones(len(lo), dtype=float)
    print(f"Rueckfall: {len(lo):,} Lernzeilen, {len(rk):,} Rankingzeilen, "
          f"Gewichte {w.min():.2f}..{w.max():.2f}, Mittel {w.mean():.3f}")

    # Keep the historical no-arrival baseline after v44 integrated these columns.
    sp = [f for f in H.MERKMALE_RUECK
          if f not in ("ank_med_30min", "ank_med_90min")]
    if USE_ARRIVALS:
        sp += ["ank_med_30min", "ank_med_90min"]
    assert len(sp) == len(set(sp)), "duplicate feature names"
    common = dict(objective="binary", n_estimators=600, learning_rate=0.05,
                  num_leaves=31, min_child_samples=20, verbose=-1, n_jobs=-1,
                  random_state=0)
    clf = lgb.LGBMClassifier(**common)
    clf.fit(lo[sp], lo.z, sample_weight=w)

    normal = lgb.LGBMRegressor(objective="l2", n_estimators=400,
                               learning_rate=0.05, num_leaves=31,
                               min_child_samples=20, verbose=-1, n_jobs=-1,
                               random_state=0)
    good = lo.z == 0
    normal.fit(lo.loc[good, sp], lo.loc[good, "y"], sample_weight=w[good])

    height = lgb.LGBMRegressor(objective="l2", n_estimators=300,
                               learning_rate=0.05, num_leaves=15,
                               min_child_samples=10, verbose=-1, n_jobs=-1,
                               random_state=0)
    extreme = lo.z == 1
    height.fit(lo.loc[extreme, sp], lo.loc[extreme, "y"], sample_weight=w[extreme])

    # Copy the large, known-good model bundle and replace only the three fallback
    # estimators.  This keeps every main-group decision byte-for-byte equivalent.
    M = joblib.load(BASE)
    M["ohne_aobt"] = normal
    M["ohne_aobt_klass"] = clf
    M["ohne_aobt_hoehe"] = height
    M["merkmale_rueck"] = sp
    M["fallback_shift"] = ({"keys": ["ADEP", "delay_15min", "hour_4h"],
                             "smoothing": 20, "source": "ranking.parquet"}
                            if USE_SHIFT else None)
    M["fallback_arrivals"] = USE_ARRIVALS
    joblib.dump(M, OUT)
    print(f"gespeichert: {OUT}")


if __name__ == "__main__":
    main()
