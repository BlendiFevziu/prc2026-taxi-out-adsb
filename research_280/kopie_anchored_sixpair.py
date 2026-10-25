"""C6 re-validated against the recipe we actually SHIP, not the one fallback_cv.py models.

Why this exists. The build gate caught it: `models.pkl` does not reproduce the shipped fallback rows
(median 74 s off, 97.8 % of rows). v46 replaced the fallback lane with the ANCHORED model of
research_240/anchor_preview_2026_production.py, and fallback_cv.py predates that, so every fallback
number measured through it since 13 September, today's C6 evidence included, used a superseded baseline.

What the anchored recipe changes, read from that script rather than assumed: the z-classifier and the
height model keep the 35 features of H.MERKMALE_RUECK and are unchanged. Only the NORMAL branch moves,
from a plain regressor to `nt_anchor + offset`, where the offset is fitted on the 41 anchored features.
So the gate weight q = p_z ** HOCH is identical, and C6 stays additive:

    yhat_C6 = anchored + delta * (versp - normal_anchor)

Nothing is uploaded. Run from the repository root:
  python3 -u -B research_280/kopie_anchored_sixpair.py
"""
import glob, json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import neural_model as NM  # noqa: F401
import lightgbm as lgb

import main_model as H
import features_stand_rate as MSQ
import fallback_cv as F
from research_240 import features_neighbour_taxi as NT

OUT = Path("research_280/results/kopie_anchored_sixpair.json")
TAB = ".cache/research_240/anchor_preview/fallback_2025_all_*.parquet"
FEAT = list(H.MERKMALE_RUECK)
WITH_NT = FEAT + NT.MERKMALE
SCHWELLE, TOL = 0.5, 10


def reg(r, l, m):
    return lgb.LGBMRegressor(objective="l2", n_estimators=r, learning_rate=.05, num_leaves=l,
                             min_child_samples=m, verbose=-1, n_jobs=-1, random_state=0)


def klass():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                              min_child_samples=20, verbose=-1, n_jobs=-1, random_state=0)


def fold(lern, pruef, c6):
    """The shipped anchored recipe; with c6 the thresholded copy increment is added on top."""
    lern = lern.copy(); pruef = pruef.copy()
    lern["versp_s"] = lern.versp_min.to_numpy() * 60
    pruef["versp_s"] = pruef.versp_min.to_numpy() * 60
    lern["z"] = (lern.y > H.SCHWELLE_EXTREM).astype(int)
    lern["kopie"] = (lern.y - lern.versp_s).abs().le(TOL).astype(int)
    quote = MSQ.passe_an(lern)
    lern = MSQ.ausser_faltung(lern)
    pruef = MSQ.wende_an(quote, pruef)
    y = lern.y.to_numpy(float)
    z0 = lern.z.eq(0).to_numpy()

    cls = klass().fit(lern[FEAT], lern.z)
    q = cls.predict_proba(pruef[FEAT])[:, 1] ** H.HOCH
    hm = reg(300, 15, 10).fit(lern.loc[~z0, FEAT], y[~z0])
    high = (1 - H.ANTEIL_HM) * pruef.versp_s.to_numpy() + H.ANTEIL_HM * hm.predict(pruef[FEAT])
    normal = reg(400, 31, 20).fit(lern.loc[z0, FEAT], y[z0]).predict(pruef[FEAT])

    a_l, a_p = lern.nt_anchor.to_numpy(float), pruef.nt_anchor.to_numpy(float)
    rows = z0 & np.isfinite(a_l)
    off = reg(400, 31, 20).fit(lern.loc[rows, WITH_NT], y[rows] - a_l[rows])
    has = np.isfinite(a_p)
    normal_a = normal.copy()
    normal_a[has] = a_p[has] + off.predict(pruef.loc[has, WITH_NT])
    anchored = np.clip((1 - q) * normal_a + q * high, 0, None)
    if not c6:
        return anchored, np.zeros(len(pruef), bool)

    pk = klass().fit(lern[FEAT], lern.kopie).predict_proba(pruef[FEAT])[:, 1]
    pk = np.where(pk >= SCHWELLE, pk, 0.0)
    delta = np.maximum(pk ** H.HOCH - q, 0.0)
    return np.clip(anchored + delta * (pruef.versp_s.to_numpy() - normal_a), 0, None), delta > 0


def main():
    lo = pd.read_parquet(sorted(glob.glob(TAB))[0])
    lo["monat"] = pd.to_datetime(lo.MVT_TIME_UTC_mvt).dt.month
    lo["versp_s"] = lo.versp_min.to_numpy() * 60
    kopie = (lo.y - lo.versp_s).abs().le(TOL).to_numpy()
    ordin = (lo.y <= H.SCHWELLE_EXTREM).to_numpy()
    print(f"  {len(lo):,} anchored fallback rows, {int(kopie.sum()):,} copies", flush=True)

    qa = np.full(len(lo), np.nan); qc = np.full(len(lo), np.nan); fires = np.zeros(len(lo), bool)
    for pair in F.PAARE:
        te = lo.monat.isin(pair).to_numpy()
        pa, _ = fold(lo[~te], lo[te], False)
        pc, fi = fold(lo[~te], lo[te], True)
        yt = lo.y.to_numpy()[te]
        qa[te] = (pa - yt) ** 2; qc[te] = (pc - yt) ** 2; fires[te] = fi
        print(f"    pair {pair}: anchored MSE {qa[te].mean():,.0f}  C6 {qc[te].mean():,.0f}  "
              f"fires {int(fi.sum())}", flush=True)
    assert not np.isnan(qa).any() and not np.isnan(qc).any()

    sec = lambda x: F.ranglistenwirkung(0, -x)
    d = qa - qc
    prec = float(kopie[fires].mean()) if fires.any() else 0.0
    res = {"note": "six pairs on the SHIPPED anchored recipe; nothing uploaded",
           "rows": int(len(lo)), "anchored_mse": float(qa.mean()), "c6_mse": float(qc.mean()),
           "fires": int(fires.sum()), "fires_share": float(fires.mean()), "precision_on_fires": prec,
           "pooled_seconds": float(sec(d.mean())),
           "pairs_positive": int(sum(d[lo.monat.isin(p).to_numpy()].mean() > 0 for p in F.PAARE))}
    for k, m in (("ordinary", ordin & ~kopie), ("copies", kopie), ("broken", ~ordin & ~kopie)):
        res[k] = {"rows": int(m.sum()), "mse_per_row_gain": float(d[m].mean()),
                  "seconds": float(sec(d[m].mean()) * m.mean())}
    order = np.argsort(d)[::-1]
    res["robustness_drop_best"] = {str(k): float(sec(d[order[k:]].mean())) for k in (0, 10, 25, 50, 100)}
    print(f"\n  anchored MSE {qa.mean():,.0f}   C6 MSE {qc.mean():,.0f}")
    print(f"  fires {int(fires.sum()):,} ({fires.mean():.2%})   precision {prec:.3f}")
    print(f"  pooled {res['pooled_seconds']:+.3f} s   pairs {res['pairs_positive']}/6")
    for k in ("ordinary", "copies", "broken"):
        print(f"    {k:<10} {res[k]['rows']:>7,} rows {res[k]['mse_per_row_gain']:>+14,.0f} MSE/row "
              f"{res[k]['seconds']:>+7.3f} s")
    print("  robustness: " + "  ".join(f"k={k} {v:+.3f}" for k, v in res["robustness_drop_best"].items()))
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
