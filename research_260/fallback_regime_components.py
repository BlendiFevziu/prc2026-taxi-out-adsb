"""Where the v46 fallback loses squared error by block-time regime: gate, height or normal branch. Diagnostic, no candidate.

Written 14.09.2026 after research_260/day_shift_regime_screen.py rejected a day-shift component. No leaderboard feedback.

The scored v46 fallback recipe (production encoding, research_240/fallback_weather_screen.py recipe) is rebuilt per six-pair
split with its components kept: extreme classifier probability p and gate q = p ** HOCH, height model, height
(1 - ANTEIL_HM) * delay + ANTEIL_HM * height model, anchored normal branch, final prediction. GATE: the final prediction must
reproduce research_240/fallback_anchor_production.py (prod_anchor) within 1e-6 s on every pair, else the script stops.

Regimes as in research_260/fallback_regime_census.py (copy |y - delay| <= 60 s; day_shift; normal; other), extreme = y > 5,400 s.
Reported on the 22,470 held-out rows: squared-error share by regime x gate band (q < 0.1, 0.1-0.5, 0.5-0.9, >= 0.9); oracle MSE
gains per fallback row and at the 2026 fallback share for
  O1 copy height      extreme copies keep q but their height becomes the delay
  O2 copy gate        extreme copies keep their height but q = 1
  O3 copy exact ext   extreme copies predicted as the delay
  O4 copy exact norm  non-extreme copies predicted as the delay
  O5 other gate       other and day-shift extremes keep their height but q = 1
Achievability, AUC only: a copy classifier trained on the training months' extreme rows (copy against non-copy) scored on held-out
extremes, and one trained on non-extreme rows scored on held-out non-extremes (extreme-classifier settings, 35 fallback inputs).

Run from the repository root:  python3 -B research_260/fallback_regime_components.py --project .
"""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, default=Path("."))
parser.add_argument("--jobs", type=int, default=4)
args = parser.parse_args()
project = args.project.resolve()
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS
import features_stand_rate as MSQ
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_240 import features_neighbour_taxi as NT

BASE = list(H.MERKMALE_RUECK)
OFFSET = BASE + [c for c in NT.MERKMALE if c not in BASE]
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
TABLES = project / ".cache/research_240/rescreen_production"
SAVED = project / ".cache/research_240/fallback_anchor_production"
OUTPUT = project / "research_260/results/fallback_regime_components.json"
SHARE_2026 = 5290 / 344841
index = {tuple(sorted(json.loads(s.read_text())["source_months"])): s.with_suffix(".parquet") for s in TABLES.glob("rescreen_*.json")}


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05, num_leaves=leaves, min_child_samples=minimum,
                             verbose=-1, n_jobs=args.jobs, random_state=0)


def classifier():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31, min_child_samples=20,
                              verbose=-1, n_jobs=args.jobs, random_state=0)


def auc(labels, scores):
    labels = np.asarray(labels, bool)
    if labels.all() or not labels.any():
        return None
    ranks = pd.Series(scores).rank().to_numpy()
    n1, n0 = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


frames = []
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    learn = [m for m in range(1, 13) if m not in pair]
    tr, va = pd.read_parquet(index[tuple(learn)]).reset_index(drop=True), pd.read_parquet(index[tuple(sorted(pair))]).reset_index(drop=True)
    tr["z"] = (tr.y > H.SCHWELLE_EXTREM).astype(int)
    mapping = MSQ.passe_an(tr)
    tr, va = MSQ.ausser_faltung(tr), MSQ.wende_an(mapping, va)
    z0, ytr = tr.z.eq(0).to_numpy(), tr.y.to_numpy(float)
    p = classifier().fit(tr[BASE], tr.z).predict_proba(va[BASE])[:, 1]
    hm = reg(300, 15, 10).fit(tr.loc[~z0, BASE], ytr[~z0]).predict(va[BASE])
    delay = 60 * va.versp_min.to_numpy(float)
    height = (1 - H.ANTEIL_HM) * delay + H.ANTEIL_HM * hm
    normal = reg(400, 31, 20).fit(tr.loc[z0, BASE], ytr[z0]).predict(va[BASE])
    a_tr, a_va = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
    rows = z0 & np.isfinite(a_tr)
    offset = reg(400, 31, 20).fit(tr.loc[rows, OFFSET], ytr[rows] - a_tr[rows])
    has = np.isfinite(a_va)
    anchored = normal.copy()
    anchored[has] = a_va[has] + offset.predict(va.loc[has, OFFSET])
    q = p ** H.HOCH
    final = np.maximum((1 - q) * anchored + q * height, 0)
    saved = pd.read_parquet(SAVED / f"predictions_{key}.parquet").set_index("MVT_ID_mvt").loc[va.MVT_ID_mvt, "prod_anchor"].to_numpy(float)
    gap = float(np.abs(final - saved).max())
    print("GATE", key, gap, flush=True)
    if gap > 1e-6:
        raise SystemExit(f"component rebuild does not reproduce prod_anchor on {key}")
    train_copy = np.abs(ytr - 60 * tr.versp_min.to_numpy(float)) <= 60
    c_ext = classifier().fit(tr.loc[~z0, BASE], train_copy[~z0].astype(int)).predict_proba(va[BASE])[:, 1]
    c_norm = classifier().fit(tr.loc[z0, BASE], train_copy[z0].astype(int)).predict_proba(va[BASE])[:, 1]
    frames.append(pd.DataFrame({"fold": key, "airport": va.ADEP_mvt.astype(str).to_numpy(), "y": va.y.to_numpy(float), "delay": delay,
                                "p": p, "q": q, "hm": hm, "height": height, "anchored": anchored, "final": final,
                                "c_ext": c_ext, "c_norm": c_norm}))

f = pd.concat(frames, ignore_index=True)
y, delay, q, height, anchored, final = (f[c].to_numpy(float) for c in ("y", "delay", "q", "height", "anchored", "final"))
copy = np.abs(y - delay) <= 60
shift = ~copy & (y >= 80000) & (y <= 93600)
normal = ~copy & (y >= -600) & (y <= 5400)
other = ~copy & ~shift & ~normal
extreme = y > H.SCHWELLE_EXTREM
se = (final - y) ** 2
sse = se.sum()
bands = {"q<0.1": q < .1, "0.1<=q<0.5": (q >= .1) & (q < .5), "0.5<=q<0.9": (q >= .5) & (q < .9), "q>=0.9": q >= .9}
regimes = {"copy_extreme": copy & extreme, "copy_nonextreme": copy & ~extreme, "day_shift": shift, "normal": normal,
           "other_extreme": other & extreme, "other_nonextreme": other & ~extreme}


def oracle_gain(mask, replacement):
    alt = final.copy()
    alt[mask] = np.maximum(replacement[mask], 0)
    gain = float(((final - y) ** 2 - (alt - y) ** 2).mean())
    return {"rows": int(mask.sum()), "fallback_mse_gain_per_row": gain, "leaderboard_mse_equivalent": gain * SHARE_2026}


report = {"note": "diagnostic; v46 fallback components rebuilt on six pairs, gate passed", "rows": int(len(f)), "fallback_rmse": float(np.sqrt(se.mean())),
          "se_share_by_regime_and_gate": {r: {b: {"rows": int((m & bm).sum()), "se_share": float(se[m & bm].sum() / sse)} for b, bm in bands.items()}
                                          for r, m in regimes.items()},
          "copy_extreme_detail": {"rows": int((copy & extreme).sum()), "mean_q": float(q[copy & extreme].mean()),
                                  "median_abs_height_minus_delay_s": float(np.median(np.abs(height - delay)[copy & extreme])),
                                  "median_abs_height_model_minus_delay_s": float(np.median(np.abs(f.hm.to_numpy() - delay)[copy & extreme]))},
          "oracles": {"O1_copy_height": oracle_gain(copy & extreme, (1 - q) * anchored + q * delay),
                      "O2_copy_gate": oracle_gain(copy & extreme, height),
                      "O3_copy_exact_extreme": oracle_gain(copy & extreme, delay),
                      "O4_copy_exact_nonextreme": oracle_gain(copy & ~extreme, delay),
                      "O5_other_gate": oracle_gain((other | shift) & extreme, height)},
          "achievability_auc": {"copy_among_extremes": auc(copy[extreme], f.c_ext.to_numpy()[extreme]),
                                "copy_among_nonextremes": auc(copy[~extreme], f.c_norm.to_numpy()[~extreme]),
                                "extreme_classifier_p": auc(extreme, f.p.to_numpy()),
                                "copy_share_among_extremes": float(copy[extreme].mean()), "copy_share_among_nonextremes": float(copy[~extreme].mean())}}
for fold, g in f.groupby("fold"):
    m = g.y.to_numpy() > H.SCHWELLE_EXTREM
    cp = np.abs(g.y.to_numpy() - g.delay.to_numpy()) <= 60
    report.setdefault("auc_by_fold", {})[fold] = {"copy_among_extremes": auc(cp[m], g.c_ext.to_numpy()[m]),
                                                  "copy_among_nonextremes": auc(cp[~m], g.c_norm.to_numpy()[~m])}
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in ("rows", "fallback_rmse", "copy_extreme_detail", "oracles", "achievability_auc", "auc_by_fold")}, indent=1))
print("SE share by regime and gate band:")
for r, v in report["se_share_by_regime_and_gate"].items():
    print(f"  {r:17s}", {b: (x["rows"], round(x["se_share"], 4)) for b, x in v.items()})
