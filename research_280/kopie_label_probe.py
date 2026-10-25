"""Step 1 and 2 of PREREG_kopie_gate.md: is the mechanism label a better classifier target,
and does its calibration transfer from 2025 to 2026?

Cheap, low variance, and decisive for the primary rules. No blend is built here, no candidate,
nothing uploaded. Both classifiers are fitted out of fold in the shape of the real case (learn on
ten months, score the held-out pair), exactly as fallback_cv.py folds.

  consequence label   z     = y > SCHWELLE_EXTREM (5400 s)   what the fallback lane uses today
  mechanism label     kopie = |y - versp| <= 10 s            what the MAIN lane uses (ERF_HOCH)

Both are then judged on the event the blend actually needs, "is y equal to versp", because that is
the event for which the height branch is exact.

Run from the repository root:
  python3 -u -B research_280/kopie_label_probe.py
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import neural_model as NM  # noqa: F401  torch before the tree libraries on macOS
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss

import main_model as H
import features_stand_rate as MSQ
import fallback_cv as F

OUT = Path("research_280/results/kopie_label_probe.json")
RANK = Path(".cache/clock_queue_release_20260912/data/ranking.parquet")
TOL_KOPIE = 10          # the main lane's own tolerance, main_model.py line 332
BINS = 10


def klass():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.05,
                              num_leaves=31, min_child_samples=20, verbose=-1, n_jobs=-1)


def ece(p, label):
    """Expected calibration error over equal-width bins, plus the reliability table."""
    p = np.asarray(p, float); label = np.asarray(label, float)
    edges = np.linspace(0, 1, BINS + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, BINS - 1)
    err, table = 0.0, []
    for b in range(BINS):
        m = idx == b
        if not m.any():
            continue
        mp, my = float(p[m].mean()), float(label[m].mean())
        err += m.mean() * abs(mp - my)
        table.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                      "mean_p": round(mp, 4), "observed": round(my, 4)})
    return float(err), table


def main():
    lo = F.vorbereiten()
    lo = lo.copy()
    lo["versp_s"] = lo.versp_min.to_numpy() * 60
    lo["kopie"] = (lo.y - lo.versp_s).abs().le(TOL_KOPIE).astype(int)
    lo["z"] = (lo.y > H.SCHWELLE_EXTREM).astype(int)
    print(f"  {len(lo):,} fallback rows")
    print(f"  mechanism label kopie : {lo.kopie.sum():,} positives ({lo.kopie.mean():.2%})")
    print(f"  consequence label z   : {lo.z.sum():,} positives ({lo.z.mean():.2%})")
    both = int(((lo.kopie == 1) & (lo.z == 1)).sum())
    print(f"  overlap: {both:,} rows are both; {int(((lo.kopie==1)&(lo.z==0)).sum()):,} copies are "
          f"INVISIBLE to the consequence label (y <= 5400 s)")
    print(f"  on copy rows max |y - versp| = {(lo.y - lo.versp_s)[lo.kopie == 1].abs().max():.1f} s\n")

    oof = {"z": np.full(len(lo), np.nan), "kopie": np.full(len(lo), np.nan)}
    for paar in F.PAARE:
        te = lo.monat.isin(paar).to_numpy()
        lern, pruef = lo[~te].copy(), lo[te].copy()
        quo = MSQ.passe_an(lern)
        lern = MSQ.ausser_faltung(lern)
        pruef = MSQ.wende_an(quo, pruef)
        for name in ("z", "kopie"):
            m = klass().fit(lern[H.MERKMALE_RUECK], lern[name])
            oof[name][te] = m.predict_proba(pruef[H.MERKMALE_RUECK])[:, 1]
        print(f"    pair {paar}: {int(te.sum()):,} rows scored", flush=True)

    y_kopie = lo.kopie.to_numpy()
    res = {"note": "out-of-fold, six pairs, shape of the real case; no blend, nothing uploaded",
           "rows": int(len(lo)), "tol_kopie_s": TOL_KOPIE,
           "positives": {"kopie": int(lo.kopie.sum()), "z": int(lo.z.sum()),
                         "copies_invisible_to_z": int(((lo.kopie == 1) & (lo.z == 0)).sum())}}

    print("\n=== RULE 1: predicting the COPY event (the event the height branch is exact for) ===")
    print(f"  {'classifier':<26}{'AUC':>9}{'Brier':>11}{'ECE':>9}")
    res["on_copy_event"] = {}
    for name, p in (("trained on z (current)", oof["z"]), ("trained on kopie (proposed)", oof["kopie"]),
                    ("p**2.5 of z (shipped weight)", oof["z"] ** H.HOCH)):
        auc = roc_auc_score(y_kopie, p); br = brier_score_loss(y_kopie, p); e, tab = ece(p, y_kopie)
        print(f"  {name:<26}{auc:>9.4f}{br:>11.5f}{e:>9.4f}")
        res["on_copy_event"][name] = {"auc": float(auc), "brier": float(br), "ece": e, "reliability": tab}

    print("\n  each classifier on its OWN label, for reference:")
    res["on_own_label"] = {}
    for name, p, lab in (("z", oof["z"], lo.z.to_numpy()), ("kopie", oof["kopie"], y_kopie)):
        auc = roc_auc_score(lab, p)
        print(f"    {name:<8} AUC {auc:.4f}")
        res["on_own_label"][name] = float(auc)

    print("\n=== RULE 2: reliability of the proposed weight vs the shipped one ===")
    for name in ("trained on kopie (proposed)", "p**2.5 of z (shipped weight)"):
        print(f"  {name}")
        for r in res["on_copy_event"][name]["reliability"]:
            if r["n"] >= 20:
                print(f"    {r['bin']:<10} n={r['n']:>6,}  says {r['mean_p']:.3f}  is {r['observed']:.3f}")

    print("\n=== TRANSFER 2025 -> 2026, no labels read, no leaderboard ===")
    if RANK.exists():
        rank = pd.read_parquet(RANK)
        r26 = rank[rank.hat_aobt == 0].copy()
        # q_stand / q_stand_n are CREATED by wende_an, so the column check must follow it.
        quo = MSQ.passe_an(lo)
        lern = MSQ.ausser_faltung(lo.copy())
        r26 = MSQ.wende_an(quo, r26)
        fehlt = [c for c in H.MERKMALE_RUECK if c not in r26.columns]
        if fehlt:
            print(f"  SKIPPED: {len(fehlt)} feature columns missing from the ranking frame, e.g. {fehlt[:5]}")
            res["transfer"] = {"skipped": True, "missing_features": fehlt[:20]}
        else:
            t = {}
            for name in ("z", "kopie"):
                m = klass().fit(lern[H.MERKMALE_RUECK], lern[name])
                p26 = m.predict_proba(r26[H.MERKMALE_RUECK])[:, 1]
                p25 = oof[name]
                qs = [.5, .75, .9, .95, .99]
                t[name] = {"n_2026": int(len(p26)),
                           "mean_2025_oof": float(p25.mean()), "mean_2026": float(p26.mean()),
                           "quantiles_2025_oof": {str(q): float(np.quantile(p25, q)) for q in qs},
                           "quantiles_2026": {str(q): float(np.quantile(p26, q)) for q in qs},
                           "share_over_0.5_2025": float((p25 > .5).mean()),
                           "share_over_0.5_2026": float((p26 > .5).mean())}
                print(f"  {name:<6} mean p: 2025 oof {p25.mean():.4f} -> 2026 {p26.mean():.4f}   "
                      f"share>0.5: {(p25>.5).mean():.4f} -> {(p26>.5).mean():.4f}")
            res["transfer"] = t
    else:
        print(f"  SKIPPED: {RANK} not found")
        res["transfer"] = {"skipped": True}

    np.save(".cache/oof_p_z.npy", oof["z"]); np.save(".cache/oof_p_kopie.npy", oof["kopie"])
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
