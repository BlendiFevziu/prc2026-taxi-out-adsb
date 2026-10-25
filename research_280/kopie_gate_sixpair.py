"""Step 3 of PREREG_kopie_gate.md: the blend arms in the six-pair harness.

Order of business, fixed in the pre-registration:
  1. instrument calibration   HOCH 1.6 vs 0.8, the scored set says 1.6 wins by 0.509 s
  2. RULE calibration         HOCH 2.5 vs 50, the scored set says 2.5 wins by 6.53 s.
                              If the top-100 rule rejects THAT, the rule is uninformative here.
  3. the five arms            A0 prod, A1 label, A2 weight, A3 full, A4 full+calibrated

The stand-rate feature q_stand stays defined on the extreme label in every arm, so the arms differ
only in what the pre-registration says they differ in. Nothing is uploaded.

Run from the repository root:
  python3 -u -B research_280/kopie_gate_sixpair.py
"""
import json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np
import neural_model as NM  # noqa: F401
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

import main_model as H
import features_stand_rate as MSQ
import fallback_cv as F

OUT = Path("research_280/results/kopie_gate_sixpair.json")
TOL = 10


def _reg(n, leaves, child):
    return lgb.LGBMRegressor(objective="l2", n_estimators=n, learning_rate=0.05,
                             num_leaves=leaves, min_child_samples=child, verbose=-1, n_jobs=-1)


def _klass():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.05,
                              num_leaves=31, min_child_samples=20, verbose=-1, n_jobs=-1)


def arm(ziel="z", hoch=H.HOCH, anteil_hm=H.ANTEIL_HM, kalibriert=False, teilung=None, gewicht=0.0,
        union=False, union_bare=False, union_schwelle=0.0, union_hoch=None):
    """ziel: which label the CLASSIFIER learns. teilung: which label PARTITIONS the two regressors
    (defaults to ziel, which is what production does). Separating them matters: changing the label
    alone also moves the 84 broken extremes into the normal regressor's training set, which is a
    different change and, measured 23.09.2026, a very damaging one.
    hoch: exponent on p. anteil_hm: weight of the height model against bare versp.
    kalibriert: fit isotonic p->kopie inside the learning months."""
    def f(lern, pruef):
        lern = lern.copy(); pruef = pruef.copy()
        lern["versp_s"] = lern.versp_min.to_numpy() * 60
        pruef["versp_s"] = pruef.versp_min.to_numpy() * 60
        lern["z"] = (lern.y > H.SCHWELLE_EXTREM).astype(int)          # always, for q_stand
        lern["kopie"] = (lern.y - lern.versp_s).abs().le(TOL).astype(int)
        quo = MSQ.passe_an(lern)
        lern = MSQ.ausser_faltung(lern)
        pruef = MSQ.wende_an(quo, pruef)
        X, Xp = lern[H.MERKMALE_RUECK], pruef[H.MERKMALE_RUECK]

        sw = None
        if gewicht:
            # Under squared error a weight error costs D * (w - q)^2 with D = (height - normal)^2,
            # so the classifier should spend its capacity where D is large. versp minus the airport
            # median is a model-free stand-in for the scale of D.
            med = lern.groupby("ADEP_mvt").y.transform("median").to_numpy()
            scale = np.abs(lern.versp_s.to_numpy() - med)
            sw = 1.0 + (scale / 3600.0) ** gewicht
        m2c = _klass().fit(X, lern[ziel], sample_weight=sw)
        p = m2c.predict_proba(Xp)[:, 1]
        p_basis = p                      # the shipped gate's own probability, kept for union_bare
        if union:
            # Keep the consequence classifier's recall on the large rows and let the mechanism
            # classifier only ADD weight where it is confident. max() can never lower p, so this
            # cannot undo anything the shipped gate already gets right.
            other = "kopie" if ziel == "z" else "z"
            p_other = _klass().fit(X, lern[other]).predict_proba(Xp)[:, 1]
            if union_schwelle:
                # C5 touched 11,779 rows at 5.7 % precision and its whole pooled gain rested on
                # about 25 rows. Only let the other classifier speak where it is actually confident.
                p_other = np.where(p_other >= union_schwelle, p_other, 0.0)
            p_other_kept = p_other        # kept so the increment can carry its OWN exponent
            p = np.maximum(p, p_other)
        if kalibriert:
            oof = np.full(len(lern), np.nan)
            for mo in sorted(lern.monat.unique()):
                inner = lern.monat.ne(mo).to_numpy()
                oof[~inner] = _klass().fit(X[inner], lern[ziel][inner]).predict_proba(X[~inner])[:, 1]
            iso = IsotonicRegression(y_min=0., y_max=1., increasing=True, out_of_bounds="clip")
            iso.fit(oof, lern["kopie"].to_numpy())
            p = iso.predict(p)
        pw = p ** hoch

        pos = lern[teilung or ziel].to_numpy() == 1
        m2 = _reg(400, 31, 20).fit(X[~pos], lern.y[~pos])
        if anteil_hm > 0:
            m2h = _reg(300, 15, 10).fit(X[pos], lern.y[pos])
            hoehe = (1 - anteil_hm) * pruef.versp_s.to_numpy() + anteil_hm * m2h.predict(Xp)
        else:
            hoehe = pruef.versp_s.to_numpy()
        if union_bare:
            # The union's EXTRA weight caused the whole loss on ordinary rows, because
            # anteil_hm holds `hoehe` far above the normal prediction. Give that increment a bare
            # versp height instead: on an ordinary row versp is close to the normal prediction, so
            # the increment is nearly free, while the shipped gate keeps its own blended height.
            pw_basis = p_basis ** hoch
            # Where the increment fires w_z is near zero, so a move of size delta toward versp gains
            # D*(2*delta - delta^2) on a true copy and costs D*delta^2 on a non-copy, i.e.
            #     EV = D * delta * (2p - delta),  maximised at delta = p.
            # With the shipped exponent delta is about p**2.5, far below p. union_hoch sets it.
            hi = hoch if union_hoch is None else union_hoch
            delta = np.maximum(p_other_kept ** hi - pw_basis, 0.0)
            rest = np.clip(1.0 - pw_basis - delta, 0.0, None)
            return np.clip(rest * m2.predict(Xp) + pw_basis * hoehe
                           + delta * pruef.versp_s.to_numpy(), 0, None)
        return np.clip((1 - pw) * m2.predict(Xp) + pw * hoehe, 0, None)
    return f


def top100(q_a, q_b):
    """Pooled gain after dropping the 100 rows where b gains most over a."""
    d = q_a - q_b
    keep = np.argsort(d)[:-100]
    return float(d.mean()), float(d[keep].mean()), F.ranglistenwirkung(0, -d[keep].mean())


def main():
    lo = F.vorbereiten()
    res = {"note": "six pairs in the shape of the real case; nothing uploaded", "arms": {}}
    print(f"  1 s on the board = {1/F.ranglistenwirkung(1, 0):,.0f} MSE per row\n")

    print("=== 1. INSTRUMENT CALIBRATION: the board says HOCH 1.6 beats 0.8 by 0.509 s ===")
    q16 = F.messe(arm(hoch=1.6), lo, "H1.6", laut=False)
    q08 = F.messe(arm(hoch=0.8), lo, "H0.8", laut=False)
    s, l_, h_ = F.vergleiche(q16, q08, "HOCH 1.6 (shipped)", "HOCH 0.8 (rejected)")
    ok_instr = h_ < 0
    print(f"  -> {'PASSED' if ok_instr else 'NOT REPRODUCED'}: instrument {s:+.3f} s, board -0.509 s")
    res["instrument"] = {"seconds": s, "ci": [l_, h_], "passed": bool(ok_instr)}

    print("\n=== 2. RULE CALIBRATION: the board says HOCH 2.5 beats HOCH 50 by 6.53 s ===")
    print("  If the top-100 rule rejects this known-good lever, the rule cannot judge this lane.")
    q25 = F.messe(arm(hoch=2.5), lo, "H2.5", laut=False)
    q50 = F.messe(arm(hoch=50.), lo, "H50", laut=False)
    s2, _, _ = F.vergleiche(q50, q25, "HOCH 50 (apparatus off)", "HOCH 2.5 (shipped)")
    full, cut, cut_s = top100(q50, q25)
    print(f"  pooled gain {full:,.0f} MSE/row ({s2:+.3f} s); without its best 100 rows "
          f"{cut:,.0f} MSE/row ({cut_s:+.3f} s)")
    rule_usable = cut > 0
    print(f"  -> top-100 rule on a lever the board values at 6.53 s: "
          f"{'SURVIVES, rule stands' if rule_usable else 'REJECTS IT, rule is uninformative here'}")
    res["rule_control"] = {"seconds": s2, "pooled_mse_per_row": full,
                           "without_top100": cut, "without_top100_seconds": cut_s,
                           "rule_usable": bool(rule_usable)}

    print("\n=== 3. ARMS ===")
    specs = {
        "A0_prod":      dict(ziel="z",     hoch=H.HOCH, anteil_hm=H.ANTEIL_HM),
        "A1_label":     dict(ziel="kopie", hoch=H.HOCH, anteil_hm=H.ANTEIL_HM),
        "A2_weight":    dict(ziel="z",     hoch=1.0,    anteil_hm=H.ANTEIL_HM),
        "A3_full":      dict(ziel="kopie", hoch=1.0,    anteil_hm=0.0),
        "A4_full_cal":  dict(ziel="kopie", hoch=1.0,    anteil_hm=0.0, kalibriert=True),
    }
    q = {}
    for name, kw in specs.items():
        q[name] = F.messe(arm(**kw), lo, name, laut=False)
        print(f"  {name:<14} MSE {q[name].mean():12,.0f}  RMSE {np.sqrt(q[name].mean()):9,.1f}", flush=True)
        np.save(f".cache/kopie_{name}.npy", q[name])

    base = q["A0_prod"]
    ordinary = (lo.y <= H.SCHWELLE_EXTREM).to_numpy()
    for name in specs:
        if name == "A0_prod":
            continue
        print(f"\n  --- {name} against A0_prod ---")
        s, l_, h_ = F.vergleiche(base, q[name], "A0_prod", name)
        full, cut, cut_s = top100(base, q[name])
        pairs = [int(np.sign((base - q[name])[lo.monat.isin(p).to_numpy()].mean()) > 0) for p in F.PAARE]
        ord_gain = float((base - q[name])[ordinary].mean())
        print(f"    pairs positive {sum(pairs)}/6 | without top 100: {cut_s:+.3f} s | "
              f"ordinary rows {ord_gain:+,.0f} MSE/row")
        res["arms"][name] = {"seconds": s, "ci": [l_, h_], "pairs_positive": sum(pairs),
                             "mse_per_row": float(q[name].mean()),
                             "without_top100_seconds": cut_s,
                             "ordinary_rows_mse_gain": ord_gain,
                             "checks": {"pooled_positive": bool(s > 0),
                                        "at_least_4_of_6": bool(sum(pairs) >= 4),
                                        "ordinary_not_worse": bool(ord_gain >= 0),
                                        "survives_top100": bool(cut > 0)}}
    res["arms"]["A0_prod"] = {"mse_per_row": float(base.mean())}
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
