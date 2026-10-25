"""Step 4: split the fallback error into a part the harness CAN measure and a part it cannot.

Why not a pooled MSE. fallback_cv.py fails its own calibration probe (experiments.csv row 71, and
again 23.09.2026: it ranks HOCH 0.8 above 1.6 when the scored set says the reverse), and the
top-100 rule rejects the height apparatus that the scored set values at 6.53 s. Both failures come
from the same place, a handful of rows whose squared error reaches 1.7e10.

Why not an expected-loss estimator either. Replacing the realised outcome by q * (versp - yhat)^2 +
(1 - q) * (mu_n - yhat)^2 removes the coin flip, but it also treats a per-airport mean as the truth
for every non-copy row, which penalises any arm whose regressor beats that mean. That trades a
variance problem for a bias problem and was dropped.

What is left is honest and enough to decide. The rows split into three classes with very different
statistical behaviour:

  ordinary      y <= 5400 s, about 21,700 rows, BOUNDED labels
                -> the sample mean here is low variance and trustworthy
  copies        |y - versp| <= 10 s, where the mechanism is exact
                -> the interesting rows, moderate variance
  broken        everything else above 5400 s, about 84 rows in 2025
                -> unpredictable by construction, dominates any pooled number

A change that damages the ordinary class is refuted outright, because that class is measurable.
A change that only moves the broken class cannot be adjudicated here and is reported as such.

Nothing is uploaded. Run from the repository root:
  python3 -u -B research_280/kopie_decomposition.py
"""
import json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np
import neural_model as NM  # noqa: F401

import main_model as H
import fallback_cv as F
from research_280.kopie_gate_sixpair import arm, TOL

OUT = Path("research_280/results/kopie_decomposition.json")
ARME = {
    "A0_prod":   dict(ziel="z",     hoch=H.HOCH, anteil_hm=H.ANTEIL_HM),
    "A5_label":  dict(ziel="kopie", hoch=H.HOCH, anteil_hm=H.ANTEIL_HM, teilung="z"),
    "A6_lbl_w1": dict(ziel="kopie", hoch=1.0,    anteil_hm=H.ANTEIL_HM, teilung="z"),
}


def main():
    lo = F.vorbereiten().copy()
    lo["versp_s"] = lo.versp_min.to_numpy() * 60
    kopie = (lo.y - lo.versp_s).abs().le(TOL).to_numpy()
    ordin = (lo.y <= H.SCHWELLE_EXTREM).to_numpy()
    broken = ~ordin & ~kopie
    klassen = {"ordinary (bounded, trustworthy)": ordin & ~kopie,
               "copies (the mechanism)": kopie,
               "broken (unadjudicable)": broken}
    print("  class sizes:")
    for k, m in klassen.items():
        print(f"    {k:<34} {int(m.sum()):>7,} rows")
    print(f"\n  1 s on the board = {1/F.ranglistenwirkung(1, 0):,.0f} MSE per row\n")

    q = {}
    for name, kw in ARME.items():
        q[name] = F.messe(arm(**kw), lo, name, laut=False)
        np.save(f".cache/kopie_{name}.npy", q[name])
        print(f"  {name:<12} pooled MSE {q[name].mean():12,.0f}", flush=True)

    base = q["A0_prod"]
    res = {"note": "decomposition by row class; pooled numbers on this lane are not adjudicable",
           "class_sizes": {k: int(m.sum()) for k, m in klassen.items()}, "arms": {}}
    print(f"\n  {'arm':<12}" + "".join(f"{k.split(' ')[0]:>16}" for k in klassen) + f"{'pooled':>16}")
    print(f"  {'':12}" + "".join(f"{'MSE/row  s':>16}" for _ in klassen) + f"{'s':>16}")
    for name in ARME:
        if name == "A0_prod":
            continue
        row, entry = f"  {name:<12}", {}
        for k, m in klassen.items():
            d = float((base[m] - q[name][m]).mean())
            # contribution to the board = per-class gain weighted by that class's share
            s = F.ranglistenwirkung(0, -d) * m.mean()
            row += f"{d:>10,.0f}{s:>+7.3f}"
            entry[k] = {"mse_per_row_gain": d, "seconds_contribution": float(s)}
        s_all = F.ranglistenwirkung(base.mean(), q[name].mean())
        row += f"{s_all:>+16.3f}"
        entry["pooled_seconds_NOT_ADJUDICABLE"] = float(s_all)
        print(row)
        res["arms"][name] = entry

    print("\n  verdict on the only class that is measurable here:")
    for name in ARME:
        if name == "A0_prod":
            continue
        d = res["arms"][name]["ordinary (bounded, trustworthy)"]["mse_per_row_gain"]
        print(f"    {name:<12} ordinary rows {d:+,.0f} MSE/row  "
              f"-> {'HARMLESS' if d > -50 else 'REFUTED, it damages measurable rows'}")

    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
