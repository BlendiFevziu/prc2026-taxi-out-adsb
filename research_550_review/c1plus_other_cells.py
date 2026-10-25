"""Test of research_550_review/PREREG_c1plus_other_cells.md. No fit, no upload.
Run from the repository root:  python3 -u -B research_550_review/c1plus_other_cells.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
for p in (HERE, HERE / "research_360_integration", HERE / "research_400_cells", HERE / "research_480_weights"):
    sys.path.insert(0, str(p))
import cells_confirm as CC  # noqa: E402
import weights as W  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

ID, N, REF = "MVT_ID_mvt", 344841, 252.91
PLUS = [joblib.load(HERE / f".cache/research_550/c1_more/c1plus_{s}.pkl") for s in L.SEEDS]
OUT = HERE / "research_550_review/results/c1plus_other_cells.json"


def seeds(frame, models):
    x = L.inputs("partial", frame)
    return {s: m.predict(x) for s, m in zip(L.SEEDS, models)}


def planning(rows, gain):
    mse = rows * gain / N
    return float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))


def decision_a(comps):
    out, rows_pass, mse_sum = {}, 0, 0.0
    for cell in ("EDDM/01", "LEMD/07"):
        spec = CC.CELLS[cell]
        f = CC.frame_new_airport(cell[:4], spec["days"], comps)
        y, b = f.y.to_numpy(float), f.comparator.to_numpy(float)
        old, new = seeds(f, L.load_models("partial")), seeds(f, PLUS)
        se = lambda c: (np.maximum(b + c, 300.0) - y) ** 2
        g = se(np.mean(list(old.values()), axis=0)) - se(np.mean(list(new.values()), axis=0))
        keep = np.ones(len(g), bool)
        keep[np.argsort(g, kind="stable")[-100:]] = False
        r = {"rows": int(len(f)), "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
             "blocks": {k: float(g[f.date.isin(v).to_numpy()].mean()) for k, v in spec["blocks"].items()},
             "seeds": {s: float((se(old[s]) - se(new[s])).mean()) for s in L.SEEDS}}
        r["rules"] = {"pooled": r["gain_mse"] > 0, "without_best100": r["without_best100"] > 0,
                      "both_blocks": min(r["blocks"].values()) > 0, "every_seed": all(v > 0 for v in r["seeds"].values())}
        r["pass"] = all(r["rules"].values())
        if r["pass"]:
            rows_pass += spec["rows_2026"]
            mse_sum += spec["rows_2026"] * r["gain_mse"]
        out[cell] = r
    out["planning_s_passing_cells"] = planning(1, mse_sum) if rows_pass else 0.0
    out["verdict"] = "pass" if rows_pass and out["planning_s_passing_cells"] >= 0.05 else "fail"
    out["passing_cells"] = [c for c in ("EDDM/01", "LEMD/07") if out[c]["pass"]] if out["verdict"] == "pass" else []
    return out


def decision_b(base):
    out = {}
    dev_ext = pd.concat([W.S.extract(tuple(W.S.STAGES["development"]), "development"), W.S.extract(tuple(W.S.STAGES["confirm"]), "confirm")],
                        ignore_index=True).drop_duplicates(ID)
    for name, ext, days in (("development", dev_ext, W.DEV_EHAM), ("fresh", W.S.extract(W.JAN10, "fresh_january"), W.JAN10)):
        f = W.keep_rows(ext, days, ("EHAM/01",))
        f = f.merge(W.eham_features(days), on=ID, how="inner", validate="one_to_one")
        W.check(f, base)
        y, b = f.y.to_numpy(float), f.comparator.to_numpy(float)
        addon = L.correction("eham_january", f)
        old, new = seeds(f, L.load_models("partial")), seeds(f, PLUS)
        se = lambda c: (np.maximum(b + (addon + c) / 2, 300.0) - y) ** 2
        g = se(np.mean(list(old.values()), axis=0)) - se(np.mean(list(new.values()), axis=0))
        keep = np.ones(len(g), bool)
        keep[np.argsort(g, kind="stable")[-100:]] = False
        out[name] = {"rows": int(len(f)), "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
                     "by_day": {d: float(g[f.date.eq(d).to_numpy()].mean()) for d in sorted(f.date.unique())},
                     "seeds": {s: float((se(old[s]) - se(new[s])).mean()) for s in L.SEEDS}}
    fr = out["fresh"]
    out["planning_s"] = planning(9752, fr["gain_mse"])
    out["rules"] = {"pooled": fr["gain_mse"] > 0, "without_best100": fr["without_best100"] > 0,
                    "seven_of_ten_days": sum(v > 0 for v in fr["by_day"].values()) >= 7,
                    "every_seed": all(v > 0 for v in fr["seeds"].values()), "planning_0_05": out["planning_s"] >= 0.05}
    out["verdict"] = "pass" if all(out["rules"].values()) else "fail"
    return out


def main():
    comps = H.components_2025()
    comps["comparator"] = L.compose_base(comps)
    base = pd.Series(comps.comparator.to_numpy(float), index=comps[ID].to_numpy())
    report = {"prereg": "research_550_review/PREREG_c1plus_other_cells.md", "a": decision_a(comps), "b": decision_b(base)}
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
