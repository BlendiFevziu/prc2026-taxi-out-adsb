"""Test 1 (C1 at weight 1.25) and test 2 (Amsterdam January: add-on averaged with C1) of
research_480_weights/PREREG_weights.md. No fit, no upload.
Run from the repository root:  python3 -u -B research_480_weights/weights.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
from research_320 import volume_capacity as V  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

S = V.S
ID, N, REF, W = "MVT_ID_mvt", 344841, 256.0858, 1.25
JAN10 = ("2025-01-03", "2025-01-10", "2025-01-11", "2025-01-12", "2025-01-15", "2025-01-17", "2025-01-22", "2025-01-27", "2025-01-29", "2025-01-31")
DEV_EHAM = ("2025-01-04", "2025-01-14", "2025-01-21", "2025-01-26", "2025-01-28", "2025-01-01", "2025-01-02", "2025-01-18",
            "2025-01-06", "2025-01-13", "2025-01-20")
FEATURE_DIRS = (".cache/codex_eham/eham_recovery", ".cache/research_320/eham_weight/features", ".cache/research_360/eham_features",
                ".cache/research_480/eham_features")
OUT = HERE / "research_480_weights/results/weights.json"


def keep_rows(frame, dates, cells, category="partial_ground"):
    with V.heavy_work("research_480: detector replay", poll_s=10.0):
        det = set(V.detector_ids(dates))
    f = frame.copy()
    f["cell"] = f.airport + "/" + f.date.str[5:7]
    return f[f.date.isin(dates) & f.cell.isin(cells) & f.category.eq(category) & ~f[ID].isin(det)].reset_index(drop=True)


def check(frame, base):
    if not np.array_equal(base.loc[frame[ID]].to_numpy(float), frame.comparator.to_numpy(float)):
        raise SystemExit("comparator not bit-identical")


def stats(gain, frame):
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-100:]] = False
    return {"rows": int(len(gain)), "gain_mse": float(gain.mean()), "without_best100": float(gain[keep].mean()),
            "by_cell": {c: float(gain[frame.cell.eq(c).to_numpy()].mean()) for c in sorted(frame.cell.unique())},
            "by_day": {d: float(gain[frame.date.eq(d).to_numpy()].mean()) for d in sorted(frame.date.unique())}}


def test1(base):
    c1_mean = lambda f: np.mean(list(V.seed_corrections(f, "c1").values()), axis=0)
    out = {}
    for name, frame in (("B1", V.eligible(S.extract(V.B1, "development"), V.detector_ids(V.B1))),
                        ("B2", V.eligible(S.extract(V.B2, "confirm"), V.detector_ids(V.B2)))):
        check(frame, base)
        c, y = c1_mean(frame), frame.y.to_numpy(float)
        g = (V.pred(frame, c) - y) ** 2 - (np.maximum(frame.comparator.to_numpy(float) + W * c, 300.0) - y) ** 2
        out[name] = {"rows": int(len(g)), "gain_mse": float(g.mean())}
    conf = keep_rows(S.extract(JAN10, "fresh_january"), JAN10, ("EDDF/01", "EGLL/01", "LEBL/01", "LSZH/01"))
    check(conf, base)
    c, y = c1_mean(conf), conf.y.to_numpy(float)
    g = (V.pred(conf, c) - y) ** 2 - (np.maximum(conf.comparator.to_numpy(float) + W * c, 300.0) - y) ** 2
    out["confirmation"] = stats(g, conf)
    mse = 106584 * out["confirmation"]["gain_mse"] / N
    out["planning_full_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    out["rules"] = {"B1_positive": out["B1"]["gain_mse"] > 0, "B2_positive": out["B2"]["gain_mse"] > 0,
                    "confirmation_positive": out["confirmation"]["gain_mse"] > 0,
                    "confirmation_without_best100": out["confirmation"]["without_best100"] > 0,
                    "three_of_four_cells": sum(v > 0 for v in out["confirmation"]["by_cell"].values()) >= 3,
                    "planning_at_least_0_1_s": out["planning_full_s"] >= 0.1}
    out["verdict"] = "pass" if all(out["rules"].values()) else "fail"
    return out


def eham_features(days):
    frames = []
    for d in days:
        for folder in FEATURE_DIRS:
            p = HERE / folder / f"features_{d}.parquet"
            if p.exists():
                frames.append(pd.read_parquet(p, columns=[ID] + list(L.EHAM_TRACK)))
                break
    return pd.concat(frames, ignore_index=True).drop_duplicates(ID)


def average_gain(frame):
    y, b = frame.y.to_numpy(float), frame.comparator.to_numpy(float)
    addon = L.correction("eham_january", frame)
    c1 = L.correction("partial", frame)
    return (np.maximum(b + addon, 300.0) - y) ** 2 - (np.maximum(b + (addon + c1) / 2, 300.0) - y) ** 2


def test2(base):
    dev_ext = pd.concat([S.extract(tuple(S.STAGES["development"]), "development"), S.extract(tuple(S.STAGES["confirm"]), "confirm")],
                        ignore_index=True).drop_duplicates(ID)
    out = {}
    for name, ext, days in (("development", dev_ext, DEV_EHAM), ("confirmation", S.extract(JAN10, "fresh_january"), JAN10)):
        f = keep_rows(ext, days, ("EHAM/01",))
        f = f.merge(eham_features(days), on=ID, how="inner", validate="one_to_one")
        check(f, base)
        out[name] = stats(average_gain(f), f)
        out[name]["days"] = int(f.date.nunique())
    c = out["confirmation"]
    mse = 9752 * c["gain_mse"] / N
    out["planning_full_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    out["rules"] = {"development_positive": out["development"]["gain_mse"] > 0, "confirmation_positive": c["gain_mse"] > 0,
                    "confirmation_without_best100": c["without_best100"] > 0,
                    "seven_of_ten_days": sum(v > 0 for v in c["by_day"].values()) >= 7,
                    "planning_at_least_0_05_s": out["planning_full_s"] >= 0.05}
    out["verdict"] = "pass" if all(out["rules"].values()) else "fail"
    return out


def main():
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    report = {"prereg": "research_480_weights/PREREG_weights.md", "test1_c1_weight_1_25": test1(base), "test2_eham_average": test2(base)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for k in ("test1_c1_weight_1_25", "test2_eham_average"):
        r = report[k]
        print(k, r["verdict"], json.dumps({kk: v for kk, v in r.items() if kk not in ("verdict",)}, default=str)[:900], flush=True)


if __name__ == "__main__":
    main()
