"""Pre-registered test of Codex's EHAM January add-on at full weight (research_320/PREREG_eham_weight.md). No fitting.

His code from the read-only checkout of 8d98953 (eham_recovery.extract for the seven features, eham_recovery_screen.inputs
for the input matrices and controls); his 21 models from the hash-checked copy of his cache. The reference rows for the
fresh dates are the research_320 extractions made with Codex's frozen v56 code; his extract() asserts the category and
all 23 summaries of every partial row against its own replay before adding the seven features.
Run from the repository root:  python3 -u -B research_320/eham_weight_experiment.py
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
EH = HERE.parent / "skenderbeu-eham-readonly"
C = HERE / ".cache/codex_eham/eham_recovery"
OUT = HERE / ".cache/research_320/eham_weight"
REPORT = HERE / "research_320/results/eham_weight.json"
DATES = ("2025-01-01", "2025-01-02", "2025-01-05", "2025-01-07", "2025-01-18", "2025-01-19", "2025-01-23")
SEEDS, CONTROLS = (177, 277, 377), (3101, 3102, 3103, 3104, 3105)
ID, V58, N, ELIGIBLE_2026 = "MVT_ID_mvt", 259.9035, 344841, 8968


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    head = subprocess.run(["git", "-C", str(EH), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    if not head.startswith("8d98953"):
        raise SystemExit("the EHAM checkout is not 8d98953")
    rec = json.loads(subprocess.run(["git", "show", "8d98953:research_310/results/eham_recovery_screen.json"], cwd=HERE,
                                    capture_output=True, text=True, check=True).stdout)["model_hashes"]
    if any(sha(C / k) != v for k, v in rec.items()):
        raise SystemExit("a copied EHAM model differs from Codex's record")
    sys.path.insert(0, str(EH))
    import main_model  # noqa: F401
    import joblib
    from research_310 import eham_recovery as ER
    from research_310 import eham_recovery_screen as ES
    from research_300 import tow_guard_test as T
    from research_300 import geometry_compare as GC
    from research_290 import stand_detector as SD
    import offblock_from_tracks as OB

    stand = HERE / ".cache/research_320/stand"
    ref = pd.concat([pd.read_parquet(stand / "confirm.parquet"), pd.read_parquet(stand / "shrink_confirm.parquet")], ignore_index=True)
    ref = ref[ref.date.isin(DATES)].drop_duplicates(ID)
    OUT.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(OUT / "reference.parquet", index=False)
    ER.reference = lambda date: OUT / "reference.parquet"
    ER.OUT = OUT / "features"
    feats = pd.concat([ER.extract(d) for d in DATES], ignore_index=True)
    ctx = ref[[ID, "y", "comparator", "ersatz_s", "schedule", "gate", "hour_sin", "hour_cos", "is_july"]]
    frame = feats.merge(ctx, on=ID, how="left", validate="one_to_one")
    mapping, maps = OB.callsign_mapping(), (SD.stand_table("EHAM"), GC.xplane_table("EHAM")[0])
    protected = set()
    for d in DATES:
        est = T.estimates(d, "EHAM", mapping, maps)
        protected |= set(est.loc[est.guarded.notna(), ID])
    frame["protected"] = frame[ID].isin(protected)
    e = frame[frame.category.eq("partial_ground") & ~frame.protected].reset_index(drop=True)
    y, base = e.y.to_numpy(float), e.comparator.to_numpy(float)

    def corr(arm, seed=None):
        x = ES.inputs(e, arm)
        seeds = SEEDS if seed is None else (seed,)
        return np.mean([joblib.load(C / f"{arm}_{s}.pkl").predict(x) for s in seeds], axis=0)

    def pred(c, s):
        return np.maximum(base + s * c, 300.0)

    def se(p):
        return (p - y) ** 2

    def judge(gain, controls, seed_gains):
        keep = np.ones(len(gain), bool)
        keep[np.argsort(gain, kind="stable")[-100:]] = False
        days = {d: float(gain[e.date.eq(d).to_numpy()].mean()) for d in DATES}
        ci = list(map(float, ES.R.block_bootstrap(gain, (e.airport + "/" + e.date).to_numpy())))
        r = {"rows": int(len(gain)), "gain_mse": float(gain.mean()), "without_best100": float(gain[keep].mean()),
             "airport_day_ci": ci, "by_day": days, "controls": controls, "seeds": seed_gains}
        r["rules"] = {"pooled_positive": r["gain_mse"] > 0, "beats_all_controls": all(r["gain_mse"] > v for v in controls.values()),
                      "best100_removed_positive": r["without_best100"] > 0, "airport_day_ci_positive": ci[0] > 0,
                      "every_seed_positive": all(v > 0 for v in seed_gains.values()),
                      "five_of_seven_days_positive": sum(v > 0 for v in days.values()) >= 5}
        r["verdict"] = "pass" if all(r["rules"].values()) else "fail"
        return r

    real = corr("real")
    w_gain = se(pred(real, 0.5)) - se(pred(real, 1.0))
    w_ctrl = {c: float((se(pred(corr(f"shuffle_{c}"), 0.5)) - se(pred(corr(f"shuffle_{c}"), 1.0))).mean()) for c in CONTROLS}
    w_seed = {s: float((se(pred(corr("real", s), 0.5)) - se(pred(corr("real", s), 1.0))).mean()) for s in SEEDS}
    v_gain = se(base) - se(pred(real, 1.0))
    v_ctrl = {a: float((se(base) - se(pred(corr(a), 1.0))).mean()) for a in ["context"] + [f"shuffle_{c}" for c in CONTROLS]}
    v_seed = {s: float((se(base) - se(pred(corr("real", s), 1.0))).mean()) for s in SEEDS}
    report = {"prereg": "research_320/PREREG_eham_weight.md", "dates": list(DATES), "ehm_rows": int(len(frame)),
              "protected_detector_rows": int(frame.protected.sum()), "eligible_rows": int(len(e)),
              "comparison_1_weight": judge(w_gain, w_ctrl, w_seed), "comparison_2_value": judge(v_gain, v_ctrl, v_seed),
              "reference_at_half_weight_vs_comparator": float((se(base) - se(pred(real, 0.5))).mean())}
    mse = ELIGIBLE_2026 * report["comparison_2_value"]["gain_mse"] / N
    report["planning"] = {"mse_per_scored_row": mse, "full_s": float(V58 - np.sqrt(V58 ** 2 - mse)) if mse > 0 else 0.0,
                          "half_s": float(V58 - np.sqrt(V58 ** 2 - mse / 2)) if mse > 0 else 0.0}
    report["minimum_meaningful_0_1_s"] = report["planning"]["full_s"] >= 0.1
    report["verdict"] = "pass" if (report["comparison_1_weight"]["verdict"] == "pass" and report["comparison_2_value"]["verdict"] == "pass"
                                   and report["minimum_meaningful_0_1_s"]) else "fail"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    for k in ("comparison_1_weight", "comparison_2_value"):
        r = report[k]
        print(k, r["verdict"], json.dumps(r["rules"]))
        print("   ", json.dumps({x: r[x] for x in ("rows", "gain_mse", "without_best100", "airport_day_ci", "by_day", "seeds")}))
        print("    controls", {a: round(v, 1) for a, v in r["controls"].items()})
    print("planning", report["planning"], "| overall verdict", report["verdict"])


if __name__ == "__main__":
    main()
