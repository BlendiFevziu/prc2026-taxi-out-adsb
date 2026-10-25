"""Pool the neighbour-sequence screens over source-month pairs (row-weighted, untrimmed).

Pre-registered six-pair rule (14.09.2026): the sequence line advances to a full-ensemble question only if,
pooled over the six pairs, (1) seq_mean beats flat_mean in MSE, (2) at least 4 of 6 pairs gain, (3) January
and July both gain inside pair 1_7, (4) the pooled MSE gain survives removing the 100 most favourable rows,
and (5) the fixed blend 0.75*LGBM + 0.25*net gains at least 0.15 s more RMSE with seq than with flat, pooled.
Run: python3 -B research_260/summarize_seq.py --screen-dir .cache/rmse260/seq-screen --output research_260/results/seq_six_pair.json
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--screen-dir", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
frames = []
for f in sorted(a.screen_dir.glob("seq_screen_predictions_*.parquet")):
    r = json.loads((a.screen_dir / f.name.replace("predictions", "result").replace(".parquet", ".json")).read_text())
    if "decision" not in r or r.get("permute_seed") is not None: continue
    d = pd.read_parquet(f); d["pair"] = "_".join(map(str, r["pair"])); frames.append(d)
d = pd.concat(frames, ignore_index=True)
cols = [c for c in ("lgbm", "flat_mean", "seq_mean", "blend_flat", "blend_seq") if c in d]
def rmse(e): return float(np.sqrt(np.mean(e ** 2)))
out = {"pairs": sorted(d.pair.unique()), "rows": int(len(d)), "pooled": {c: {"rmse": rmse(d[c] - d.y), "mse": float(np.mean((d[c] - d.y) ** 2))} for c in cols}}
gain = (d.flat_mean - d.y) ** 2 - (d.seq_mean - d.y) ** 2
out["seq_vs_flat"] = {"pooled_mse_gain": float(gain.mean()), "pooled_rmse_gain": out["pooled"]["flat_mean"]["rmse"] - out["pooled"]["seq_mean"]["rmse"],
                      "mse_gain_without_top100": float(np.sort(gain.to_numpy())[:-100].mean()),
                      "by_pair_rmse_gain": {p: rmse(g.flat_mean - g.y) - rmse(g.seq_mean - g.y) for p, g in d.groupby("pair")},
                      "by_airport_rmse_gain": {p: rmse(g.flat_mean - g.y) - rmse(g.seq_mean - g.y) for p, g in d.groupby("airport")},
                      "by_month_rmse_gain": {int(p): rmse(g.flat_mean - g.y) - rmse(g.seq_mean - g.y) for p, g in d.groupby("month")}}
fe = np.abs(d.flat_mean - d.y); worst = fe >= fe.quantile(0.99)
out["seq_vs_flat"]["worst1pct_mse_gain"] = float(gain[worst].mean()); out["seq_vs_flat"]["rest99_mse_gain"] = float(gain[~worst].mean())
if "blend_seq" in d:
    out["blend"] = {"lgbm_rmse": out["pooled"]["lgbm"]["rmse"], "blend_flat_rmse": out["pooled"]["blend_flat"]["rmse"], "blend_seq_rmse": out["pooled"]["blend_seq"]["rmse"],
                    "extra_gain_with_seq": out["pooled"]["blend_flat"]["rmse"] - out["pooled"]["blend_seq"]["rmse"],
                    "by_pair_extra": {p: rmse(g.blend_flat - g.y) - rmse(g.blend_seq - g.y) for p, g in d.groupby("pair")}}
p17 = d[d.pair == "1_7"]
jan = rmse(p17[p17.month == 1].flat_mean - p17[p17.month == 1].y) - rmse(p17[p17.month == 1].seq_mean - p17[p17.month == 1].y) if len(p17) else np.nan
jul = rmse(p17[p17.month == 7].flat_mean - p17[p17.month == 7].y) - rmse(p17[p17.month == 7].seq_mean - p17[p17.month == 7].y) if len(p17) else np.nan
bp = out["seq_vs_flat"]["by_pair_rmse_gain"]
out["rule"] = {"pooled_mse_gain_positive": out["seq_vs_flat"]["pooled_mse_gain"] > 0, "pairs_positive": int(sum(v > 0 for v in bp.values())), "pairs": len(bp),
               "january_gain_1_7": jan, "july_gain_1_7": jul, "survives_top100": out["seq_vs_flat"]["mse_gain_without_top100"] > 0,
               "blend_extra_ge_0_15": bool(out.get("blend", {}).get("extra_gain_with_seq", 0) >= 0.15)}
out["rule"]["pass"] = bool(out["rule"]["pooled_mse_gain_positive"] and out["rule"]["pairs_positive"] >= 4 and len(bp) == 6 and jan > 0 and jul > 0 and out["rule"]["survives_top100"] and out["rule"]["blend_extra_ge_0_15"])
a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2))
print(json.dumps({"rows": out["rows"], "pooled": {c: round(v["rmse"], 3) for c, v in out["pooled"].items()}, "seq_vs_flat_rmse_gain": round(out["seq_vs_flat"]["pooled_rmse_gain"], 3), "by_pair": {k: round(v, 3) for k, v in bp.items()}, "blend": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.get("blend", {}).items() if k != "by_pair_extra"}, "rule": out["rule"]}, indent=1))
