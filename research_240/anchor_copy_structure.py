"""Block times copied from EOBT_1, LOBT or IOBT in the main lane: structure, error share and oracle bound.

Descriptive, January/July 2025 holdout of the v45 candidate-stage ensemble (trained without January/July). A copy is a
block time within 10 s of the anchor (TAXITIME within 10 s of MVT_TIME - anchor); it only matters when the anchor is
more than 60 s from AOBT_3. Schedule copies (TAXITIME within 10 s of the delay) are reported separately because the
existing gate already targets them. Also the label-free share of main rows whose EOBT_1 is far from AOBT_3 in 2025
January/July and in the 2026 ranking file. No model, no leaderboard feedback.

Run:  python3 -B research_240/anchor_copy_structure.py <output.json>
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
REL = P / ".cache/clock_queue_release_20260912"
cols = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "y", "ersatz_s", "ersatz_eobt_s", "ersatz_lobt_s", "ersatz_iobt_s", "versp_min", "hat_aobt"]
held = pd.read_parquet(REL / "data/months_1_7.parquet", columns=cols)
pred = pd.read_parquet(REL / "candidate/holdout_predictions.parquet", columns=["MVT_ID_mvt", "prediction"])
m = held[held.hat_aobt.eq(1) & held.y.notna()].merge(pred, on="MVT_ID_mvt", validate="one_to_one").reset_index(drop=True)
y, e = m.y.to_numpy(float), (m.y - m.prediction).to_numpy(float)
n, sse = len(m), float(np.sum(e**2))
p = m.ersatz_s.to_numpy(float)
sched = np.abs(y - m.versp_min.to_numpy(float) * 60) <= 10
anchors = {a: m[c].to_numpy(float) for a, c in (("eobt", "ersatz_eobt_s"), ("lobt", "ersatz_lobt_s"), ("iobt", "ersatz_iobt_s"))}
copy = {a: (np.abs(y - v) <= 10) & (np.abs(v - p) > 60) for a, v in anchors.items()}
any_copy = copy["eobt"] | copy["lobt"] | copy["iobt"]
non_sched = any_copy & ~sched
implied = np.where(copy["eobt"], anchors["eobt"], np.where(copy["lobt"], anchors["lobt"], anchors["iobt"]))
gap = np.abs(implied - p)
month = pd.to_datetime(m.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
month = np.where(np.isin(month, [1, 7]), month, month % 12 + 1)
airports = m.ADEP_mvt.astype(str).to_numpy()


def block(mask):
    k = int(mask.sum())
    if not k:
        return {"rows": 0}
    return {"rows": k, "share_of_main": k / n, "ensemble_rmse_s": float(np.sqrt(np.mean(e[mask]**2))), "ensemble_bias_s": float(e[mask].mean()),
            "mse_share": float(np.sum(e[mask]**2) / sse), "oracle_mse_removed": float((np.sum(e[mask]**2) - np.sum((y[mask] - implied[mask])**2)) / n)}


combo = pd.Series(np.char.add(np.char.add(np.char.add(np.where(copy["eobt"], "E", "-"), np.where(copy["lobt"], "L", "-")),
                                          np.where(copy["iobt"], "I", "-")), np.where(sched, "S", "-")))[any_copy]
report = {"note": "descriptive; v45 candidate-stage holdout, January/July 2025 main lane", "main_rows": n, "main_rmse_s": float(np.sqrt(sse / n)),
          "anchor_copies_gap_over_60s": block(any_copy), "of_which_schedule_copies": block(any_copy & sched), "not_schedule_copies": block(non_sched),
          "combination_counts_E_L_I_S": combo.value_counts().to_dict(),
          "not_schedule_copies_by_gap": {label: block(non_sched & lo_hi) for label, lo_hi in
                                        (("60-300 s", gap <= 300), ("300-900 s", (gap > 300) & (gap <= 900)), ("over 900 s", gap > 900))},
          "not_schedule_copies_sign": {"anchor_before_aobt3": block(non_sched & (implied > p)), "anchor_after_aobt3": block(non_sched & (implied < p))},
          "not_schedule_copies_by_month": {str(mo): block(non_sched & (month == mo)) for mo in (1, 7)},
          "not_schedule_copies_by_airport": {a: {**block(non_sched & (airports == a)), "airport_share": float(non_sched[airports == a].mean())} for a in sorted(set(airports))}}
rank = pd.read_parquet(REL / "data/ranking.parquet", columns=["ersatz_s", "ersatz_eobt_s", "hat_aobt"])
rank = rank[rank.hat_aobt.eq(1)]
far = lambda frame, s: float(np.mean(np.abs(frame.ersatz_eobt_s.to_numpy(float) - frame.ersatz_s.to_numpy(float)) > s))
report["label_free_eobt_far_from_aobt3"] = {"2025_jan_jul_over_60s": far(m, 60), "2026_over_60s": far(rank, 60),
                                            "2025_jan_jul_over_300s": far(m, 300), "2026_over_300s": far(rank, 300)}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
fmt = lambda s: "rows 0" if not s["rows"] else f"rows {s['rows']:>6,} ({s['share_of_main']:.2%}) rmse {s['ensemble_rmse_s']:6.1f} bias {s['ensemble_bias_s']:+6.1f} mse share {s['mse_share']:.2%} oracle -{s['oracle_mse_removed']:.0f} MSE"
print("main", n, round(report["main_rmse_s"], 4))
for k in ("anchor_copies_gap_over_60s", "of_which_schedule_copies", "not_schedule_copies"):
    print(f"{k:28s}", fmt(report[k]))
print("combinations E L I S:", report["combination_counts_E_L_I_S"])
for k, s in report["not_schedule_copies_by_gap"].items():
    print(f"  gap {k:10s}", fmt(s))
for k, s in report["not_schedule_copies_sign"].items():
    print(f"  {k:20s}", fmt(s))
for k, s in report["not_schedule_copies_by_month"].items():
    print(f"  month {k}", fmt(s))
for a, s in report["not_schedule_copies_by_airport"].items():
    print(f"  {a}", fmt(s), f"airport share {s['airport_share']:.2%}")
print("label-free EOBT_1 far from AOBT_3:", report["label_free_eobt_far_from_aobt3"])
