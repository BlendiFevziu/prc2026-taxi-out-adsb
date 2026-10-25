"""Test of research_560_holds/PREREG_hold_nowcast.md. No fit, no upload.
Run from the repository root:  python3 -u -B research_560_holds/hold_nowcast.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ("1_7", "2_8", "3_9", "4_10", "5_11", "6_12")
BANDS = ((0.1, 0.2), (0.2, 1.0000001))
SHRINK, LONG_S, WINDOW_S = 50.0, 2700.0, 7200.0
N, REF, ROWS_2026 = 344841, 249.3908, 4400
OUT = ROOT / "research_560_holds/results/hold_nowcast.json"


def share2(fallback, main):
    """Share of main-lane departures at the airport within +/-2 h whose MVT - AOBT_3 exceeds 45 min (NaN if none)."""
    out = np.full(len(fallback), np.nan)
    for ap, idx in fallback.groupby("airport").groups.items():
        m = main[main.airport.eq(ap)].sort_values("t")
        t, long_ = m.t.to_numpy(float), m.long.to_numpy(float)
        cum = np.concatenate([[0.0], np.cumsum(long_)])
        q = fallback.loc[idx, "t"].to_numpy(float)
        lo, hi = np.searchsorted(t, q - WINDOW_S, "left"), np.searchsorted(t, q + WINDOW_S, "right")
        n = hi - lo
        s = np.where(n > 0, (cum[hi] - cum[lo]) / np.maximum(n, 1), np.nan)
        out[fallback.index.get_indexer(idx)] = s
    return out


def epoch(s):
    return (pd.to_datetime(s, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def band_of(s):
    b = np.full(len(s), -1)
    for i, (lo, hi) in enumerate(BANDS):
        b[(s >= lo) & (s < hi)] = i
    return b


def corrections(rows):
    out = {}
    for i in range(len(BANDS)):
        r = rows[rows.band.eq(i)]
        out[i] = float(len(r) / (len(r) + SHRINK) * (r.y - r.ref).mean()) if len(r) else 0.0
    return out


def main():
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
    raw = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted((ROOT / "data").glob("training_2025-*.parquet"))], ignore_index=True)
    dep = raw[raw.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt")
    main_ = dep[dep.AOBT_3_flt.notna()].copy()
    main_["airport"], main_["t"] = main_.ADEP_mvt.astype(str), epoch(main_.MVT_TIME_UTC_mvt)
    main_["long"] = (main_.t - epoch(main_.AOBT_3_flt)) > LONG_S
    frames = []
    for p in PAIRS:
        f = pd.read_parquet(ROOT / f".cache/research_240/fallback_anchor_production/predictions_{p}.parquet",
                            columns=["MVT_ID_mvt", "y", "prod_anchor", "ADEP_mvt"])
        f["pair"] = p
        frames.append(f)
    fb = pd.concat(frames, ignore_index=True).rename(columns={"prod_anchor": "ref", "ADEP_mvt": "airport"})
    fb = fb[fb.airport.ne("LIRF")].merge(dep[["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]], on="MVT_ID_mvt", how="left").reset_index(drop=True)
    fb["t"] = epoch(fb.MVT_TIME_UTC_mvt)
    fb["share2"] = share2(fb, main_)
    fb["band"] = band_of(fb.share2.to_numpy())
    fb["new"] = fb.ref
    per_pair = {}
    for p in PAIRS:
        corr = corrections(fb[fb.pair.ne(p)])
        m = fb.pair.eq(p) & fb.band.ge(0)
        fb.loc[m, "new"] = np.maximum(fb.loc[m, "ref"] + fb.loc[m, "band"].map(corr), 300.0)
        per_pair[p] = {"corrections": corr}
    y = fb.y.to_numpy(float)
    g = (fb.ref.to_numpy(float) - y) ** 2 - (fb.new.to_numpy(float) - y) ** 2
    keep = np.ones(len(g), bool)
    keep[np.argsort(g, kind="stable")[-20:]] = False
    for p in PAIRS:
        per_pair[p]["gain_per_row"] = float(g[fb.pair.eq(p).to_numpy()].mean())
    r = {"prereg": "research_560_holds/PREREG_hold_nowcast.md", "rows": int(len(fb)),
         "rows_by_band": {str(i): int(fb.band.eq(i).sum()) for i in range(len(BANDS))},
         "hold_rate_by_band": {str(i): float(((fb.y > 5400) & fb.band.eq(i)).sum() / max(fb.band.eq(i).sum(), 1)) for i in range(-1, len(BANDS))},
         "gain_per_row": float(g.mean()), "without_best20": float(g[keep].mean()), "pairs": per_pair,
         "corrections_all_pairs": corrections(fb)}
    mse = ROWS_2026 * r["gain_per_row"] / N
    r["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    r["rules"] = {"pooled_positive": r["gain_per_row"] > 0, "without_best20_positive": r["without_best20"] > 0,
                  "four_of_six_pairs": sum(v["gain_per_row"] > 0 for v in per_pair.values()) >= 4,
                  "planning_at_least_0_1_s": r["planning_s"] >= 0.1}
    r["verdict"] = "pass" if all(r["rules"].values()) else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=2, default=str) + "\n")
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
