"""Rank 2026 rows by the expected squared-error reduction if their label kind were known (value of information).
Rome fallback rows: three kinds from the Rome classifier fitted on all 2025 Rome fallback rows with delay > 1 h.
Other fallback rows: copy vs not, with the 2025 copy share by airport and delay band. Reference: v80 (best scored).
Label-free for 2026. Run from the repository root:  python3 -u -B research_700_probe/inventory.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_500_midnight", ROOT / "research_510_rome_classifier"):
    sys.path.insert(0, str(p))
import rome_classifier as C  # noqa: E402  (imports main_model first)
import rule as R  # noqa: E402
ID, T, N, S0 = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 245.046
EXTRA = ["FLIGHT_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt"]
OUT = ROOT / ".cache/research_700/inventory.parquet"

def load(files):
    d = pd.concat([pd.read_parquet(p, columns=R.COLS + [T] + EXTRA) for p in files], ignore_index=True)
    return R.annotate(d[d.PHASE_mvt.eq("DEP") & d.AOBT_3_flt.isna()].drop_duplicates(ID)).reset_index(drop=True)

tr = load(sorted((ROOT / "data").glob("training_2025-*.parquet")))
tr["kind"] = R.label_kind(tr[T], tr.delay)
te = load([ROOT / "data/ranking.parquet"])
v80 = pd.read_parquet(ROOT / ".cache/research_640/v80/candidate_v80.parquet").set_index(ID)[T]
te["p"] = te[ID].map(v80).astype(float)
assert te.p.notna().all() and te[T].isna().all()

rome_tr = tr[tr.ADEP_mvt.eq("LIRF") & (tr.delay > 3600)].reset_index(drop=True)
rome_te = te[te.ADEP_mvt.eq("LIRF") & (te.delay > 3600)].reset_index(drop=True)
q = C.fit_predict(C.features(rome_tr), rome_tr.kind, C.features(rome_te))
t = float(rome_tr.loc[rome_tr.kind.eq("normal"), T].median())
mu = np.c_[rome_te.delay.to_numpy(float), np.full(len(rome_te), t), np.full(len(rome_te), 86400.0 + t)]
rome_te["q_copy"], rome_te["q_normal"], rome_te["q_shift24"] = q[:, 0], q[:, 1], q[:, 2]
rome_te["voi"] = ((rome_te.p.to_numpy()[:, None] - mu) ** 2 * q).sum(1)
rome_te["group"] = "rome"

oth_tr = tr[~tr.ADEP_mvt.eq("LIRF") & (tr.delay > 1800)].copy()
band = lambda f: pd.cut(f.delay / 3600, [0.5, 1, 2, 4, 8, 1000]).astype(str)
oth_tr["band"] = band(oth_tr)
share = oth_tr.assign(c=oth_tr.kind.eq("copy")).groupby(["ADEP_mvt", "band"]).c.agg(["mean", "size"])
oth = te[~te.ADEP_mvt.eq("LIRF") & (te.delay > 1800)].copy()
oth["band"] = band(oth)
oth = oth.join(share, on=["ADEP_mvt", "band"])
oth["q_copy"] = oth["mean"].fillna(0.0)
oth["voi"] = oth.q_copy * (oth.p - oth.delay) ** 2          # learning "copy" moves p to delay; "not copy" keeps p
oth["group"] = "other_fallback"

inv = pd.concat([rome_te, oth], ignore_index=True).sort_values("voi", ascending=False).reset_index(drop=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
inv.to_parquet(OUT, index=False)
print("rows", len(inv), "by group", inv.group.value_counts().to_dict())
print("2025 copy share outside Rome by airport and band:\n", share.round(3).to_string())
tot = 0.0
for k in (5, 10, 20, 50, 100, 200, 400):
    s = inv.voi.iloc[:k].sum()
    print(f"top {k:4d}: expected SSE reduction {s:.3e}  ->  RMSE {np.sqrt(S0**2 - s/N):.2f}")
print("all:", f"{inv.voi.sum():.3e} -> {np.sqrt(max(S0**2 - inv.voi.sum()/N, 0)):.2f}")
pd.set_option("display.width", 220)
print(inv[["group", "ADEP_mvt", "delay", "crossing", "p", "q_copy", "q_normal", "q_shift24", "voi"]].head(30).round(3).to_string())
