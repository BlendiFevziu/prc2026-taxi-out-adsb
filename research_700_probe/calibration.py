"""Leave-one-month-out calibration of the Rome kind classifier on 2025 Rome fallback rows with delay > 1 h,
and its value-of-information estimate against the realised squared error of a value-of-the-mean prediction."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_500_midnight", ROOT / "research_510_rome_classifier"):
    sys.path.insert(0, str(p))
import rome_classifier as C  # noqa
import rule as R  # noqa
T = "TAXITIME_SEC_mvt"
d = pd.concat([pd.read_parquet(p, columns=R.COLS + [T, "FLIGHT_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt"]) for p in sorted((ROOT / "data").glob("training_2025-*.parquet"))], ignore_index=True)
d = R.annotate(d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.eq("LIRF") & d.AOBT_3_flt.isna()].drop_duplicates(R.ID))
d = d[d.delay > 3600].reset_index(drop=True)
d["kind"] = R.label_kind(d[T], d.delay); d["month"] = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True).dt.month
x = C.features(d); q = np.zeros((len(d), 3)); tt = np.zeros(len(d))
for m in sorted(d.month.unique()):
    a, b = d.month.ne(m).to_numpy(), d.month.eq(m).to_numpy()
    q[b] = C.fit_predict(x[a], d.kind[a], x[b]); tt[b] = d.loc[a & d.kind.eq("normal").to_numpy(), T].median()
mu = np.c_[d.delay.to_numpy(float), tt, 86400 + tt]
y = d[T].to_numpy(float)
pred = (q * mu).sum(1)                        # expected value under the classifier
voi = ((pred[:, None] - mu) ** 2 * q).sum(1)  # expected SE of that prediction, as the classifier believes
real = (pred - y) ** 2
band = pd.cut(d.delay / 3600, [1, 2, 4, 8, 14, 100]).astype(str)
out = pd.DataFrame({"band": band, "cross": d.crossing.clip(0, 1), "kind": d.kind, "qc": q[:, 0], "qn": q[:, 1], "qs": q[:, 2], "voi": voi, "real": real})
g = out.groupby(["band", "cross"]).agg(n=("kind", "size"), copy=("kind", lambda k: (k == "copy").mean()), q_copy=("qc", "mean"),
    shift=("kind", lambda k: (k == "shift24").mean()), q_shift=("qs", "mean"), voi=("voi", "sum"), real=("real", "sum"))
g["real/voi"] = g.real / g.voi
print(g.round(3).to_string())
print("total believed", f"{voi.sum():.3e}", "realised", f"{real.sum():.3e}", "ratio", round(real.sum() / voi.sum(), 3))
s = out.sort_values("voi", ascending=False)
for k in (10, 50, 100):
    print(f"top {k} by believed voi: believed {s.voi.iloc[:k].sum():.3e} realised {s.real.iloc[:k].sum():.3e}")
