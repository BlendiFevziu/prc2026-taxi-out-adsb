"""Audit of Rome's fallback rows whose pushback crosses local midnight (descriptive, no fit, no upload).

2025: per delay band, the mix of copy / normal / +24 h labels, the shipped fallback's six-pair out-of-sample SSE, and the
SSE of a mix predictor p_copy x delay + p_shift x (86,400 + t) + p_normal x t whose shares come from the OTHER eleven
months (leave one month out) and t is the median normal taxi of those months. 2026: the 42 crossing rows with v68's
value, the band mix and the expected squared error of v68's value against the mix value.
Run from the repository root:  python3 -u -B research_500_midnight/audit_crossing.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule as R  # noqa: E402

BANDS = [0, 2, 4, 8, 12, 18, 40]
OUT = R.ROOT / "research_500_midnight/results/audit_crossing.json"
V68 = R.ROOT / ".cache/research_480/v68/candidate_v68.parquet"
PREDS = R.ROOT / ".cache/research_240/fallback_anchor_production"


def band(delay_s):
    return pd.cut(np.asarray(delay_s) / 3600, BANDS).astype(str)


def rows_2025():
    d = pd.concat([pd.read_parquet(p, columns=R.COLS + [R.T]) for p in sorted((R.ROOT / "data").glob("training_2025-*.parquet"))],
                  ignore_index=True)
    d = R.annotate(d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.eq("LIRF") & d.AOBT_3_flt.isna()])
    d = d[d.crossing >= 1].reset_index(drop=True)
    d["kind"] = R.label_kind(d[R.T], d.delay)
    d["band"] = band(d.delay)
    d["month"] = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True).dt.month
    pred = pd.concat([pd.read_parquet(PREDS / f"predictions_{p}.parquet", columns=[R.ID, "prod_anchor"])
                      for p in ("1_7", "2_8", "3_9", "4_10", "5_11", "6_12")]).set_index(R.ID).prod_anchor
    d["shipped"] = d[R.ID].map(pred).astype(float)
    return d


def mix_value(train, delay, b):
    t = float(train.loc[train.kind.eq("normal"), R.T].median()) if train.kind.eq("normal").any() else 1000.0
    z = train[train.band.eq(b)]
    if len(z) == 0:
        return None, {}
    p = {k: float(z.kind.eq(k).mean()) for k in ("copy", "normal", "shift24")}
    return p["copy"] * delay + p["shift24"] * (86400 + t) + p["normal"] * t, p


def main():
    d = rows_2025()
    d["mix"] = np.nan
    for m in sorted(d.month.unique()):
        train = d[d.month.ne(m)]
        for i in d.index[d.month.eq(m)]:
            d.loc[i, "mix"] = mix_value(train, d.delay[i], d.band[i])[0]
    y = d[R.T].to_numpy(float)
    out = {"rows_2025": int(len(d)), "bands_2025": {}}
    for b, z in d.groupby("band"):
        yy = z[R.T].to_numpy(float)
        out["bands_2025"][b] = {"rows": int(len(z)), "kinds": {k: int(v) for k, v in z.kind.value_counts().items()},
                                "sse_shipped": float(((z.shipped - yy) ** 2).sum()),
                                "sse_mix_lomo": float(((z.mix.fillna(z.shipped) - yy) ** 2).sum())}
    out["sse_shipped"] = float(((d.shipped - y) ** 2).sum())
    out["sse_mix_lomo"] = float(((d.mix.fillna(d.shipped) - y) ** 2).sum())
    out["months_mix_better"] = int(sum(((z.mix.fillna(z.shipped) - z[R.T]) ** 2).sum() < ((z.shipped - z[R.T]) ** 2).sum()
                                       for _, z in d.groupby("month")))
    out["months"] = int(d.month.nunique())

    rk = R.annotate(pd.read_parquet(R.ROOT / "data/ranking.parquet", columns=R.COLS))
    rk = rk[rk.PHASE_mvt.eq("DEP") & rk.ADEP_mvt.eq("LIRF") & rk.AOBT_3_flt.isna() & rk.crossing.ge(1)].reset_index(drop=True)
    rk["band"] = band(rk.delay)
    v68 = pd.read_parquet(V68).set_index(R.ID)[R.T]
    rk["v68"] = rk[R.ID].map(v68).astype(float)
    rows = []
    for r in rk.itertuples(index=False):
        value, p = mix_value(d, r.delay, r.band)
        t = float(d.loc[d.kind.eq("normal"), R.T].median())
        outcomes = {"copy": r.delay, "normal": t, "shift24": 86400 + t}
        exp = lambda v: sum(p.get(k, 0) * (v - o) ** 2 for k, o in outcomes.items())
        rows.append({"id": str(getattr(r, R.ID)), "band": r.band, "delay_h": round(r.delay / 3600, 2), "v68_h": round(r.v68 / 3600, 2),
                     "mix_h": round(value / 3600, 2) if value is not None else None, "p": p,
                     "expected_se_v68": exp(r.v68), "expected_se_mix": exp(value) if value is not None else None})
    out["rows_2026"] = rows
    out["expected_sse_2026"] = {"v68": float(sum(x["expected_se_v68"] for x in rows)),
                                "mix": float(sum(x["expected_se_mix"] if x["expected_se_mix"] is not None else x["expected_se_v68"] for x in rows))}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows_2026"}, indent=1))
    print(pd.DataFrame(rows)[["band", "delay_h", "v68_h", "mix_h"]].sort_values("delay_h").to_string(index=False))


if __name__ == "__main__":
    main()
