"""Extreme-row anchor census: which observable timestamp does the airport block time equal?

Pre-registered 14.09.2026 (the team's clone), descriptive, no model, no leaderboard feedback.
Extends research_240/block_source_check.py and inblock_substitution_check.py with anchors that
were not tabulated there: the previous-day departure of the SAME flight number at the same
airport (its AOBT_3, SCHED, EOBT_1, MVT and hidden BLOCK), the previous departure from the same
stand (its AOBT_3, MVT), the flight's own network arrival (ARVT_1, ARVT_3) minus a day, and
the last arrival at the same stand (MVT landing, SCHED, its own AOBT_3 at origin).
Question: are the 'other' fallback extremes (tag_versch 0, not copied, y > 5400 s) explained by
any observable clock within 60 s? If a single anchor covers > 20 % of them, it becomes a
candidate for a rank-time rule; otherwise this line is closed.
All twelve 2025 source files, raw, no cache. Run: python3 -B research_260/extreme_anchor_census.py <out.json>
"""
import glob, json, sys
import numpy as np, pandas as pd

COLS = ["MVT_ID_mvt","FLIGHT_mvt","ADEP_mvt","PHASE_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt","STAND_mvt","TAXITIME_SEC_mvt","LOBT_flt","IOBT_flt","EOBT_1_flt",
        "ARVT_1_flt","AOBT_3_flt","ARVT_3_flt","CALLSIGN_flt"]
frames = [pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob("data/training_2025-*.parquet"))]
d = pd.concat(frames, ignore_index=True)
for c in [c for c in COLS if ("TIME_UTC" in c or c.endswith("_flt")) and c not in ("CALLSIGN_flt",)]:
    d[c] = pd.to_datetime(d[c], utc=True)
def ts(s):
    return (s - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
d["t_mvt"] = ts(d.MVT_TIME_UTC_mvt); d["t_block"] = ts(d.BLOCK_TIME_UTC_mvt); d["t_sched"] = ts(d.SCHED_TIME_UTC_mvt)
for c in ["LOBT_flt","IOBT_flt","EOBT_1_flt","ARVT_1_flt","AOBT_3_flt","ARVT_3_flt"]:
    d["t_"+c[:-4].lower()] = ts(d[c])
dep = d[d.PHASE_mvt == "DEP"].copy(); arr = d[d.PHASE_mvt == "ARR"].copy()
dep["y"] = dep.TAXITIME_SEC_mvt.astype(float)
dep = dep.sort_values("t_mvt").reset_index(drop=True)
# previous-day same flight number at the same airport (previous departure record of that FLIGHT_mvt)
g = dep.sort_values(["ADEP_mvt","FLIGHT_mvt","t_mvt"])
for c in ["t_block","t_aobt_3","t_sched","t_eobt_1","t_mvt","t_lobt","t_iobt"]:
    g["pf_"+c] = g.groupby(["ADEP_mvt","FLIGHT_mvt"])[c].shift(1)
dep = dep.merge(g[["MVT_ID_mvt"]+[c for c in g.columns if c.startswith("pf_")]], on="MVT_ID_mvt", how="left")
# previous departure from same stand (by MVT time)
g = dep.sort_values(["ADEP_mvt","STAND_mvt","t_mvt"])
for c in ["t_aobt_3","t_mvt","t_sched","t_block"]:
    g["ps_"+c] = g.groupby(["ADEP_mvt","STAND_mvt"])[c].shift(1)
dep = dep.merge(g[["MVT_ID_mvt"]+[c for c in g.columns if c.startswith("ps_")]], on="MVT_ID_mvt", how="left")
# last arrival at same stand at or before MVT (in-block time order), with its landing, sched, and origin AOBT
a = arr[["ADEP_mvt","STAND_mvt","t_block","t_mvt","t_sched","t_aobt_3","t_arvt_3"]].dropna(subset=["t_block"]).sort_values("t_block")
a = a.rename(columns={"t_block":"la_t_block","t_mvt":"la_t_mvt","t_sched":"la_t_sched","t_aobt_3":"la_t_aobt_3","t_arvt_3":"la_t_arvt_3"})
dep = dep.sort_values("t_mvt")
dep = pd.merge_asof(dep, a, left_on="t_mvt", right_on="la_t_block", by=["ADEP_mvt","STAND_mvt"], direction="backward")
anchors = {
 "own_sched":"t_sched","own_lobt":"t_lobt","own_iobt":"t_iobt","own_eobt1":"t_eobt_1","own_aobt3":"t_aobt_3",
 "own_arvt1_minus_1d":None,"own_arvt3_minus_1d":None,
 "prevflight_block":"pf_t_block","prevflight_aobt3":"pf_t_aobt_3","prevflight_sched":"pf_t_sched","prevflight_eobt1":"pf_t_eobt_1","prevflight_mvt":"pf_t_mvt",
 "prevstand_dep_aobt3":"ps_t_aobt_3","prevstand_dep_mvt":"ps_t_mvt","prevstand_dep_block":"ps_t_block",
 "lastarr_inblock":"la_t_block","lastarr_landing":"la_t_mvt","lastarr_sched":"la_t_sched","lastarr_origin_aobt3":"la_t_aobt_3","lastarr_arvt3":"la_t_arvt_3",
 "own_sched_minus_1d":None,"own_sched_plus_1d":None,
}
dep["own_arvt1_minus_1d"] = dep.t_arvt_1 - 86400; dep["own_arvt3_minus_1d"] = dep.t_arvt_3 - 86400
dep["own_sched_minus_1d"] = dep.t_sched - 86400; dep["own_sched_plus_1d"] = dep.t_sched + 86400
copied = (dep.t_block - dep.t_sched).abs() <= 10
same_date = dep.MVT_TIME_UTC_mvt.dt.floor("D") == dep.SCHED_TIME_UTC_mvt.dt.floor("D")
fb = dep.t_aobt_3.isna()
groups = {
 "all": np.ones(len(dep), bool),
 "fallback_all": fb.values,
 "fallback_extreme_other": (fb & (dep.y > 5400) & ~copied & same_date).values,
 "fallback_extreme_not_copied": (fb & (dep.y > 5400) & ~copied).values,
 "main_neither_over_1800": (~fb & ((dep.t_aobt_3 - dep.t_block) > 1800) & ~copied).values,
 "main_over_1h": (~fb & (dep.y > 3600) & ~copied).values,
}
out = {"rows": int(len(dep)), "groups": {}}
for gname, m in groups.items():
    sub = dep[m]; res = {"rows": int(len(sub))}
    best = None
    for an, col in anchors.items():
        v = sub[col] if col else sub[an]
        diff = (sub.t_block - v).abs()
        r = {"avail": float(v.notna().mean()), "within_10s": float((diff <= 10).mean()), "within_60s": float((diff <= 60).mean()), "within_600s": float((diff <= 600).mean())}
        res[an] = r
    # any anchor within 60 s
    mats = np.column_stack([((sub.t_block - (sub[c] if c else sub[a])).abs() <= 60).fillna(False).values for a, c in anchors.items()])
    res["any_anchor_within_60s"] = float(mats.any(1).mean())
    res["se_share_of_fallback"] = None
    out["groups"][gname] = res
# squared-error mass: how much of fallback SSE (vs a 900 s constant) sits in each group
fbsub = dep[fb]; sse = ((fbsub.y - 900) ** 2)
for gname in ["fallback_extreme_other", "fallback_extreme_not_copied"]:
    out["groups"][gname]["share_of_fallback_sse_vs_900"] = float(sse[groups[gname][fb.values]].sum() / sse.sum())
# per-airport for the 'other' extremes
m = groups["fallback_extreme_other"]; sub = dep[m]
out["fallback_extreme_other_by_airport"] = sub.groupby("ADEP_mvt").agg(rows=("y","size"), median_y_h=("y", lambda s: float(s.median()/3600))).to_dict("index")
# top 30 raw rows for eyeballing
cols = ["MVT_ID_mvt","ADEP_mvt","FLIGHT_mvt","CALLSIGN_flt","STAND_mvt","y","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","LOBT_flt","IOBT_flt","EOBT_1_flt","ARVT_1_flt","ARVT_3_flt"]
top = sub.sort_values("y", ascending=False).head(30)[cols].copy()
for c in cols[6:]: top[c] = top[c].astype(str)
out["top30_other_extremes"] = top.to_dict("records")
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
print(json.dumps({k: {a: v for a, v in r.items() if a in ("rows","any_anchor_within_60s","share_of_fallback_sse_vs_900")} for k, r in out["groups"].items()}, indent=1))
