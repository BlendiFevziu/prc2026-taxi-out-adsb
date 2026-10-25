"""Run from the repository root: python3 -u -B research_800_final/identity_scan.py

Descriptive only. 2025 training labels. Which timestamp does BLOCK_TIME equal, per airport?

Generalises Rome's 'copy' (BLOCK == SCHED): for every airport, every timestamp column c and offset k in
{0, +-1h, +-2h, +-3h, +-24h}, the share of departures with |BLOCK - (c + k)| <= 60 s, and how much of the
airport's squared error (label about the airport-month median) those rows carry.
Reference identity: BLOCK == AOBT_3 (the normal case) is printed separately.
"""
import glob
import numpy as np
import pandas as pd

D = "data/"   # run from the repository root
cols = ["ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt",
        "EOBT_1_flt", "AOBT_3_flt", "TAXITIME_SEC_mvt"]
d = pd.concat([pd.read_parquet(f, columns=cols).assign(m=f[-18:-11]) for f in sorted(glob.glob(D + "training_2025-*.parquet"))])
d = d[d.PHASE_mvt.eq("DEP") & d.BLOCK_TIME_UTC_mvt.notna()].reset_index(drop=True)
y = d.TAXITIME_SEC_mvt.astype(float)
med = y.groupby([d.ADEP_mvt, d.m]).transform("median")
d["se"] = (y - med) ** 2
blk = d.BLOCK_TIME_UTC_mvt
TIMES = ["SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt"]
OFFS = {"0": 0, "+1h": 3600, "-1h": -3600, "+2h": 7200, "-2h": -7200, "+3h": 10800, "-3h": -10800, "+24h": 86400, "-24h": -86400}

print("rows (DEP with BLOCK):", len(d), "| AOBT_3 present:", round(d.AOBT_3_flt.notna().mean() * 100, 1), "%\n")
ref = (blk - d.AOBT_3_flt).dt.total_seconds().abs() <= 60
print("== reference: BLOCK == AOBT_3 (within 60 s), share of rows that have AOBT_3")
print((ref[d.AOBT_3_flt.notna()].groupby(d.ADEP_mvt).mean() * 100).round(1).to_string(), "\n")

res = []
for c in TIMES:
    t = d[c]
    for name, off in OFFS.items():
        diff = (blk - (t + pd.Timedelta(seconds=off))).dt.total_seconds().abs()
        hit = (diff <= 60) & t.notna()
        # exclude trivially equal columns: only keep if not also identical to AOBT_3 (the normal case)
        only = hit & ~ref
        for ap, g in d.groupby("ADEP_mvt"):
            ix = g.index
            n = hit[ix].sum()
            if n:
                res.append((ap, c.replace("_UTC_mvt", "").replace("_flt", ""), name, len(ix), n, n / len(ix) * 100,
                            d.se[ix][hit[ix]].sum() / d.se[ix].sum() * 100, only[ix].sum()))
r = pd.DataFrame(res, columns=["airport", "column", "offset", "rows", "hits", "hit%", "SEshare%", "hits_not_AOBT3"])
r["lift"] = r["SEshare%"] / r["hit%"]
for ap in ["LIRF", "LTFM", "LFPG", "LSZH", "EHAM", "EGLL", "EDDM", "EDDF", "LEBL", "LEMD"]:
    s = r[(r.airport == ap) & (r.hits >= 20)].sort_values("SEshare%", ascending=False).head(6)
    print(f"-- {ap}")
    print(s[["column", "offset", "hits", "hit%", "SEshare%", "lift", "hits_not_AOBT3"]].round(2).to_string(index=False))
