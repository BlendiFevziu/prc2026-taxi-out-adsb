"""Run from the repository root: python3 -u -B research_800_final/label_mechanisms_by_airport.py

Descriptive only, 2025 training labels, no model, no leaderboard information.

For fallback rows (departures without AOBT_3_flt) at every airport: how often is the label a schedule copy
(|y - (MVT - SCHED)| <= 60 s), a +24 h stamp (20 h < y < 28 h), or normal, by delay band and midnight crossing.
Label-kind definition and crossing definition are copied from research_500_midnight/rule.py.
Also: squared-error share a hypothetical 'predict the delay' rule would change, to size what is at stake.
"""
import glob
import numpy as np
import pandas as pd

D = "data/"   # run from the repository root
TZ = {"LIRF": "Europe/Rome", "LEMD": "Europe/Madrid", "LEBL": "Europe/Madrid", "LFPG": "Europe/Paris",
      "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam", "EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin",
      "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt"]
raw = pd.concat([pd.read_parquet(f, columns=cols).assign(src=f[-18:-8]) for f in sorted(glob.glob(D + "training_2025-*.parquet"))])
d = raw[raw.PHASE_mvt.eq("DEP") & raw.AOBT_3_flt.isna() & raw.ADEP_mvt.isin(TZ)].copy()
mvt = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True)
sch = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True)
d["delay"] = (mvt - sch).dt.total_seconds()
d["y"] = d.TAXITIME_SEC_mvt.astype(float)
cross = np.zeros(len(d), int)
for ap, tz in TZ.items():
    m = d.ADEP_mvt.eq(ap).to_numpy()
    a = (mvt[m] - pd.Timedelta(minutes=15)).dt.tz_convert(tz).dt.normalize().dt.tz_localize(None)
    b = sch[m].dt.tz_convert(tz).dt.normalize().dt.tz_localize(None)
    cross[m] = (a - b).dt.days.to_numpy()
d["cross"] = cross
d["kind"] = np.select([(d.y > 20 * 3600) & (d.y < 28 * 3600), (d.y - d.delay).abs() <= 60], ["shift24", "copy"], "normal")
d = d[d.y.notna()]
d["se_copyrule"] = (d.y - d.delay) ** 2          # squared error if we predicted the delay
d["se_med"] = (d.y - d.groupby("ADEP_mvt").y.transform("median")) ** 2   # squared error vs airport median (scale only)
tot = d.se_med.sum()

print(f"fallback rows 2025 (all 12 months): {len(d):,}; labelled kinds: {d.kind.value_counts().to_dict()}\n")
print("== per airport, all fallback rows: n, copy%, shift24%, extreme(y>5400)%, share of fallback SE (about airport median)")
g = d.groupby("ADEP_mvt").agg(n=("y", "size"), copy=("kind", lambda k: (k == "copy").mean() * 100),
                              s24=("kind", lambda k: (k == "shift24").mean() * 100),
                              ext=("y", lambda v: (v > 5400).mean() * 100), se=("se_med", "sum"))
g["se_share%"] = g.se / tot * 100
print(g.drop(columns="se").sort_values("se_share%", ascending=False).round(1).to_string())

bands = [(3600, 7200, "1-2h"), (7200, 14400, "2-4h"), (14400, 28800, "4-8h"), (28800, 50400, "8-14h"), (50400, 1e9, ">14h")]
print("\n== delay band x crossing, airports with >= 15 rows over 1 h delay")
for ap in g.sort_values("se", ascending=False).index:
    a = d[d.ADEP_mvt.eq(ap) & d.delay.gt(3600)]
    if len(a) < 15:
        print(f"-- {ap}: only {len(a)} rows with delay > 1 h"); continue
    print(f"-- {ap}: {len(a)} rows with delay > 1 h")
    for lo, hi, nm in bands:
        for cr, cn in ((a.cross.eq(0), "no crossing"), (a.cross.ge(1), "crossing")):
            s = a[a.delay.gt(lo) & a.delay.le(hi) & cr]
            if len(s) >= 5:
                print(f"   {nm:>5} {cn:11s} n={len(s):4d}  copy {np.mean(s.kind=='copy')*100:5.1f}%  shift24 {np.mean(s.kind=='shift24')*100:5.1f}%  normal {np.mean(s.kind=='normal')*100:5.1f}%  "
                      f"SE if rule=delay vs median: {s.se_copyrule.sum()/max(s.se_med.sum(),1):.2f}x")
