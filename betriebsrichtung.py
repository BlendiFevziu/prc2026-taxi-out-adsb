"""Betriebsrichtung aus den ANKUENFTEN ableiten. Landungen erfolgen gegen den Wind,
die Anflugrichtung ist also die Betriebsrichtung. Abfluege koennen eine eigene Bahn
benutzen (EDDF 18 ist reine Abflugbahn) und taugen dafuer nicht."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
SP=["PHASE_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","AOBT_3_flt","ADEP_mvt","ADES_mvt",
    "RUNWAY_mvt","TAXITIME_SEC_mvt"]
d=pd.concat([pd.read_parquet(f,columns=SP) for f in sorted(glob.glob('data/training_2025-*.parquet'))],
            ignore_index=True)
for c in ("MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","AOBT_3_flt"):
    d[c]=pd.to_datetime(d[c],utc=True,errors="coerce")
d["ph"]=d.PHASE_mvt.astype(str).str.upper().str[:3]
d["ri"]=pd.to_numeric(d.RUNWAY_mvt.astype(str).str.extract(r"^(\d{2})",expand=False),errors="coerce")
d=d.dropna(subset=["ri","MVT_TIME_UTC_mvt"])

ank=d[d.ph=="ARR"].copy(); ank["flh"]=ank.ADES_mvt.astype(str)
ank["slot"]=ank.MVT_TIME_UTC_mvt.dt.floor("60min")
def stand(s):
    v=s.value_counts(); return pd.Series({"zustand":v.idxmax(),"reinheit":v.iloc[0]/v.sum(),"n":v.sum()})
z=ank.groupby(["flh","slot"]).ri.apply(stand).unstack().reset_index()
z=z[z.n>=5].sort_values(["flh","slot"])
z["vorher"]=z.groupby("flh").zustand.shift()
z["wechsel"]=(z.zustand!=z.vorher)&z.vorher.notna()
z["tag"]=z.slot.dt.date

print(f"{'Flughafen':10s}{'Wechsel/Tag':>12}{'Phase Median':>14}{'Reinheit':>10}{'Richtungen (Anteil)':>34}")
for f_,g in z.groupby("flh"):
    proTag=g.groupby("tag").wechsel.sum().mean()
    ph=g.groupby(g.wechsel.cumsum()).size()*1.0
    v=g.zustand.value_counts(normalize=True).head(3)
    txt=", ".join(f"{int(k):02d}: {p*100:.0f}%" for k,p in v.items())
    print(f"{f_:10s}{proTag:>12.2f}{ph.median():>12.1f} h{g.reinheit.mean()*100:>9.0f}%{txt:>34}")

dep=d[d.ph=="DEP"].copy(); dep["flh"]=dep.ADEP_mvt.astype(str)
dep["slot"]=dep.MVT_TIME_UTC_mvt.dt.floor("60min")
dep["y"]=pd.to_numeric(dep.TAXITIME_SEC_mvt,errors="coerce")
dep["rest"]=(dep.AOBT_3_flt-dep.BLOCK_TIME_UTC_mvt).dt.total_seconds()
dep=dep.merge(z[["flh","slot","zustand","reinheit"]],on=["flh","slot"],how="inner").dropna(subset=["y"])
print(f"\nAbfluege mit abgeleitetem Zustand: {len(dep):,}")
print("\nRollzeit je Betriebsrichtung, nur die beiden haeufigsten je Flughafen:")
print(f"{'Flughafen':10s}{'Richtung A':>18}{'Richtung B':>18}{'Unterschied y':>15}{'Unterschied rest':>18}")
for f_,g in dep.groupby("flh"):
    v=g.zustand.value_counts().head(2)
    if len(v)<2 or v.iloc[1]<2000: continue
    a,b=v.index[0],v.index[1]
    ga,gb=g[g.zustand==a],g[g.zustand==b]
    print(f"{f_:10s}{f'{int(a):02d} (n={len(ga):,})':>18}{f'{int(b):02d} (n={len(gb):,})':>18}"
          f"{ga.y.median()-gb.y.median():>14.0f}s{ga.rest.median()-gb.rest.median():>17.0f}s")

zz=z[z.wechsel][["flh","slot"]].rename(columns={"slot":"wslot"}).sort_values("wslot")
dep=dep.sort_values("MVT_TIME_UTC_mvt")
dep=pd.merge_asof(dep,zz,left_on="MVT_TIME_UTC_mvt",right_on="wslot",by="flh",direction="backward")
dep["seit"]=(dep.MVT_TIME_UTC_mvt-dep.wslot).dt.total_seconds()/60
print("\nStoert ein Richtungswechsel die Zeiten? Nur Flughaefen mit sauberem Zustand (Reinheit ueber 90 %):")
s=dep[dep.reinheit>0.9]
print(f"{'Fenster nach Wechsel':>22}{'n':>10}{'Median y':>11}{'Mittel y':>11}{'Median rest':>13}")
for lo,hi,nm in [(0,60,"0-1 h"),(60,180,"1-3 h"),(180,720,"3-12 h"),(720,10**9,"ueber 12 h")]:
    g=s[(s.seit>=lo)&(s.seit<hi)]
    if len(g): print(f"{nm:>22}{len(g):>10,}{g.y.median():>11.0f}{g.y.mean():>11.0f}{g.rest.median():>13.0f}")
