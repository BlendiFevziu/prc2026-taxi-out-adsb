import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT=Path.cwd(); sys.path[:0]=[str(ROOT),str(ROOT/"research_360_integration")]
import main_model  # torch first
import harness as H, layers as L
c=H.components_2025(); c["base"]=L.compose_base(c)
m=pd.DataFrame({"id":c.MVT_ID_mvt,"y":c.y.astype(float),"p":c.base.astype(float),"ap":c.airport,"lane":"main"})
f=pd.read_parquet(".cache/research_240/fallback_anchor_production/predictions_1_7.parquet")
f=pd.DataFrame({"id":f.MVT_ID_mvt,"y":f.y.astype(float),"p":f.prod_anchor.astype(float),"ap":f.ADEP_mvt,"lane":"fallback"})
d=pd.concat([m,f],ignore_index=True); d["se"]=(d.p-d.y)**2; d["err"]=d.p-d.y
N=len(d); tot=d.se.sum()
print(f"rows {N}  RMSE {np.sqrt(tot/N):.2f}")
def gain_if_zero(mask):  # RMSE if these rows were perfect
    return np.sqrt(tot/N)-np.sqrt((tot-d.se[mask].sum())/N)
s=d.se.sort_values(ascending=False)
for k in (10,30,100,300,1000,3000):
    idx=s.index[:k]; print(f"top {k:5d} rows: {d.se[idx].sum()/tot*100:5.1f}% of SE, RMSE gain if fixed {gain_if_zero(d.index.isin(idx)):6.2f} s")
for lo,hi in ((0,600),(600,1800),(1800,3600),(3600,10800),(10800,1e9)):
    mk=d.err.abs().between(lo,hi,inclusive="left")
    print(f"|err| {lo:>6}-{hi:<10.0f} rows {mk.sum():7d}  SE {d.se[mk].sum()/tot*100:5.1f}%  gain if fixed {gain_if_zero(mk):6.2f} s")
print("\nSE share by lane x airport (top 12):")
g=d.groupby(["lane","ap"]).agg(rows=("se","size"),se=("se","sum"),big=("err",lambda e:(e.abs()>3600).sum()))
g["share%"]=g.se/tot*100; g["gain_if_fixed_s"]=[gain_if_zero((d.lane==a)&(d.ap==b)) for a,b in g.index]
print(g.sort_values("se",ascending=False).head(12).round(2).to_string())
big=d[d.err.abs()>3600]
print("\nrows with |err|>1h:",len(big)," over/under:",(big.err>0).sum(),(big.err<0).sum(), " share SE %.1f%%"%(big.se.sum()/tot*100))
print(big.groupby(["lane","ap"]).size().sort_values(ascending=False).head(10).to_string())

raw=pd.concat([pd.read_parquet(f"data/training_2025-{a}-01_2025-{b}-01.parquet") for a,b in (("01","02"),("07","08"))])
raw=raw[raw.PHASE_mvt.eq("DEP") if "DEP" in set(raw.PHASE_mvt) else raw.index==raw.index].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
top=d.reindex(d.se.sort_values(ascending=False).index[:25]).copy()
cols=["MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","AOBT_3_flt","EOBT_1_flt","AIRCRAFT_OPERATOR_flt","STAND_mvt","FLIGHT_TYPE_flt"]
r=raw.reindex(top.id)[cols].reset_index(drop=True)
top=top.reset_index(drop=True)
out=pd.concat([top[["lane","ap","y","p"]],r],axis=1)
out["delay_h"]=((pd.to_datetime(out.MVT_TIME_UTC_mvt)-pd.to_datetime(out.SCHED_TIME_UTC_mvt)).dt.total_seconds()/3600).round(1)
out["blk_minus_sched_h"]=((pd.to_datetime(out.BLOCK_TIME_UTC_mvt)-pd.to_datetime(out.SCHED_TIME_UTC_mvt)).dt.total_seconds()/3600).round(2)
out["share%"]=(top.se/tot*100).round(2)
pd.set_option("display.width",250)
print(out[["lane","ap","y","p","share%","delay_h","blk_minus_sched_h","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","MVT_TIME_UTC_mvt","AIRCRAFT_OPERATOR_flt","FLIGHT_TYPE_flt"]].to_string())
