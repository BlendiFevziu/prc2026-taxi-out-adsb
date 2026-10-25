import sys, numpy as np, pandas as pd, glob
from pathlib import Path
ROOT=Path.cwd(); sys.path[:0]=[str(ROOT),str(ROOT/"research_360_integration")]
import main_model
import harness as H, layers as L
ID,T="MVT_ID_mvt","TAXITIME_SEC_mvt"
v80=pd.read_parquet(".cache/research_640/v80/candidate_v80.parquet").set_index(ID)[T]
base=H.current_base_2026()
own=H.ownership_v62().set_index(ID)
d=own[["airport","owner"]].copy()
d["v80"]=v80.reindex(d.index).to_numpy()
d["base"]=base.reindex(d.index).to_numpy()
d["at_base"]=np.isclose(d.v80,d.base.astype(v80.dtype),atol=1e-6)
d["month"]=pd.to_datetime(pd.read_parquet("data/ranking.parquet",columns=[ID,"MVT_TIME_UTC_mvt","PHASE_mvt"]).query("PHASE_mvt=='DEP'").set_index(ID).MVT_TIME_UTC_mvt.reindex(d.index),utc=True).dt.month
lane=np.where(d.owner.eq("fallback"),"fallback","main")
d["lane"]=lane
print("rows",len(d)," at base:",int(d.at_base.sum()))
g=d.groupby(["lane","airport","month"]).agg(rows=("at_base","size"),at_base=("at_base","sum"))
g["pct_at_base"]=(g.at_base/g.rows*100).round(1)
print(g.to_string())
# coverage categories from the audit cache
a=pd.concat([pd.read_parquet(p,columns=[ID,"airport","category"]) for p in sorted(glob.glob(".cache/research_320/airport_coverage/day_2026-*.parquet"))]).drop_duplicates(ID).set_index(ID)
d["cat"]=a.category.reindex(d.index)
m=d[d.lane.eq("main")&d.at_base]
print("\nmain rows still at base, by ADS-B category:"); print(pd.crosstab(m.airport,m.cat.fillna("none")).to_string())
