import pandas as pd, numpy as np, glob
r=pd.read_parquet("data/ranking.parquet")
print("ranking BLOCK non-null:",r.BLOCK_TIME_UTC_mvt.notna().sum(),"of",len(r), " TAXITIME non-null:",r.TAXITIME_SEC_mvt.notna().sum())
print("ranking phases:",r.PHASE_mvt.value_counts().to_dict())
t=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/training_2025-*.parquet"))]).drop_duplicates("MVT_ID_mvt")
t=t[t.PHASE_mvt.eq(r.PHASE_mvt.mode()[0])] if t.PHASE_mvt.nunique()>1 else t
t=t[t.TAXITIME_SEC_mvt.notna()]
for c in ("MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt"): t[c]=pd.to_datetime(t[c],utc=True)
t["fb"]=t.AOBT_3_flt.isna(); t["noflt"]=t.CALLSIGN_flt.isna()
t["delay_h"]=(t.MVT_TIME_UTC_mvt-t.SCHED_TIME_UTC_mvt).dt.total_seconds()/3600
t["bs_h"]=(t.BLOCK_TIME_UTC_mvt-t.SCHED_TIME_UTC_mvt).dt.total_seconds()/3600
y=t.TAXITIME_SEC_mvt
t["kind"]=np.select([y<7200, (y>=36000)&(t.bs_h<-6)&(t.delay_h.abs()<3), (y>=36000)&(t.bs_h.abs()<0.1), (y>=36000)],
                    ["normal","prevday_block","copy_sched","other_giant"],"mid")
print("rows",len(t))
print(pd.crosstab([t.ADEP_mvt],[t.kind]).to_string())
print("\nprevday_block rows by fallback / no-flt:");print(pd.crosstab(t.kind.eq("prevday_block"),[t.fb,t.noflt]).to_string())
pb=t[t.kind.eq("prevday_block")]
print("\nprevday_block: y quantiles", pb.TAXITIME_SEC_mvt.quantile([.1,.5,.9]).round().to_dict())
print("per month:",pb.MVT_TIME_UTC_mvt.dt.month.value_counts().sort_index().to_dict())
print("sched hour UTC:",pb.SCHED_TIME_UTC_mvt.dt.hour.value_counts().sort_index().to_dict())
print("block hour UTC:",pb.BLOCK_TIME_UTC_mvt.dt.hour.value_counts().sort_index().to_dict())
print(pb[["ADEP_mvt","FLIGHT_mvt","AIRCRAFT_TYPE_mvt","STAND_mvt","RUNWAY_mvt","FLIGHT_RULE_mvt","delay_h","bs_h","TAXITIME_SEC_mvt"]].head(30).to_string())
