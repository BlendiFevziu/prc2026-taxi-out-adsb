"""Pre-registered 14 September 2026, before any new fit: local vs origin sequences.

Start from the team's 7095116 recipe. Keep his ADEP grouping as origin control and
change ONLY the grouping to ADEP for departures / ADES for arrivals in local.
Use saved v45 91-column feature tables, research categorical encoding as the team did,
train source months 2-6 and 8-12, evaluate source months 1 and 7, separately too.
Stage A: FLAT / origin SEQ / local SEQ, matched seeds 13 and 21, 12 epochs, same
optimizer, architecture and row order. No feature/weight/epoch tuning on holdout.
Original FLAT/SEQ seed-13 replication must be within 0.5 s of the team's 229.7266 /
224.5488 component RMSE, or investigate before interpreting the local comparison.
The implementation also must pass exact tensor/preprocessing/network parity tests.
Screen: either sequence must beat matched FLAT by >=0.3 s, improve January and
July separately and retain positive MSE gain after dropping 100 best-gain rows.
Prefer local over origin only if it also beats origin in both months and survives
that top-100 check. Otherwise keep origin as an explicitly nonlocal context lead.
Stage B: compare each passing sequence in the exact frozen full v46 main ensemble,
replacing only the 25% neural component. First screen two seeds, then complete all
five seeds (13/21/29/37/45) only if main gain >=0.3 s, overall gain positive, both
months positive, top-100 gain positive, >=7/10 airports positive, and the sequence
beats the matched FLAT substitution. Fallback predictions stay bit-for-bit fixed.
No release from this screen: a changed local grouping still needs independent
source-pair confirmation, a permutation control and airport/month input-drift checks.
Weather is deferred until this comparison is settled. No upload or main merge.

Inputs: original .cache/clock_queue_release_20260912/data/months_*.parquet, which
research_250/release_pipeline.py builds; private data/training_2025-*.parquet.
prepare makes immutable ignored caches. fit saves RAW residuals and weights, no
per-seed prediction floor. Historical floor-before-average scores are additionally
reported only for comparison to the team. One fit at a time, MPS needed for real runs.
Usage: python3 -u -B research_260/airport_ablation.py --release <release> --output
  .cache/sequence_airport_20260914 prepare
  ... fit --mode flat|origin|local --seed 13
Summarize saved raw predictions with the paired full-ensemble follow-up script.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import neural_model as NM
import torch
import main_model as H
import joblib
import numpy as np
import pandas as pd
from research_250 import candidate_features as B
from research_260 import sequence_features as F, sequence_model as S

TRAIN = [2,3,4,5,6,8,9,10,11,12]
VALID = [1,7]
FEATURES = B.main_features(H.MERKMALE_HAUPT)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_json(path,value):
    tmp = path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    tmp.replace(path)


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'),message,flush=True)


def source_paths(project,months):
    paths = [p for p in sorted((project/'data').glob('training_2025-*.parquet')) if int(p.name[14:16]) in months]
    if len(paths)!=len(months): raise ValueError('missing or ambiguous source files')
    return paths


def codes():
    return {p.name:sha(p) for p in [Path(__file__),Path(F.__file__),Path(S.__file__),Path(NM.__file__)]}


def prepare(args):
    out = args.output
    if (out/'manifest.json').exists():
        raise ValueError('already prepared; use a fresh output for a new recipe')
    columns = list(dict.fromkeys(FEATURES+['MVT_ID_mvt','MVT_TIME_UTC_mvt','hat_aobt','y','ersatz_s']))
    frames, provenance = {},{}
    for side,months in [('train',TRAIN),('valid',VALID)]:
        path = args.release/'data'/('months_'+'_'.join(map(str,months))+'.parquet')
        d = pd.read_parquet(path,columns=columns)
        d = d[d.hat_aobt.eq(1)&d.y.notna()].reset_index(drop=True)
        assert d.MVT_ID_mvt.is_unique and np.isfinite(d.ersatz_s).all()
        frames[side]=d
        provenance[side]={'cache':str(path),'sha256':sha(path),'rows':len(d),'source_months':months}
    assert set(frames['train'].MVT_ID_mvt).isdisjoint(frames['valid'].MVT_ID_mvt)
    for col in H.KAT:
        cats = pd.unique(frames['train'][col].astype(str))
        for d in frames.values(): d[col]=pd.Categorical(d[col].astype(str),categories=cats)
    prep = None
    for side,d in frames.items():
        xk,xn,prep = S.prepare(d[FEATURES],FEATURES,H.KAT,prep)
        np.save(out/f'{side}_xk.npy',xk)
        np.save(out/f'{side}_xn.npy',xn)
        np.save(out/f'{side}_residual.npy',(d.y.to_numpy(float)-d.ersatz_s.to_numpy(float)).astype(np.float32))
        d[['MVT_ID_mvt','MVT_TIME_UTC_mvt','ADEP_mvt','y','ersatz_s']].to_parquet(out/f'{side}_rows.parquet',index=False)
        del xk,xn
    joblib.dump(prep,out/'preprocessing.pkl')
    ids = {side:d.MVT_ID_mvt.to_numpy() for side,d in frames.items()}
    del frames,d; gc.collect()
    for side,months in [('train',TRAIN),('valid',VALID)]:
        paths = source_paths(args.project,months)
        provenance[side]['raw_sha256'] = {p.name:sha(p) for p in paths}
        raw = pd.concat([pd.read_parquet(p,columns=F.RAW) for p in paths],ignore_index=True)
        for mode in ['origin','local']:
            log(f'build {side} {mode}')
            x,m = F.neighbours(raw,ids[side],mode)
            np.save(out/f'{side}_{mode}_xs.npy',x)
            np.save(out/f'{side}_{mode}_ms.npy',m)
            provenance[side][mode]={'mean_valid_neighbours':float(m.sum(1).mean()),'no_neighbours':int((m.sum(1)==0).sum()),'all_finite':bool(np.isfinite(x).all())}
            del x,m; gc.collect()
        del raw; gc.collect()
    manifest={'schema':F.SCHEMA,'code_sha256':codes(),'provenance':provenance,'features':FEATURES,
              'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
              'cache_sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.suffix in ('.npy','.pkl','.parquet')}}
    write_json(out/'manifest.json',manifest)
    log('PREPARED '+json.dumps(provenance))


def fit(args):
    out=args.output
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['code_sha256']!=codes(): raise ValueError('code changed since preparation')
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable; do not silently launch a slow CPU run')
    name=f'{args.mode}_{args.seed}'
    if (out/f'{name}.json').exists():
        log(f'{name} already complete; report, do not overwrite')
        return
    required=['preprocessing.pkl','train_xk.npy','train_xn.npy','train_residual.npy','valid_xk.npy','valid_xn.npy','valid_rows.parquet']
    if args.mode!='flat': required += [f'{side}_{args.mode}_{field}.npy' for side in ['train','valid'] for field in ['xs','ms']]
    for file in required:
        if sha(out/file)!=manifest['cache_sha256'][file]: raise ValueError('cache changed: '+file)
    load=lambda name:np.load(out/(name+'.npy'),mmap_mode='c')
    prep=joblib.load(out/'preprocessing.pkl')
    xk,xn,y=load('train_xk'),load('train_xn'),load('train_residual')
    xs=load(f'train_{args.mode}_xs') if args.mode!='flat' else None
    ms=load(f'train_{args.mode}_ms') if args.mode!='flat' else None
    log(f'FIT {name} on MPS, {len(y)} rows')
    started=time.time()
    progress=lambda msg:log(name+' '+msg)
    package=S.fit_net(args.mode!='flat',args.seed,xk,xn,xs,ms,y,prep['kdim'],log=progress)
    package.update(preprocessing=prep,mode=args.mode,feature_schema=F.SCHEMA,manifest_sha256=sha(out/'manifest.json'))
    temporary=out/(name+'.pkl.partial')
    joblib.dump(package,temporary)
    temporary.replace(out/(name+'.pkl'))
    del xk,xn,xs,ms,y; gc.collect(); torch.mps.empty_cache()
    xk,xn=load('valid_xk'),load('valid_xn')
    xs=load(f'valid_{args.mode}_xs') if args.mode!='flat' else None
    ms=load(f'valid_{args.mode}_ms') if args.mode!='flat' else None
    residual=S.predict(package,xk,xn,xs,ms)
    # Serialized state, not a transient live network, is used for both predictions.
    reloaded=joblib.load(out/(name+'.pkl'))
    n=min(4096,len(xk))
    replay=S.predict(reloaded,xk[:n],xn[:n],xs[:n] if xs is not None else None,ms[:n] if ms is not None else None)
    maxdiff=float(np.abs(replay-residual[:n]).max())
    if maxdiff>1e-3: raise ValueError('saved-model replay failed')
    rows=pd.read_parquet(out/'valid_rows.parquet')
    prediction=np.maximum(rows.ersatz_s.to_numpy(float)+residual,300.)
    rows['residual']=residual
    rows.to_parquet(out/(name+'_predictions.parquet'),index=False)
    month=pd.to_datetime(rows.MVT_TIME_UTC_mvt,utc=True).dt.month.to_numpy()
    error=prediction-rows.y.to_numpy(float)
    rmse=float(np.sqrt(np.mean(error**2)))
    historical={'flat':229.72661995910727,'origin':224.54881227985408}
    result={'mode':args.mode,'seed':args.seed,'rows':len(rows),'rmse':rmse,
            'by_month_rmse':{str(m):float(np.sqrt(np.mean(error[month==m]**2))) for m in np.unique(month)},
            'seconds':time.time()-started,'saved_replay_max_abs_s':maxdiff,
            'model_sha256':sha(out/(name+'.pkl')),'predictions_sha256':sha(out/(name+'_predictions.parquet')),
            'manifest_sha256':sha(out/'manifest.json')}
    if args.seed==13 and args.mode in historical:
        result['kevin_replication']={'difference_s':rmse-historical[args.mode],'passed':abs(rmse-historical[args.mode])<=.5}
    write_json(out/(name+'.json'),result)
    log('RESULT '+json.dumps(result))
    if not result.get('kevin_replication',{}).get('passed',True):
        raise RuntimeError('replication gate failed; investigate before further fits')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    f=sub.add_parser('fit'); f.add_argument('--mode',choices=['flat','origin','local'],required=True); f.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    args.project,args.release,args.output=args.project.resolve(),args.release.resolve(),args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'fit':fit}[args.command](args)


if __name__=='__main__': main()
