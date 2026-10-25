"""Label-free sequence input audit, January/July 2025 vs ranking 2026.

Check all tensors finite, IDs/order preserved, original builder parity on real
rows, and feature moments within airport/month cells (not only pooled). The
0.25-SD threshold is a review warning, not automatic evidence of harmful drift.
Generate aggregates only; never print or export hidden labels or row-level data.
"""
import argparse
import ast
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from research_260 import sequence_features as F
from research_260.airport_ablation import sha,write_json,log,source_paths


def moments(x,mask):
    rows=mask.any(1)
    out={'rows':len(x),'zero_neighbour_share':float((~rows).mean()),
         'mean_valid_neighbours':float(mask.sum(1).mean()),'fields':{}}
    for j,name in enumerate(F.FIELDS):
        values=x[:,:,j][mask]
        out['fields'][name]={'mean':float(values.mean(dtype=np.float64)),
                             'sd':float(values.std(dtype=np.float64))}
    return out


def original_parity(raw,ids):
    path=Path(__file__).with_name('neighbour_sequence_screen.py')
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('secs','neighbours')]
    columns=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='RAW' for t in n.targets))
    env=dict(np=np,pd=pd,K=24,W=90.,NF=16,RAW=columns,args=SimpleNamespace(project=Path('.')))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    with patch.object(pd,'read_parquet',side_effect=lambda path,columns:raw[columns].copy()):
        expected,em=env['neighbours']({'source_files':['in_memory']},ids)
    x,m=F.neighbours(raw,ids,'origin')
    np.testing.assert_array_equal(expected,x)
    np.testing.assert_array_equal(em,m)
    return {'rows':len(ids),'exact':True,'reference_code_sha256':sha(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report={'note':__doc__,'datasets':{},'warnings':[]}
    for year,cache in [(2025,'months_1_7.parquet'),(2026,'ranking.parquet')]:
        meta=pd.read_parquet(args.release/'data'/cache,columns=['MVT_ID_mvt','hat_aobt','MVT_TIME_UTC_mvt','ADEP_mvt'])
        meta=meta[meta.hat_aobt.eq(1)].reset_index(drop=True)
        paths=source_paths(args.project,[1,7]) if year==2025 else [args.project/'data/ranking.parquet']
        raw=pd.concat([pd.read_parquet(p,columns=F.RAW) for p in paths],ignore_index=True)
        if year==2025:
            sample=meta.MVT_ID_mvt.to_numpy()[::max(1,len(meta)//2048)]
            report['real_row_original_parity']=original_parity(raw,sample)
        cells=meta.ADEP_mvt.astype(str)+'|'+pd.to_datetime(meta.MVT_TIME_UTC_mvt,utc=True).dt.month.astype(str)
        key=str(year); report['datasets'][key]={}
        for mode in ['origin','local']:
            log(f'preflight {year} {mode}')
            x,m=F.neighbours(raw,meta.MVT_ID_mvt.to_numpy(),mode)
            assert x.shape==(len(meta),24,16) and np.isfinite(x).all() and (x[~m]==0).all()
            summary=moments(x,m)
            summary['all_finite']=True
            summary['by_airport_month']={}
            for cell in sorted(cells.unique()):
                idx=np.flatnonzero(cells.to_numpy()==cell)
                summary['by_airport_month'][cell]=moments(x[idx],m[idx])
            report['datasets'][key][mode]=summary
            del x,m;gc.collect()
        del raw;gc.collect()
    for mode in ['origin','local']:
        a,b=[report['datasets'][year][mode]['by_airport_month'] for year in ['2025','2026']]
        if set(a)!=set(b): raise ValueError('airport/month cell mismatch')
        for cell in a:
            for name,av in a[cell]['fields'].items():
                bv=b[cell]['fields'][name]
                delta=bv['mean']-av['mean']
                z=delta/av['sd'] if av['sd']>1e-8 else None
                if (z is not None and abs(z)>.25) or (z is None and abs(delta)>.02):
                    report['warnings'].append({'mode':mode,'cell':cell,'feature':name,'mean_2025':av['mean'],
                                                'mean_2026':bv['mean'],'standardized_mean_shift':z})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    log('PREFLIGHT '+str(report['real_row_original_parity'])+' warnings='+str(report['warnings']))


if __name__=='__main__':main()
