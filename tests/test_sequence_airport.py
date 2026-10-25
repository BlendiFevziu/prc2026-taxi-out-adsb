"""Contracts that the original sequence screen did not test."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import neural_model as NM
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from research_260 import sequence_features as F
from research_260 import sequence_model as S
from research_260.sequence_ensemble_check import postprocess
from research_260.sequence_inference import encode


def fixture():
    t = pd.Timestamp('2025-01-15 12:00', tz='UTC')
    return pd.DataFrame({
        'MVT_ID_mvt':[1,2,3,4], 'FLIGHT_ID_mvt':[100,200,300,400],
        'ADEP_mvt':['EDDF','EDDM','EDDF','EDDM'], 'ADES_mvt':['EGLL','EDDF','LIRF','EHAM'],
        'PHASE_mvt':['DEP','ARR','ARR','DEP'],
        'MVT_TIME_UTC_mvt':[t,t-pd.Timedelta(minutes=1),t+pd.Timedelta(minutes=1),t],
        'BLOCK_TIME_UTC_mvt':[t-pd.Timedelta(minutes=20),t+pd.Timedelta(minutes=9),t+pd.Timedelta(minutes=61),t-pd.Timedelta(minutes=10)],
        'SCHED_TIME_UTC_mvt':[t]*4, 'AOBT_3_flt':[t-pd.Timedelta(minutes=15)]*4,
        'ARVT_3_flt':[t+pd.Timedelta(minutes=2)]*4, 'STAND_mvt':['A']*4,
        'RUNWAY_mvt':['25C']*4, 'AIRCRAFT_TYPE_mvt':['A320']*4, 'AIRCRAFT_OPERATOR_flt':['ABC']*4})


def reference():
    """Load only pure definitions from the unchanged historical script; never its CLI."""
    path = Path(__file__).resolve().parents[1]/'research_260/neighbour_sequence_screen.py'
    tree = ast.parse(path.read_text())
    names = ('secs','neighbours','SeqNet','prepare')
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef,ast.ClassDef)) and n.name in names]
    raw = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='RAW' for t in n.targets))
    env = dict(np=np,pd=pd,torch=torch,nn=nn,NM=NM,K=24,W=90.,NF=16,RAW=raw,args=SimpleNamespace(project=Path('.')))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env


class SequenceAirportTests(unittest.TestCase):
    def test_correct_airport(self):
        raw = fixture()
        for mode, expected in [('origin',1.),('local',1/6)]:
            x,m = F.neighbours(raw,[1],mode)
            self.assertEqual(m.sum(),1)
            self.assertAlmostEqual(float(x[0,:,5][m[0]][0]),expected,places=6)

    def test_original_builder_exact(self):
        raw, ref = fixture(), reference()
        with patch.object(pd,'read_parquet',side_effect=lambda path,columns: raw[columns].copy()):
            a,b = ref['neighbours']({'source_files':['test']},np.array([1,4]))
        x,m = F.neighbours(raw,[1,4],'origin')
        np.testing.assert_array_equal(x,a)
        np.testing.assert_array_equal(m,b)

    def test_no_departure_labels(self):
        raw = fixture()
        for mode in ['origin','local']:
            x,m = F.neighbours(raw,[1,4],mode)
            changed = raw.assign(TAXITIME_SEC_mvt=99999.)
            changed.loc[changed.PHASE_mvt.eq('DEP'),'BLOCK_TIME_UTC_mvt'] = pd.Timestamp('2040-01-01',tz='UTC')
            a,b = F.neighbours(changed,[1,4],mode)
            np.testing.assert_array_equal(x,a)
            np.testing.assert_array_equal(m,b)

    def test_clock_units_and_shifts(self):
        raw = fixture()
        x,m = F.neighbours(raw,[1,4])
        for shifted in [False,True]:
            d = raw.copy()
            for col in ['MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','ARVT_3_flt']:
                d[col] = d[col].dt.as_unit('us')
                if shifted: d[col] += pd.Timedelta(days=90)
            d.FLIGHT_ID_mvt += 1000000
            a,b = F.neighbours(d,[1,4])
            np.testing.assert_array_equal(x,a)
            np.testing.assert_array_equal(m,b)

    def test_query_order_and_empty(self):
        x,m = F.neighbours(fixture(),[1,4])
        a,b = F.neighbours(fixture(),[4,1])
        np.testing.assert_array_equal(x[::-1],a)
        np.testing.assert_array_equal(m[::-1],b)
        x,m = F.neighbours(fixture(),[])
        self.assertEqual(x.shape,(0,24,16))

    def test_window_and_isolation(self):
        raw = fixture()
        raw.loc[1,'MVT_TIME_UTC_mvt'] -= pd.Timedelta(hours=3)
        x,m = F.neighbours(raw,[1])
        self.assertFalse(m.any())
        self.assertFalse(x.any())

    def test_invalid_contracts(self):
        for kwargs in [dict(k=23),dict(k=0),dict(window_min=0),dict(mode='unknown')]:
            with self.assertRaises(ValueError): F.neighbours(fixture(),[1],**kwargs)
        with self.assertRaises(ValueError): F.neighbours(fixture(),[2])
        with self.assertRaises(ValueError): F.neighbours(pd.concat([fixture(),fixture()]),[1])

    def test_reference_network_exact(self):
        ref = reference()
        for seq in [False,True]:
            torch.manual_seed(13)
            old = ref['SeqNet']([(5,4)],3,seq).eval()
            torch.manual_seed(13)
            new = S.SeqNet([(5,4)],3,seq).eval()
            self.assertEqual(list(old.state_dict()),list(new.state_dict()))
            for key,value in old.state_dict().items():
                torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
            xk,xn,xs,ms = torch.zeros(2,1,dtype=torch.int64),torch.ones(2,3),torch.randn(2,24,16),torch.zeros(2,24,dtype=torch.bool)
            with torch.no_grad():
                torch.testing.assert_close(old(xk,xn,xs,ms),new(xk,xn,xs,ms),rtol=0,atol=0)
                self.assertTrue(torch.isfinite(new(xk,xn,xs,ms)).all())

    def test_reference_preprocessing_exact(self):
        train = pd.DataFrame({'cat':['a','b','nan'],'n':[1.,np.nan,4.]})
        valid = pd.DataFrame({'cat':['z','b'],'n':[np.nan,100.]})
        ref = reference()
        a,b,fit = ref['prepare'](train,['cat','n'],['cat'])
        x,n,prep = S.prepare(train,['cat','n'],['cat'])
        np.testing.assert_array_equal(a,x); np.testing.assert_array_equal(b,n)
        a,b,_ = ref['prepare'](valid,['cat','n'],['cat'],fit)
        x,n,_ = S.prepare(valid,['cat','n'],['cat'],prep)
        np.testing.assert_array_equal(a,x); np.testing.assert_array_equal(b,n)

    def test_floor_after_blending_and_gate(self):
        old = postprocess(250.,0.,-100.,0.,1000.)
        new = postprocess(250.,0.,300.,0.,1000.)
        self.assertEqual(old,300.)
        self.assertEqual(new,325.)
        self.assertNotEqual(new,old+.25*(300.-(-100.)))
        self.assertEqual(postprocess(250.,0.,300.,1.,1000.),1000.)

    def test_inference_unseen_categories_keep_research_semantics(self):
        train=pd.DataFrame({'cat':['a','nan'],'n':[1.,2.]})
        _,_,prep=S.prepare(train,['cat','n'],['cat'])
        raw=pd.DataFrame({'cat':['never_seen','a'],'n':[3.,4.]})
        x,_=encode(raw,prep)
        self.assertEqual(x[0,0],prep['karten']['cat']['nan'])
        self.assertEqual(x[1,0],prep['karten']['cat']['a'])


if __name__ == '__main__': unittest.main()
