"""Contracts for generated data, independent evaluation, and saved inference."""
import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lab'));sys.path.insert(0,str(ROOT))
import synthetic_data as s
import server
from predict_saved import predict


class SyntheticDataTests(unittest.TestCase):
    def test_augmentation_changes_cannot_change_seed_or_evaluation(self):
        first=s.dataset(42,100,'full',0);second=s.dataset(42,200,'narrow',1)
        for name in ['seed','validation','test']:
            self.assertEqual(first[name],second[name])
        self.assertNotEqual(first['synthetic'],second['synthetic'])
        self.assertEqual(first,s.dataset(42,100,'full',0))

    def test_corruption_changes_labels_not_features(self):
        clean=s.simulate(100,5,'synthetic',label_noise=0)
        bad=s.simulate(100,5,'synthetic',label_noise=1)
        xc,yc=s.arrays(clean);xb,yb=s.arrays(bad)
        np.testing.assert_array_equal(xc,xb)
        np.testing.assert_array_equal(yb,1-yc)
        self.assertTrue(all(r['label_flipped'] for r in bad))

    def test_audit_rejects_duplicates_and_invalid_features(self):
        data=s.dataset(42,100,'full',0)
        s.audit(data)
        duplicate=copy.deepcopy(data);duplicate['validation'][0]=copy.deepcopy(duplicate['seed'][0])
        with self.assertRaises(ValueError):s.audit(duplicate)
        invalid=copy.deepcopy(data);invalid['synthetic'][0]['queue_depth']=float('nan')
        with self.assertRaises(ValueError):s.audit(invalid)

    def test_audit_metadata_cannot_become_features_and_scaler_is_train_only(self):
        data=s.dataset(42,100,'full',0);rows=copy.deepcopy(data['seed'])
        x,y=s.arrays(rows)
        for row in rows:
            row['id']='changed';row['oracle_label']=1-row['oracle_label'];row['label_flipped']=True
        np.testing.assert_array_equal(s.arrays(rows)[0],x)
        model=s.train(rows)
        np.testing.assert_allclose(model.named_steps['standardscaler'].mean_,x.mean(0))
        self.assertEqual(model.n_features_in_,2)

    def test_training_reload_and_fresh_inputs_use_the_saved_contract(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            report=s.run(s.parser().parse_args(['--samples','100','--output',folder]))
            self.assertTrue(report['reload_verified'])
            result=predict(folder,[[50,3],[90,8]])
            self.assertEqual(set(result),{'seed_only','synthetic_only','mixed'})
            self.assertEqual(len(result['mixed']['class']),2)
            for bad in [[[101,3]],[[5]],[[0,float('nan')]],[]]:
                with self.assertRaises(ValueError):predict(folder,bad)
            with self.assertRaises(ValueError):s.run(s.parser().parse_args(['--output',folder]))

    def test_browser_options_remain_bounded(self):
        self.assertTrue(server.parse_command('python synthetic_data.py --samples 2000 --label-noise 0.4 --output outputs/new'))
        for suffix in ['--samples 5001','--label-noise nan','--coverage arbitrary','--output ../escape','--seed -1','--split train']:
            with self.assertRaises(ValueError):server.parse_command('python synthetic_data.py '+suffix)


if __name__=='__main__':unittest.main()
