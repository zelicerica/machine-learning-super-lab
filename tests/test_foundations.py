import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'lab'))
import foundations as f
from predict_saved import predict

class FoundationContracts(unittest.TestCase):
    def test_splits_are_disjoint_complete_and_reproducible(self):
        a=f.partition(100,42)
        self.assertEqual([len(x) for x in a],[60,20,20])
        self.assertEqual(len(set(np.concatenate(a))),100)
        for x,y in zip(a,f.partition(100,42)):np.testing.assert_array_equal(x,y)

    def test_terminal_reward_does_not_bootstrap(self):
        table=np.ones((16,4))*100
        f.q_update(table,14,1,10,15,True,rate=1)
        self.assertEqual(table[14,1],10)
        self.assertEqual(f.grid_step(14,1),(15,10.0,True))
        self.assertEqual(f.grid_step(0,0),(0,-1.0,False))

    def test_saved_linear_model_reloads_and_predicts_new_values(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            report=f.run(f.parser().parse_args(['linear','--noise','0','--output',d]))
            result=predict(d,[[10,2,1],[20,2,1]])['linear']
            np.testing.assert_allclose(result,[39,59],atol=1e-8)
            self.assertTrue(report['reload_verified'])
            with self.assertRaises(ValueError):f.run(f.parser().parse_args(['linear','--output',d]))

    def test_scaler_fits_only_training_data_and_is_persisted(self):
        import joblib
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            f.run(f.parser().parse_args(['logistic','--output',d]))
            x,y=f.make_classification(n_samples=900,n_features=8,n_informative=5,n_redundant=1,weights=[.65,.35],flip_y=.12,class_sep=1.15,random_state=42)
            train=f.partition(len(x),42,y)[0]
            scaler=joblib.load(Path(d)/'model.joblib')['logistic'].steps[0][1]
            np.testing.assert_allclose(scaler.mean_,x[train].mean(0))
            self.assertFalse(np.allclose(scaler.mean_,x.mean(0)))

    def test_counting_model_reload_is_normalized_for_unknown_context(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            f.run(f.parser().parse_args(['ngram','--output',d]))
            values=predict(d,'previously unseen words')
            self.assertAlmostEqual(sum(values.values()),1)
            self.assertTrue(all(v>0 for v in values.values()))

if __name__=='__main__':unittest.main()
