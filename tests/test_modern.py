"""Behavioral invariants for the modern curriculum's implementations."""
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'lab'));sys.path.insert(0,str(ROOT))
import torch
from modern_models import Config,TinyTransformer,AdapterLinear,attach_adapters,cache_bytes
from modern_lab import cache_equivalence,export_model,predict,pass_at_k,wilson,dpo_loss,policy_objective,notes,tickets,arithmetic,RewardScorer
from modern_serving import create_server,request_tokens,metrics
from modern_evaluation import parse_choice,elo
torch.set_num_threads(2)


class ModernModels(unittest.TestCase):
    def test_causality_and_cache_match_all_components(self):
        configurations=[Config(),Config(kv_heads=1),Config(kv_heads=2),Config(window=4),Config(position='rope'),Config(position='alibi'),Config(qk_norm=True),Config(norm='rms',swiglu=True),Config(latent=True),Config(depth_attention=True),Config(experts=4)]
        for config in configurations:
            with self.subTest(config=config):
                torch.manual_seed(9);model=TinyTransformer(config).eval();ids=torch.randint(1,255,(1,12));changed=ids.clone();changed[:,6:]=(changed[:,6:]+11)%256
                with torch.no_grad():before=model(ids)[0];after=model(changed)[0]
                self.assertTrue(torch.allclose(before[:,:6],after[:,:6],atol=1e-6,rtol=1e-5))
                self.assertLess(cache_equivalence(model,ids),1e-5)

    def test_cache_storage_ratios_and_window_bound(self):
        sizes=[]
        for heads in [4,2,1]:
            model=TinyTransformer(Config(kv_heads=heads));_,c=model(torch.ones(1,20,dtype=torch.long));sizes.append(cache_bytes(c))
        self.assertEqual(sizes[0],2*sizes[1]);self.assertEqual(sizes[0],4*sizes[2])
        model=TinyTransformer(Config(window=4));_,c=model(torch.ones(1,20,dtype=torch.long));self.assertEqual(c[0]['key'].shape[2],4);self.assertEqual(c[0]['total'],20)

    def test_adapter_starts_at_base_and_packs_real_nibbles(self):
        torch.manual_seed(2);linear=torch.nn.Linear(32,16,bias=False);plain=AdapterLinear(linear);packed=AdapterLinear(linear,quantized=True);x=torch.randn(4,32)
        self.assertTrue(torch.equal(plain(x),linear(x)))
        self.assertEqual(packed.packed.numel()*2,linear.weight.numel())
        self.assertLessEqual(float((packed.weight()-linear.weight).abs().max()),float(packed.scale.max()/2)+1e-6)
        base=packed.packed.clone();packed(x).square().mean().backward();self.assertIsNotNone(packed.b.grad);self.assertTrue(torch.equal(base,packed.packed))

    def test_save_reload_all_adaptation_modes(self):
        for mode in [None,'lora','packed4']:
            model=TinyTransformer()
            if mode:attach_adapters(model,mode=='packed4')
            with tempfile.TemporaryDirectory() as folder:
                export_model(Path(folder),model,'service api:',mode)
                result=predict(folder,'service queue:');self.assertEqual(len(result['token_ids']),20)
                with self.assertRaises(ValueError):predict(folder,[])

    def test_splits_are_disjoint_at_declared_unit(self):
        for dataset in [notes(),tickets(),arithmetic()]:
            sets=[set(v) for v in dataset.values()]
            for i in range(3):
                for j in range(i):self.assertFalse(sets[i]&sets[j])

    def test_preference_gradient_pushes_correct_direction(self):
        good=torch.tensor([0.],requires_grad=True);bad=torch.tensor([0.],requires_grad=True);zero=torch.zeros(1)
        loss=dpo_loss(good,bad,zero,zero);loss.backward();self.assertLess(good.grad.item(),0);self.assertGreater(bad.grad.item(),0)

    def test_group_sequence_ratio_is_length_normalized(self):
        new=torch.tensor([[math.log(1.1),math.log(1.1)]]);old=torch.zeros_like(new);adv=torch.ones(1)
        a=policy_objective(new,old,adv,'gspo');b=policy_objective(new.repeat(1,2),old.repeat(1,2),adv,'gspo')
        self.assertAlmostEqual(a.item(),b.item());self.assertAlmostEqual(a.item(),-1.1,places=6)

    def test_clipping_limits_positive_advantage(self):
        loss=policy_objective(torch.tensor([[math.log(2.)]]),torch.zeros(1,1),torch.ones(1),'ppo')
        self.assertAlmostEqual(loss.item(),-1.2,places=6)

    def test_sampling_metric_and_interval_edges(self):
        self.assertEqual(pass_at_k(8,0,4),0);self.assertEqual(pass_at_k(8,8,4),1)
        self.assertAlmostEqual(pass_at_k(8,1,4),.5)
        self.assertEqual(wilson(0,10)[0],0);self.assertEqual(wilson(10,10)[1],1)
        with self.assertRaises(ValueError):pass_at_k(2,1,3)

    def test_judge_abstention_and_rating_symmetry(self):
        self.assertIsNone(parse_choice('A because it is longer'));self.assertEqual(parse_choice(' b\n'),'B')
        ratings={'a':1000.,'b':1000.};elo(ratings,'a','b');self.assertEqual(sum(ratings.values()),2000);self.assertGreater(ratings['a'],ratings['b'])

    def test_streaming_real_http_and_request_validation(self):
        model=TinyTransformer().eval();server,token=create_server(model);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}'
        try:
            cached=request_tokens(base,token,'service api:',4,True);plain=request_tokens(base,token,'service api:',4,False)
            self.assertEqual(cached['token_ids'],plain['token_ids']);self.assertEqual(cached['output_tokens'],4)
            self.assertGreater(cached['time_to_first_token_seconds'],0);self.assertGreaterEqual(cached['time_per_output_token_seconds'],0)
            with self.assertRaises(HTTPError) as caught:request_tokens(base,'wrong','hello')
            self.assertEqual(caught.exception.code,403)
            for prompt,count in [('',4),('a'*96,4),('hello',True),('hello',33)]:
                with self.subTest(prompt=prompt,count=count),self.assertRaises(HTTPError) as rejected:
                    request_tokens(base,token,prompt,count)
                self.assertEqual(rejected.exception.code,400)
            self.assertIsNone(metrics([.1],.2)['time_per_output_token_seconds'])
        finally:server.shutdown();server.server_close();thread.join(timeout=5)


if __name__=='__main__':unittest.main()
