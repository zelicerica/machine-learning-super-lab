"""Behavioral checks for learning, data leakage, retrieval, and saved interfaces."""
import contextlib, io, json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'lab'));sys.path.insert(0,str(ROOT))
import torch
import neural_builds as n
import rag_workshop as r
from predict_saved import predict
import server

class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_paraphrase_groups_do_not_cross_splits(self):
        data=n.operations();groups={key:{row['group'] for row in rows} for key,rows in data.items()}
        for left,right in [('train','validation'),('train','test'),('validation','test')]:
            self.assertFalse(groups[left]&groups[right])
        self.assertEqual([len(groups[k]) for k in ['train','validation','test']],[56,20,20])
        vocab=n.vocabulary([{'text':'one training sentence','target':'one'}])
        self.assertEqual(n.encode(['never seen'],vocab)[0,1:3].tolist(),[3,3])

    def test_masked_loss_targets_are_hidden_and_padding_excluded(self):
        vocab=n.vocabulary(n.operations()['train']);ids=n.encode(['please restart amber in east'],vocab)
        noisy,targets=n.masked(ids,torch.Generator().manual_seed(1));chosen=targets.ne(-100)
        self.assertTrue(chosen.any());self.assertTrue(noisy[chosen].eq(4).all())
        self.assertTrue(torch.equal(targets[chosen],ids[chosen]));self.assertFalse(chosen[ids.eq(0)].any())

    def test_recurrent_predictions_cannot_read_future_tokens(self):
        torch.manual_seed(1);model=n.RecurrentLM(24).eval();a=torch.tensor([[1,5,6,7,8]]);b=a.clone();b[0,4]=12
        with torch.no_grad():self.assertTrue(torch.equal(model(a)[:,:4],model(b)[:,:4]))

    def test_sequence_encoder_ignores_trailing_padding(self):
        torch.manual_seed(1);model=n.SequenceMapper(24).eval()
        with torch.no_grad():
            a=model.generate(torch.tensor([[1,5,6,2]]));b=model.generate(torch.tensor([[1,5,6,2,0,0]]))
        self.assertTrue(torch.equal(a,b))

    def test_frozen_encoder_stays_fixed_while_head_learns(self):
        vocab=n.vocabulary(n.operations()['train']);model=n.IntentClassifier(n.MaskedEncoder(len(vocab)))
        for p in model.encoder.parameters():p.requires_grad_(False)
        before={k:v.clone() for k,v in model.encoder.state_dict().items()};head=model.head.weight.detach().clone()
        ids=n.encode(['please restart amber in east'],vocab)
        n.fit(model,3,lambda m,g:torch.nn.functional.cross_entropy(m(ids),torch.tensor([2])),.01,2,True)
        self.assertTrue(all(torch.equal(v,model.encoder.state_dict()[k]) for k,v in before.items()))
        self.assertFalse(torch.equal(head,model.head.weight))

    def test_hand_gradient_matches_two_independent_calculations(self):
        result=n.gradient_check();self.assertEqual(result['autograd'],-16);self.assertLess(result['absolute_error'],1e-7)

    def test_retrieval_distinguishes_source_hit_from_answer_evidence(self):
        index=r.make_index(8,0);metrics,rows=r.evaluate(r.QUESTIONS['validation'],index,1,.12)
        self.assertGreater(metrics['source_recall_at_k'],metrics['answer_evidence_recall_at_k'])
        self.assertTrue(any(row['gold_source_found'] and not row['answer_string_in_gold_chunk'] for row in rows))

    def test_current_filter_and_zero_overlap_query(self):
        index=r.make_index(40,8)
        self.assertTrue(all(p['current'] for p in index['passages']))
        self.assertEqual(r.retrieve('xyzzy quux',index),[])
        with self.assertRaises(ValueError):r.chunks(r.DOCS,8,8)
        self.assertTrue(any(not p['current'] for p in r.make_index(40,8,True)['passages']))

    def test_rag_index_reloads_and_returns_evidence_not_fake_answer(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            r.run(r.parser().parse_args(['--output',folder]))
            result=predict(folder,'What is the current Orion restore window?')
            self.assertIn('48 hours',result['generator_prompt']);self.assertNotIn('answer',result)
            with self.assertRaises(ValueError):r.run(r.parser().parse_args(['--output',folder]))

    def test_new_browser_commands_keep_output_boundaries(self):
        for command in ['python neural_builds.py encoder --steps 20 --output outputs/encoder-test',
                        'python rag_workshop.py --mode compare --top-k 2 --output outputs/rag-test']:
            self.assertTrue(server.parse_command(command))
        for command in ['python neural_builds.py bogus --output outputs/x',
                        'python neural_builds.py transfer --adapt-steps 2001 --output outputs/x',
                        'python rag_workshop.py --output ../escape',
                        'python rag_workshop.py --model unknown --output outputs/x']:
            with self.assertRaises(ValueError):server.parse_command(command)

if __name__=='__main__':unittest.main()
