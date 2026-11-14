import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from lab import collate, parse_label, read_rows, retrieve
from scratch_lm import TinyLM, ExperimentalLM
from mechanisms import ToySSM, ToyMoE


class LabTests(unittest.TestCase):
    def test_splits_are_disjoint_and_balanced(self):
        sets = []
        for split in ["train", "validation", "test"]:
            rows = read_rows(split)
            keys = [r["text"].casefold() for r in rows]
            self.assertEqual(len(keys), len(set(keys)))
            counts = [sum(r["label"] == label for r in rows)
                      for label in ["environment", "memory", "data"]]
            self.assertEqual(len(set(counts)), 1)
            sets.append(set(keys))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_strict_schema(self):
        self.assertEqual(parse_label('{"label":"data"}'), "data")
        for bad in ['data', '{"label":"other"}', '{"label":"data","extra":1}', '[]']:
            self.assertIsNone(parse_label(bad))

    def test_padding_does_not_mask_real_eos(self):
        b = collate([{"input_ids":[5,2],"labels":[-100,2]},
                     {"input_ids":[5,6,2],"labels":[-100,6,2]}], 2, "cpu")
        self.assertEqual(b["labels"][0].tolist(), [-100,2,-100])
        self.assertEqual(b["attention_mask"][0].tolist(), [1,1,0])

    def test_causal_mask_prevents_future_leakage(self):
        torch.manual_seed(42)
        model = TinyLM().eval()
        a = torch.tensor([[3,4,5,6]])
        b = torch.tensor([[3,4,99,99]])
        with torch.no_grad():
            torch.testing.assert_close(model(a)[:,:2], model(b)[:,:2])

    def test_experimental_models_are_causal_and_trainable(self):
        for kind in ["ssm", "moe", "hybrid"]:
            model = ExperimentalLM(kind).eval()
            a, b = torch.tensor([[3,4,5,6]]), torch.tensor([[3,4,99,99]])
            torch.testing.assert_close(model(a)[:,:2], model(b)[:,:2])
            model(a).square().mean().backward()
            self.assertIsNotNone(model.token.weight.grad)

    def test_streaming_state_matches_whole_sequence(self):
        model = ToySSM()
        x = torch.randn(2,12,16)
        whole, _ = model(x)
        first, state = model(x[:,:6])
        second, _ = model(x[:,6:], state)
        torch.testing.assert_close(whole, torch.cat([first,second], dim=1))

    def test_moe_routes_each_token_once_and_backpropagates(self):
        model = ToyMoE()
        x = torch.randn(2,12,16)
        y, counts = model(x)
        self.assertEqual(int(counts.sum()), 24)
        y.square().mean().backward()
        self.assertIsNotNone(model.router.weight.grad)

    def test_retrieval_has_explicit_no_match(self):
        docs = [{"id":"1", "text":"CUDA memory"}, {"id":"2", "text":"Python import"}]
        self.assertEqual(retrieve("CUDA", docs)[0]["id"], "1")
        self.assertEqual(retrieve("banana", docs), [])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
