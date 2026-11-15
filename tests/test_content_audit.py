"""Regression checks for lesson placement and truthful experiment protocols."""
import contextlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lab'))
sys.path.insert(0, str(ROOT))
import torch
import foundations
import lab
import mechanisms
import modern_lab
from modern_evaluation import CASE_BANKS
import server

class ContentAuditTests(unittest.TestCase):
    def test_every_lesson_ends_in_multiple_distinct_annotated_sources(self):
        course = json.loads((ROOT / 'public/course.json').read_text(encoding='utf-8'))
        for lesson in course['lessons']:
            with self.subTest(lesson=lesson['id']):
                refs = lesson['references']
                self.assertGreaterEqual(len(refs), 2)
                self.assertEqual(len(refs), len({r['url'] for r in refs}))
                footer = re.search(r'<section class="lesson-references">(.*?)</section>$', lesson['html'], re.S)
                self.assertIsNotNone(footer)
                for ref in refs:
                    self.assertTrue(ref['url'].startswith('https://'))
                    self.assertGreater(len(ref['why']), 30)
                    self.assertIn(ref['url'], footer[1])

    def test_foundations_use_only_their_own_worked_example_image(self):
        course = json.loads((ROOT / 'public/course.json').read_text(encoding='utf-8'))
        lessons = [l for l in course['lessons'] if l['track']=='Foundations' and l['kind']=='implementation']
        self.assertEqual(len(lessons), 18)
        for lesson in lessons:
            images = re.findall(r'<img[^>]+src="([^"]+)"', lesson['html'])
            self.assertEqual(images, [f"/images/expanded/foundations/{lesson['id']}.svg"])
            self.assertEqual((ROOT/'public'/images[0].lstrip('/')).read_bytes(),
                             (ROOT/'course'/images[0].lstrip('/')).read_bytes())

    def test_numerical_exercises_compute_the_stated_update_and_precision(self):
        result = mechanisms.math_demo()['one_weight_update']
        self.assertEqual(result['gradient'], -16)
        self.assertAlmostEqual(result['updated_weight'], 2.6, places=5)
        self.assertAlmostEqual(result['new_squared_error'], .64, places=5)
        torch.manual_seed(42); eight = mechanisms.quantize_demo(127)
        torch.manual_seed(42); four = mechanisms.quantize_demo(7)
        self.assertAlmostEqual(eight['scale']*127, four['scale']*7, places=5)
        self.assertGreater(four['mean_squared_error'], eight['mean_squared_error'])
        self.assertEqual(four['int8_weight_bytes_plus_scale'], eight['int8_weight_bytes_plus_scale'])

    def test_existing_evaluation_report_is_preserved_before_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'report.json'; output.write_text('{"evidence":"keep"}')
            with patch.object(lab, 'load') as load, self.assertRaises(ValueError):
                lab.evaluate(SimpleNamespace(output=str(output)))
            load.assert_not_called()
            self.assertEqual(output.read_text(), '{"evidence":"keep"}')

    def test_mechanism_commands_accept_only_supported_modes_and_ranges(self):
        for mode in ['math','quantization','distillation','architectures']:
            server.parse_command('python mechanisms.py --demo '+mode)
        server.parse_command('python mechanisms.py --demo quantization --qmax 7')
        for args in ['--demo wrong','--demo math --qmax 7','--demo quantization --qmax 3','--demo math --extra anything']:
            with self.assertRaises(ValueError):server.parse_command('python mechanisms.py '+args)

    def test_special_evaluations_reject_misleading_test_split_flags(self):
        for module, experiment in [(foundations,'adversarial'),(foundations,'reinforcement'),(modern_lab,'serving')]:
            with tempfile.TemporaryDirectory() as folder:
                output=Path(folder)/'new'
                args=module.parser().parse_args([experiment,'--split','test','--output',str(output)])
                with self.assertRaises(ValueError):module.run(args)
                self.assertFalse(output.exists())

    def test_preference_and_judge_evidence_uses_disjoint_declared_partitions(self):
        pairs=modern_lab.reward_pairs()
        for name, rows in pairs.items():
            for other, other_rows in pairs.items():
                if name!=other:self.assertFalse(set(rows)&set(other_rows))
        self.assertFalse({x['question'] for x in CASE_BANKS['validation']}&{x['question'] for x in CASE_BANKS['test']})
        for split in ['validation','test']:
            with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
                args=modern_lab.parser().parse_args(['reward','--steps','1','--split',split,'--output',folder])
                result=modern_lab.run(args)
                self.assertEqual(json.loads((Path(folder)/'example-input.json').read_text()), pairs[split][0][0])
                self.assertEqual(result['data']['selected_split'],split)
                self.assertEqual(result['data']['partition_sizes'], {'train':4,'validation':2,'test':2})

    def test_reinforce_collects_a_fresh_rollout_for_each_policy_update(self):
        # Exclude warm-start fitting so this counts actual policy optimizer steps.
        original=torch.optim.AdamW.step
        count=[]
        def tracked(optimizer,*args,**kwargs):
            count.append(1)
            return original(optimizer,*args,**kwargs)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            args=modern_lab.parser().parse_args(['reinforce','--steps','2','--output',folder])
            with patch.object(modern_lab,'fit'),patch.object(torch.optim.AdamW,'step',tracked):
                result=modern_lab.run(args)
            self.assertEqual(len(count),2)
            self.assertEqual(result['metrics']['policy_updates_per_rollout'],1)

if __name__=='__main__':unittest.main()
