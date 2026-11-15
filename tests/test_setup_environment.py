"""Environment setup preservation and Windows path regression checks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('setup_environment',Path(__file__).parents[1]/'lab/setup_environment.py')
setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)

class SetupPreservationTests(unittest.TestCase):
    def test_long_path_selects_fresh_environment_without_touching_existing(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/('long-download-folder-'*4);root.mkdir()
            old=root/'.venv';old.mkdir();(old/'keep.txt').write_text('learner environment')
            first=setup.select_fresh_environment(root);second=setup.select_fresh_environment(root)
            self.assertNotEqual(first,second)
            self.assertTrue(first.is_relative_to(root/'.lab-environments'))
            self.assertFalse(first.exists())
            self.assertEqual((old/'keep.txt').read_text(),'learner environment')
    def test_runtime_update_keeps_exact_previous_configuration_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=root/'.runtime-python';config.write_text('previous interpreter\n',encoding='utf-8')
            backup=setup.publish_runtime(root,'new interpreter')
            self.assertEqual(backup.read_text(encoding='utf-8'),'previous interpreter\n')
            self.assertEqual(config.read_text(encoding='utf-8'),'new interpreter\n')

if __name__=='__main__':unittest.main()
