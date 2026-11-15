import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('labserver',Path(__file__).parents[1]/'server.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class LocalRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.patches=[patch.object(s,'RUNS',Path(cls.temp.name)),patch.object(s,'runtime_python',lambda:sys.executable)]
        for p in cls.patches:p.start()
        cls.http=s.ThreadingHTTPServer(('127.0.0.1',0),s.Handler)
        cls.base='http://127.0.0.1:'+str(cls.http.server_port)
        threading.Thread(target=cls.http.serve_forever,daemon=True).start()
    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown();cls.http.server_close()
        for p in cls.patches:p.stop()
        cls.temp.cleanup()
    def request(self,path,payload=None,headers=None):
        h={'Content-Type':'application/json','X-Lab-Token':s.TOKEN};h.update(headers or {})
        request=Request(self.base+path,data=None if payload is None else json.dumps(payload).encode(),headers=h)
        try:
            with urlopen(request,timeout=10) as r:return r.status,json.loads(r.read())
        except HTTPError as e:return e.code,json.loads(e.read())
    def wait_job(self,ident):
        end=time.monotonic()+10
        while time.monotonic()<end:
            status,job=self.request('/api/jobs/'+ident)
            if job['status'] not in ('running','queued'):return job
            time.sleep(.02)
        self.fail('Job did not finish')
    def test_cross_origin_and_missing_token_cannot_execute(self):
        body={'kind':'code','code':'print(123)'}
        self.assertEqual(self.request('/api/run',body,{'Origin':'https://untrusted.example'})[0],403)
        self.assertEqual(self.request('/api/run',body,{'X-Lab-Token':''})[0],403)
    def test_untrusted_host_cannot_read_session(self):
        self.assertEqual(self.request('/api/session',headers={'Host':'untrusted.example'})[0],403)
    def test_static_traversal_and_unknown_source_rejected(self):
        self.assertEqual(self.request('/../server.py')[0],404)
        self.assertEqual(self.request('/api/source?name=../server.py')[0],400)
    def test_command_rejects_shell_injection_and_output_escape(self):
        for c in ['python mechanisms.py; echo unsafe','python lab.py train --output ../escape','python lab.py train --steps 999999','python -c "print(1)"']:
            with self.subTest(c=c),self.assertRaises(ValueError):s.parse_command(c)
    def test_foundation_commands_and_saved_model_paths(self):
        for c in ['python foundations.py graph --epochs 10 --ablate-edges --output outputs/graph-check','python predict_saved.py --model-dir outputs/example --input outputs/example/example-input.json --output outputs/prediction.json']:
            self.assertEqual(s.parse_command(c)[0],sys.executable)
        for c in ['python foundations.py unknown','python predict_saved.py --model-dir ../other','python foundations.py tree --output /tmp/other']:
            with self.assertRaises(ValueError):s.parse_command(c)
        self.assertEqual(self.request('/api/artifact?path=../server.py')[0],404)
        self.assertEqual(self.request('/api/example?path=../')[0],400)
    def test_modern_commands_match_curriculum_and_bound_options(self):
        import modern_curriculum
        for lesson in modern_curriculum.SPECS:
            for command in lesson[4]:
                with self.subTest(command=command):
                    self.assertEqual(s.parse_command(command.replace('{run}','check'))[0],sys.executable)
        for option in ['--beta nan','--beta inf','--steps 1001','--threads 17','--seed -1','--reward invented','--judge invented','--split train','--output ../escape']:
            with self.subTest(option=option),self.assertRaises(ValueError):
                s.parse_command('python modern_lab.py attention '+option)
        for name in ['modern_models.py','modern_evaluation.py','modern_serving.py','modern_lab.py']:
            status,body=self.request('/api/source?name='+name)
            self.assertEqual(status,200);self.assertGreater(len(body['code']),100)
    def test_modern_downloads_have_expected_content(self):
        for path,needle in [('/download/modern.md',b'Build attention'),('/download/research.md',b'Attention Is All You Need')]:
            with urlopen(self.base+path,timeout=10) as response:
                self.assertEqual(response.status,200);self.assertIn(needle,response.read())
    def test_actual_execution_and_saved_output(self):
        status,j=self.request('/api/run',{'kind':'code','code':'print(sum([2, 3, 7]))'})
        self.assertEqual(status,202);done=self.wait_job(j['id'])
        self.assertEqual(done['status'],'succeeded');self.assertIn('\n12\n',done['output'])
        self.assertTrue((Path(self.temp.name)/(j['id']+'.json')).is_file())
        self.assertEqual(self.request('/api/jobs/'+j['id']+'?offset='+str(done['offset']))[1]['output'],'')
    def test_saved_question_and_numeric_examples_are_valid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'outputs/example';out.mkdir(parents=True)
            for value in ['What is the current Orion restore window?', [[0.2,0.5]]]:
                (out/'example-input.json').write_text(json.dumps(value))
                with patch.object(s,'LAB',root):
                    self.assertEqual(self.request('/api/example?path=outputs/example'),(200,value))
    def test_tokenizer_vocabulary_is_not_an_evaluation_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'outputs').mkdir()
            (root/'outputs/vocab.json').write_text(json.dumps({'accuracy':25873,'n':94,'split':5916}))
            (root/'outputs/evaluation.json').write_text(json.dumps({'accuracy':0.5,'schema_valid_rate':1,'n':12,'split':'test','predictions':[]}))
            with patch.object(s,'LAB',root):reports=s.reports()
            self.assertEqual(len(reports),1)
            self.assertEqual(reports[0]['path'],'outputs/evaluation.json')
    def test_failure_reports_actual_traceback(self):
        _,j=self.request('/api/run',{'kind':'code','code':'raise ValueError("intentional test failure")'})
        done=self.wait_job(j['id']);self.assertEqual(done['status'],'failed');self.assertIn('ValueError: intentional test failure',done['output'])
    def test_stop_terminates_running_job(self):
        _,j=self.request('/api/run',{'kind':'code','code':'import time\nprint("started",flush=True)\ntime.sleep(20)'})
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            _,current=self.request('/api/jobs/'+j['id'])
            if 'started\n' in current['output']:break
            time.sleep(.02)
        self.assertEqual(self.request('/api/cancel',{'id':j['id']})[0],200)
        self.assertEqual(self.wait_job(j['id'])['status'],'cancelled')
if __name__=='__main__':unittest.main()
