"""Local course site and real Python lab runner. Standard-library server.
Default binding is loopback only. Code in the workbench runs as your user.
"""
from __future__ import annotations
import argparse
import base64
import importlib.util
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT=Path(__file__).resolve().parent
LAB=ROOT/'lab'; PUBLIC=ROOT/'public'; RUNS=ROOT/'runs'
TOKEN=secrets.token_urlsafe(32)
PREVIEW=False
JOBS={}; JOB_LOCK=threading.Lock()
ALLOWED_SCRIPTS={'lab.py','scratch_lm.py','mechanisms.py','compare_reports.py','foundations.py','predict_saved.py','neural_builds.py','rag_workshop.py','synthetic_data.py'}
ALLOWED_SCRIPTS.add('modern_lab.py')
MODERN_EXPERIMENTS={'attention','components','experts','research','tuning','preference','reward','reinforce','ppo','grpo','gspo','evaluation','judging','serving'}
FOUNDATIONS={'linear','logistic','tree','ensemble','neighbors','kernel','bayes','gaussian','cluster','projection','dense','convolution','graph','autoencoder','variational','adversarial','ngram','reinforcement'}
ALLOWED_SUBCOMMANDS={'doctor','lock','tokenize','infer','eval','train','rag'}
ALLOWED_MODELS={'HuggingFaceTB/SmolLM2-135M-Instruct','Qwen/Qwen2.5-Coder-0.5B-Instruct','HuggingFaceTB/SmolVLM-256M-Instruct'}
TEMPLATES={
 'hello': 'print("Hello, Machine Learning Super Lab!")\n\nweight = 1.0\nlearning_rate = 0.1\nfor step in range(5):\n    prediction = weight * 2\n    loss = (prediction - 6) ** 2\n    gradient = 2 * (prediction - 6) * 2\n    print(f"Step {step}: weight={weight:.4f}, prediction={prediction:.4f}, loss={loss:.4f}")\n    weight -= learning_rate * gradient\n',
 'tensor': 'import torch\n\n# Two examples, three features each.\ninputs = torch.tensor([[1., 2., 3.], [4., 5., 6.]])\nweights = torch.tensor([[0.1], [0.2], [0.3]], requires_grad=True)\npredictions = inputs @ weights\nloss = predictions.square().mean()\nloss.backward()\nprint("Input shape:", inputs.shape)\nprint("Predictions:", predictions.detach())\nprint("Gradient:\\n", weights.grad)\n',
 'state': 'state = 0.0\ninputs = [2.0, 0.0, 0.0, 3.0]\nfor current_input in inputs:\n    state = 0.5 * state + current_input\n    print(f"Input {current_input:.1f} -> state {state:.3f}")\n\n# Change 0.5 to 0.9. What does a slower decay retain?\n',
 'routing': 'import torch\nfrom mechanisms import ToyMoE\n\ntorch.manual_seed(42)\nmodel = ToyMoE()\ninputs = torch.randn(2, 12, 16)\noutput, counts = model(inputs)\nprint("Output shape:", tuple(output.shape))\nprint("Tokens per expert:", counts.tolist())\nprint("Every token routed once:", int(counts.sum()) == 24)\n',
}
TEMPLATES['foundation']='from foundations import parser, run\nimport time\n\n# Change noise, predict the effect, then compare reports.\nargs = parser().parse_args(["linear", "--noise", "8", "--output", f"outputs/editor-{time.time_ns()}"])\nrun(args)\n'

def runtime_python():
    override=os.environ.get('SML_LAB_PYTHON')
    config=ROOT/'.runtime-python'
    if not override and config.is_file(): override=config.read_text().strip()
    if override and Path(override).is_file(): return override
    local=ROOT/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    return str(local) if local.is_file() else sys.executable

def runtime_status():
    exe=runtime_python()
    try:
        p=subprocess.run([exe,'-c','import importlib.util,json,sys; print(json.dumps({"ready": all(importlib.util.find_spec(n) is not None for n in ["torch","transformers","peft"]), "foundations_ready":all(importlib.util.find_spec(n) is not None for n in ["torch","sklearn","matplotlib","scipy"]), "python":sys.version.split()[0]}))'],capture_output=True,text=True,timeout=12)
        data=json.loads(p.stdout) if p.returncode==0 else {'ready':False}
    except Exception: data={'ready':False}
    data['device']='Central processing unit by default'
    data['executable']=exe
    data['local_only']=not PREVIEW
    return data

def output_path(value):
    # Every terminal-supplied output stays in the lab outputs directory.
    p=(LAB/value).resolve(); base=(LAB/'outputs').resolve()
    if not p.is_relative_to(base) or p==base: raise ValueError('Use a path inside outputs/, such as outputs/my-run.')
    return value

def parse_command(command):
    if not isinstance(command,str) or len(command)>3000: raise ValueError('Command is too long.')
    parts=shlex.split(command)
    if not parts or parts[0] not in ('python','python3','py'): raise ValueError('Use a course command starting with python. Type help to see examples.')
    args=parts[1:]
    if args==['-m','unittest','discover','-s','tests','-v']: return [runtime_python(),*args]
    if not args or args[0] not in ALLOWED_SCRIPTS: raise ValueError('Use one of the course scripts. The code editor is available for your own Python.')
    script=args[0]
    if script=='mechanisms.py':
        if len(args)!=1: raise ValueError('mechanisms.py takes no arguments.')
        return [runtime_python(),*args]
    if script=='compare_reports.py':
        if len(args)!=3: raise ValueError('Supply two report paths inside outputs/.')
        for p in args[1:]: output_path(p)
        return [runtime_python(),*args]
    start=1
    if script=='lab.py':
        if len(args)<2 or args[1] not in ALLOWED_SUBCOMMANDS: raise ValueError('Unknown lab command.')
        start=2
    if script=='foundations.py':
        if len(args)<2 or args[1] not in FOUNDATIONS: raise ValueError('Unknown foundation experiment.')
        start=2
    if script=='neural_builds.py':
        if len(args)<2 or args[1] not in {'deep','recurrent','encoder','seq2seq','transfer'}: raise ValueError('Unknown neural build.')
        start=2
    if script=='modern_lab.py':
        if len(args)<2 or args[1] not in MODERN_EXPERIMENTS: raise ValueError('Unknown modern language-model experiment.')
        start=2
    flags={'--model','--device','--threads','--seed','--adapter','--prompt','--max-new-tokens','--temperature','--split','--output','--steps','--batch-size','--max-length','--lr','--architecture'}
    bools={'--retrieve-only','--refresh'}
    if script=='foundations.py':
        flags={'--device','--threads','--seed','--split','--output','--epochs','--steps','--episodes','--depth','--neighbors','--clusters','--components','--latent','--order','--noise','--strength','--beta'}
        bools={'--ablate-edges'}
    if script=='predict_saved.py':
        flags={'--model-dir','--input','--output'};bools=set()
    if script=='neural_builds.py':
        flags={'--output','--steps','--adapt-steps','--labels-per-class','--depth','--lr','--seed','--threads','--split'};bools=set()
    if script=='rag_workshop.py':
        flags={'--output','--mode','--chunk-size','--overlap','--top-k','--threshold','--split','--prompt','--model','--threads','--seed','--max-new-tokens'};bools={'--include-stale'}
    if script=='synthetic_data.py':
        flags={'--output','--samples','--label-noise','--coverage','--seed','--split'};bools=set()
    if script=='modern_lab.py':
        flags={'--output','--steps','--seed','--threads','--split','--beta','--reward','--judge'};bools=set()
    i=start
    while i<len(args):
        flag=args[i]
        if flag in bools: i+=1; continue
        if flag not in flags or i+1==len(args): raise ValueError(f'Unknown or incomplete option: {flag}')
        val=args[i+1]
        if flag in ('--output','--adapter','--model-dir','--input'): output_path(val)
        if flag=='--model' and val not in ALLOWED_MODELS: raise ValueError('Choose one of the three model checkpoints used by this course.')
        limits={'--steps':2000,'--adapt-steps':2000,'--labels-per-class':100,'--top-k':8,'--chunk-size':500,'--batch-size':8,'--max-length':2048,'--max-new-tokens':512,'--threads':32}
        if flag in limits and not 1<=int(val)<=limits[flag]: raise ValueError(f'{flag} must be between 1 and {limits[flag]} for the browser runner.')
        if script=='synthetic_data.py':
            if flag=='--samples' and not 20<=int(val)<=5000:raise ValueError('Use 20–5000 generated rows.')
            if flag=='--label-noise' and not 0<=float(val)<=1:raise ValueError('Label noise must be between zero and one.')
            if flag=='--coverage' and val not in {'full','narrow'}:raise ValueError('Choose full or narrow coverage.')
            if flag=='--seed' and not 0<=int(val)<=2**32-4:raise ValueError('Seed is outside the allowed range.')
            if flag=='--split' and val not in {'validation','test'}:raise ValueError('Choose validation or test.')
        if script=='modern_lab.py':
            if flag=='--steps' and not 1<=int(val)<=1000:raise ValueError('Use 1–1000 updates.')
            if flag=='--threads' and not 1<=int(val)<=16:raise ValueError('Use 1–16 threads.')
            if flag=='--seed' and not 0<=int(val)<2**32:raise ValueError('Use an unsigned 32-bit seed.')
            if flag=='--beta' and not 0<=float(val)<=2:raise ValueError('Use beta between zero and two.')
            if flag=='--split' and val not in {'validation','test'}:raise ValueError('Choose validation or test.')
            if flag=='--reward' and val not in {'exact','zero-prefix'}:raise ValueError('Choose exact or zero-prefix reward.')
            if flag=='--judge' and val not in {'fixture','model'}:raise ValueError('Choose fixture or model judge.')
        i+=2
    return [runtime_python(),*args]

def serializable_job(job,offset=0):
    return {k:v for k,v in job.items() if k not in ('process','output')} | {'output':job['output'][offset:],'offset':len(job['output'])}

def save_job(job):
    RUNS.mkdir(exist_ok=True)
    (RUNS/(job['id']+'.json')).write_text(json.dumps(serializable_job(job),indent=2),encoding='utf-8')

def stop_process(job):
    job['cancel_requested']=True
    p=job.get('process')
    if p and p.poll() is None:
        try:
            if os.name=='nt': subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,timeout=10)
            else: os.killpg(p.pid,signal.SIGTERM)
        except (OSError,subprocess.TimeoutExpired): p.kill()

def append(job,text):
    with JOB_LOCK:
        if len(job['output'])<900000: job['output']+=text

def execute(job,commands):
    job['status']='running';job['started_at']=time.time()
    timer=threading.Timer(1800,lambda:stop_process(job));timer.daemon=True;timer.start()
    try:
        env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['PYTHONIOENCODING']='utf-8'
        env['PYTHONPATH']=str(LAB)+os.pathsep+env.get('PYTHONPATH','')
        for command in commands:
            if job['cancel_requested']: break
            append(job,'\n$ '+shlex.join(['python' if x==command[0] else x for x in command])+'\n')
            options={'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
            p=subprocess.Popen(command,cwd=LAB,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                encoding='utf-8',errors='replace',bufsize=1,env=env,**options)
            job['process']=p
            with p.stdout:
                for line in p.stdout: append(job,line)
            code=p.wait();job['exit_code']=code
            if code!=0: break
        job['status']='cancelled' if job['cancel_requested'] else ('succeeded' if job.get('exit_code')==0 else 'failed')
    except Exception as exc:
        append(job,f'\nRunner error: {type(exc).__name__}: {exc}\n');job['status']='failed'
    finally:
        timer.cancel();job['finished_at']=time.time();job['process']=None
        append(job,'\n['+job['status']+']\n');save_job(job)

def create_job(kind,payload):
    if kind=='command': commands=[parse_command(payload.get('command',''))];label=payload['command']
    elif kind=='code':
        code=payload.get('code','')
        if not isinstance(code,str) or not code.strip() or len(code)>50000: raise ValueError('Enter Python code (maximum 50,000 characters).')
        RUNS.mkdir(exist_ok=True);path=RUNS/('exercise-'+uuid.uuid4().hex[:10]+'.py');path.write_text(code,encoding='utf-8')
        commands=[[runtime_python(),str(path)]];label='Workbench Python exercise'
    elif kind=='setup':
        exe=ROOT/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        commands=[[sys.executable,'-m','venv',str(ROOT/'.venv')],
          [str(exe),'-m','pip','install','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cpu'],
          [str(exe),'-m','pip','install','-r',str(LAB/'requirements.txt'),'-r',str(LAB/'requirements-foundations.txt')]]
        if sys.platform=='darwin': commands[1]=[str(exe),'-m','pip','install','torch==2.6.0']
        label='Install the local learning environment'
    elif kind=='foundations-setup':
        commands=[[runtime_python(),'-m','pip','install','-r',str(LAB/'requirements-foundations.txt')]]
        label='Add foundation packages to the current runtime; preserve installed PyTorch'
    elif kind=='predict':
        folder=output_path(payload.get('model_dir',''))
        values=payload.get('inputs')
        encoded=json.dumps(values,allow_nan=False)
        if len(encoded)>100000: raise ValueError('Keep the example input below 100,000 characters.')
        token=uuid.uuid4().hex[:10];source=LAB/'outputs'/('prediction-input-'+token+'.json')
        source.parent.mkdir(exist_ok=True);source.write_text(encoded)
        commands=[[runtime_python(),'predict_saved.py','--model-dir',folder,'--input',str(source),'--output','outputs/prediction-'+token+'.json']]
        label='Reload saved model and predict a new input'
    elif kind=='vision':
        raw=base64.b64decode(payload.get('image',''),validate=True)
        if not raw or len(raw)>5_000_000: raise ValueError('Choose an image smaller than five megabytes.')
        extension='.png' if raw.startswith(b'\x89PNG\r\n\x1a\n') else '.jpg' if raw.startswith(b'\xff\xd8\xff') else None
        if extension is None: raise ValueError('Choose a portable network graphics or JPEG image.')
        prompt=payload.get('prompt','Describe the image briefly.')
        if not isinstance(prompt,str) or len(prompt)>1000: raise ValueError('Keep the image question below 1,000 characters.')
        folder=LAB/'inputs';folder.mkdir(exist_ok=True)
        path=folder/('image-'+uuid.uuid4().hex[:10]+extension);path.write_bytes(raw)
        commands=[[runtime_python(),'vision_lab.py',str(path),'--prompt',prompt]]
        label='Describe a local image with the vision-language model'
    else: raise ValueError('Unknown runner action.')
    with JOB_LOCK:
        if any(j['status'] in ('queued','running') for j in JOBS.values()): raise RuntimeError('An exercise is already running. Let it finish or stop it first.')
        ident=uuid.uuid4().hex[:12]
        job={'id':ident,'label':label,'kind':kind,'status':'queued','output':'','created_at':time.time(),
             'cancel_requested':False,'lesson_id':payload.get('lesson_id'),'exit_code':None}
        JOBS[ident]=job
    threading.Thread(target=execute,args=(job,commands),daemon=True).start()
    return job

def reports():
    result=[]
    for p in sorted((LAB/'outputs').rglob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        if p.stat().st_size>2_000_000: continue
        try:
            data=json.loads(p.read_text())
            if isinstance(data,dict):
                evaluation=isinstance(data.get('predictions'),list) and data.get('split') in ('validation','test') and isinstance(data.get('accuracy'),(int,float))
                training=isinstance(data.get('validation_loss_before'),(int,float)) and isinstance(data.get('validation_loss_after'),(int,float))
                foundation=data.get('schema')=='superlab.report.v1' and isinstance(data.get('metrics'),dict)
                if evaluation or training or foundation: result.append({'path':str(p.relative_to(LAB)).replace('\\','/'),'data':data})
        except (ValueError,OSError): continue
        if len(result)>=40: break
    return result

class Handler(BaseHTTPRequestHandler):
    server_version='SMLLabLocal/1.0'
    def log_message(self,format,*args):
        if '/api/jobs/' not in self.path: super().log_message(format,*args)
    def valid_host(self):
        host=self.headers.get('Host','').split(':')[0]
        return PREVIEW or host in ('localhost','127.0.0.1','[::1]')
    def respond(self,body,status=200,ctype='application/json; charset=utf-8',headers=None):
        if isinstance(body,(dict,list)): body=json.dumps(body).encode()
        elif isinstance(body,str): body=body.encode()
        self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        for name,value in (headers or {}).items(): self.send_header(name,str(value))
        self.end_headers();self.wfile.write(body)
    def serve_video(self,target):
        data=target.read_bytes();size=len(data);requested=self.headers.get('Range')
        headers={'Accept-Ranges':'bytes'}
        if not requested: return self.respond(data,ctype='video/mp4',headers=headers)
        match=re.fullmatch(r'bytes=(\d{0,18})-(\d{0,18})',requested)
        if not match or not any(match.groups()):
            return self.respond(b'',416,'video/mp4',{**headers,'Content-Range':f'bytes */{size}'})
        left,right=match.groups()
        if not left:
            count=int(right);start=max(0,size-count);end=size-1
        else:
            start=int(left);end=min(int(right),size-1) if right else size-1
        if start>=size or end<start:
            return self.respond(b'',416,'video/mp4',{**headers,'Content-Range':f'bytes */{size}'})
        headers['Content-Range']=f'bytes {start}-{end}/{size}'
        return self.respond(data[start:end+1],206,'video/mp4',headers)
    def do_GET(self):
        if not self.valid_host(): return self.respond({'error':'Invalid host'},403)
        parsed=urlparse(self.path);path=parsed.path
        if path=='/api/session': return self.respond({'token':TOKEN,'runtime':runtime_status(),'templates':TEMPLATES,'workspace':workspace_info()})
        if path=='/api/workspace': return self.respond(workspace_info())
        if path=='/api/runtime': return self.respond(runtime_status())
        if path=='/api/results': return self.respond(reports())
        if path=='/api/example':
            rel=parse_qs(parsed.query).get('path',[''])[0]
            try: target=(LAB/output_path(rel)/'example-input.json').resolve()
            except ValueError: return self.respond({'error':'Invalid output directory'},400)
            if not target.is_relative_to((LAB/'outputs').resolve()) or not target.is_file():return self.respond({'error':'Run this lesson to save its example input first.'},404)
            # Examples may be a JSON string (a question), not only a list/object.
            return self.respond(json.dumps(json.loads(target.read_text())).encode())
        if path=='/api/artifact':
            rel=parse_qs(parsed.query).get('path',[''])[0]
            target=(LAB/rel).resolve()
            if not target.is_relative_to((LAB/'outputs').resolve()) or target.suffix!='.png' or not target.is_file(): return self.respond({'error':'Unknown plot'},404)
            return self.respond(target.read_bytes(),ctype='image/png')
        if path=='/api/jobs': return self.respond([serializable_job(j,len(j['output'])) for j in JOBS.values()][-20:])
        if path.startswith('/api/jobs/'):
            job=JOBS.get(path.split('/')[-1])
            if not job: return self.respond({'error':'Job not found'},404)
            try: offset=max(0,int(parse_qs(parsed.query).get('offset',['0'])[0]))
            except ValueError: offset=0
            return self.respond(serializable_job(job,offset))
        if path=='/api/source':
            name=parse_qs(parsed.query).get('name',['lab.py'])[0]
            if name not in ALLOWED_SCRIPTS|{'vision_lab.py','make_data.py','modern_models.py','modern_evaluation.py','modern_serving.py','qlora_gpu.py'}: return self.respond({'error':'Unknown source file'},400)
            return self.respond({'name':name,'code':(LAB/name).read_text()})
        if path=='/download/guide.md': return self.respond((ROOT/'course/SML-Lab-Zero-to-Hero.md').read_bytes(),ctype='text/markdown; charset=utf-8')
        if path=='/download/foundations.md': return self.respond((ROOT/'course/Expanded-Foundations.md').read_bytes(),ctype='text/markdown; charset=utf-8')
        if path=='/download/builds.md': return self.respond((ROOT/'course/Building-Models-and-Retrieval.md').read_bytes(),ctype='text/markdown; charset=utf-8')
        if path=='/download/modern.md': return self.respond((ROOT/'course/Modern-Language-Engineering.md').read_bytes(),ctype='text/markdown; charset=utf-8')
        if path=='/download/research.md': return self.respond((ROOT/'course/Research-Landmarks.md').read_bytes(),ctype='text/markdown; charset=utf-8')
        target=(PUBLIC/('index.html' if path=='/' else path.lstrip('/'))).resolve()
        if not target.is_relative_to(PUBLIC.resolve()) or not target.is_file(): return self.respond({'error':'Not found'},404)
        if target.suffix.lower()=='.mp4': return self.serve_video(target)
        return self.respond(target.read_bytes(),ctype=mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
    def do_POST(self):
        if not self.valid_host(): return self.respond({'error':'Invalid host'},403)
        origin=self.headers.get('Origin')
        if origin and urlparse(origin).netloc!=self.headers.get('Host'): return self.respond({'error':'Cross-origin request rejected'},403)
        if not secrets.compare_digest(self.headers.get('X-Lab-Token',''),TOKEN): return self.respond({'error':'Refresh this page before running an exercise.'},403)
        if not self.headers.get('Content-Type','').startswith('application/json'): return self.respond({'error':'JSON request required'},415)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=7_000_000: return self.respond({'error':'Invalid request size'},413)
            payload=json.loads(self.rfile.read(length))
            if not isinstance(payload,dict): raise ValueError('Expected an object.')
            path=urlparse(self.path).path
            if path=='/api/run': return self.respond(serializable_job(create_job(payload.get('kind'),payload)),202)
            if path=='/api/cancel':
                job=JOBS.get(payload.get('id'))
                if not job: return self.respond({'error':'Job not found'},404)
                stop_process(job);return self.respond({'ok':True})
            return self.respond({'error':'Unknown action'},404)
        except RuntimeError as exc: return self.respond({'error':str(exc)},409)
        except (ValueError,TypeError,KeyError) as exc: return self.respond({'error':str(exc)},400)

def workspace_info():
    release=json.loads((ROOT/'release.json').read_text(encoding='utf-8'))
    catalog=json.loads((PUBLIC/'course.json').read_text(encoding='utf-8'))
    count=len(catalog.get('lessons',[]))
    return {'release':release['id'],'name':release['name'],'lessons':count,
            'expected_lessons':release['expected_lessons'],'references':len(catalog.get('references',{})),
            'catalog_matches_release':count==release['expected_lessons'],
            'project_folder':str(ROOT),'catalog_file':str(PUBLIC/'course.json'),
            'python':runtime_python(),'local_only':not PREVIEW}

def main():
    global PREVIEW
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=int(os.environ.get('PORT','8765')))
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--no-browser',action='store_true');p.add_argument('--preview',action='store_true')
    p.add_argument('--strictPort',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();PREVIEW=a.preview
    if not a.preview and a.host not in ('127.0.0.1','localhost'): p.error('The learner server binds to localhost only.')
    RUNS.mkdir(exist_ok=True)
    try: server=ThreadingHTTPServer((a.host,a.port),Handler)
    except OSError as exc:
        p.exit(1,f'Cannot start the lab on port {a.port}: {exc}\nAn older copy may still be running. Stop its launcher with Ctrl+C, then start this folder again:\n{ROOT}\nOr use --port 8766 for a separate workspace address.\n')
    url=f'http://localhost:{server.server_port}'
    info=workspace_info()
    print(f'\nMachine Learning Super Lab — {info["release"]}\n{info["lessons"]} lessons | {info["references"]} references\nServing folder: {ROOT}\nOpen: {url}\nKeep this window open. Press Ctrl+C to stop.\n',flush=True)
    if not info['catalog_matches_release']: print('Catalog differs from this release. Run python build_course.py in this folder, then restart.',flush=True)
    if not a.no_browser: threading.Timer(.6,lambda:webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        for j in JOBS.values():
            if j['status'] in ('queued','running'): stop_process(j)
        server.server_close()

if __name__=='__main__': main()
