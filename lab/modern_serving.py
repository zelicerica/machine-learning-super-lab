"""Loopback-only token streaming service and a real HTTP timing experiment.

Start from the lab folder with: python modern_serving.py --model-dir outputs/run
The port is separate from the learning website. No production deployment claim.
"""
import argparse
import json
import secrets
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
import torch
import numpy as np
from modern_models import restore


def metrics(arrivals, end_seconds):
    if not arrivals:raise ValueError("No tokens arrived.")
    intervals=np.diff(arrivals)
    return {"time_to_first_token_seconds":float(arrivals[0]),
            "time_per_output_token_seconds":float((arrivals[-1]-arrivals[0])/(len(arrivals)-1)) if len(arrivals)>1 else None,
            "inter_token_p95_seconds":float(np.percentile(intervals,95)) if len(intervals) else None,
            "last_token_seconds":float(arrivals[-1]),"request_seconds":end_seconds,"output_tokens":len(arrivals)}


def create_server(model, port=0):
    token=secrets.token_urlsafe(24);lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            # A token and loopback bind prevent arbitrary websites from triggering work.
            if self.path!='/generate' or self.headers.get('X-Lab-Token')!=token:
                self.send_error(403);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('Invalid request size.')
                data=json.loads(self.rfile.read(length));prompt=data['prompt'];count=data.get('tokens',16);cached=data.get('cached',True)
                if not isinstance(prompt,str) or type(count) is not int or not 1<=count<=32 or not isinstance(cached,bool):raise ValueError('Invalid generation request.')
                if not 0<len(prompt.encode()) or len(prompt.encode())+count>model.config.context:raise ValueError('Input exceeds context limit.')
            except (ValueError,TypeError,KeyError):self.send_error(400);return
            self.send_response(200);self.send_header('Content-Type','application/x-ndjson');self.send_header('Cache-Control','no-store');self.end_headers()
            from modern_lab import tokens
            try:
                with lock,torch.inference_mode():
                    ids=torch.tensor([tokens(prompt)]);caches=None
                    for i in range(count):
                        logits,caches_new=model(ids[:,-1:] if cached and caches is not None else ids,caches if cached else None)
                        nxt=logits[:,-1].argmax(-1,keepdim=True);ids=torch.cat((ids,nxt),1);caches=caches_new
                        self.wfile.write((json.dumps({'token_id':int(nxt[0,0]),'index':i})+'\n').encode());self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    return server,token


def request_tokens(base,token,prompt,count=16,cached=True):
    req=Request(base+'/generate',data=json.dumps({'prompt':prompt,'tokens':count,'cached':cached}).encode(),headers={'Content-Type':'application/json','X-Lab-Token':token})
    started=time.perf_counter();arrivals=[];ids=[]
    with urlopen(req,timeout=30) as response:
        for line in response:
            row=json.loads(line);ids.append(row['token_id']);arrivals.append(time.perf_counter()-started)
    return {**metrics(arrivals,time.perf_counter()-started),'arrivals':arrivals,'token_ids':ids}


def serving_experiment(a):
    from modern_models import TinyTransformer,Config
    from modern_lab import notes,fit,export_model
    torch.manual_seed(a.seed);model=TinyTransformer(Config(position='rope',kv_heads=2))
    history,_=fit(model,notes()['train'],a.steps,a.seed,response_only=False)
    export_model(a.out,model,'service api event start: ')
    server,token=create_server(model);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}'
    runs=[]
    try:
        request_tokens(base,token,'service api: ',4,True)
        for prompt in ['service api: ','service api event start: service queue event restore: ']:
            for cached in [False,True]:
                for repeat in range(3):
                    result=request_tokens(base,token,prompt,16,cached);result.update(prompt_bytes=len(prompt.encode()),cached=cached,repeat=repeat);runs.append(result)
    finally:server.shutdown();server.server_close();thread.join(timeout=5)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for row in runs[::3]:ax.plot(range(1,17),row['arrivals'],label=f"{row['prompt_bytes']} bytes; cache {row['cached']}")
    ax.set(xlabel='Output token index',ylabel='Client elapsed seconds',title='Real loopback streaming requests');ax.legend();fig.tight_layout();fig.savefig(a.out/'latency.png',dpi=130);plt.close(fig)
    return {'metrics':{'requests':runs,'measurement':'Client-observed loopback HTTP, warmed model, serial requests, CPU; three repeats per condition'},'history':history,'reload_verified':True,'artifacts':['latency.png'],
            'limitations':'Operating-system scheduling and HTTP buffering affect timing. No internet latency, concurrent-load test, GPU kernel, or production service-level agreement. Small cache workloads may be slower. Single-output-token requests have no defined subsequent-token average.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',required=True);p.add_argument('--port',type=int,default=8766);a=p.parse_args()
    torch.set_num_threads(2);model=restore(torch.load(Path(a.model_dir)/'model.pt',map_location='cpu',weights_only=True));server,token=create_server(model,a.port)
    print(json.dumps({'url':f'http://127.0.0.1:{server.server_port}/generate','header':'X-Lab-Token','token':token}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
