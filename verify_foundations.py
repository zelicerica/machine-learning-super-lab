"""Run every new implementation and record validation evidence; never overwrite runs."""
from pathlib import Path
import json, subprocess, sys, time
ROOT=Path(__file__).resolve().parent
CONFIGS={
'linear':['--noise','8'],'logistic':[],'tree':[],'ensemble':[], 'neighbors':[],'kernel':[],'bayes':[], 'gaussian':[], 'cluster':['--noise','1.2'],'projection':[], 'dense':['--epochs','15'],'convolution':['--epochs','15'],'graph':['--epochs','100'],'autoencoder':['--epochs','25'],'variational':['--epochs','25'],'adversarial':['--steps','600'],'ngram':[],'reinforcement':[]}
def main():
 folder=ROOT/'lab/outputs'/('verification-'+str(int(time.time())))
 folder.mkdir(parents=True)
 outcomes=[]
 for name,options in CONFIGS.items():
  command=[sys.executable,'foundations.py',name,*options,'--output',str(folder/name)]
  r=subprocess.run(command,cwd=ROOT/'lab',capture_output=True,text=True,timeout=180)
  (folder/(name+'.log')).write_text(r.stdout+r.stderr,encoding='utf-8')
  report=folder/name/'report.json'
  record={'experiment':name,'exit_code':r.returncode,'report':str(report)}
  if r.returncode==0:
   data=json.loads(report.read_text());record['seconds']=data['seconds'];record['metrics']=data['metrics'];record['artifacts_exist']=all((report.parent/p).is_file() for p in data['artifacts'])
  else:record['error']=(r.stdout+r.stderr)[-1000:]
  outcomes.append(record);print(json.dumps(record),flush=True)
 (folder/'verification.json').write_text(json.dumps(outcomes,indent=2))
 print('Evidence directory:',folder)
 if any(o['exit_code'] or not o.get('artifacts_exist') for o in outcomes):raise SystemExit(1)
if __name__=='__main__':main()
