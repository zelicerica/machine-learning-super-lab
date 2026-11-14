"""Verify the changed-setting experiment supplied by every foundation lesson."""
from pathlib import Path
import json,shlex,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
folder=ROOT/'lab/outputs'/('variant-check-'+str(time.time_ns()));folder.mkdir(parents=True)
course=json.loads((ROOT/'public/course.json').read_text(encoding='utf-8'));results=[]
for lesson in course['lessons']:
 if lesson.get('needs')!='foundations':continue
 args=shlex.split(lesson['commands'][1]);args[0]=sys.executable;args[args.index('--output')+1]=str(folder/lesson['id'])
 p=subprocess.run(args,cwd=ROOT/'lab',capture_output=True,text=True,timeout=180)
 (folder/(lesson['id']+'.log')).write_text(p.stdout+p.stderr,encoding='utf-8')
 record={'experiment':lesson['id'],'exit_code':p.returncode}
 if p.returncode:record['error']=(p.stdout+p.stderr)[-1500:]
 results.append(record);print(json.dumps(record),flush=True)
(folder/'verification.json').write_text(json.dumps(results,indent=2))
print('Evidence directory:',folder)
if any(r['exit_code'] for r in results):raise SystemExit(1)
