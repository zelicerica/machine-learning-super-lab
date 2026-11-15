"""Create a fresh private Windows lab environment, including long-path support.

An existing environment is never modified. Runtime selection changes only after
hashed wheel installation and package checks succeed. No administrator settings
or Windows security policies are changed.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import shutil
import struct
import subprocess
import sys
import uuid
import venv

ROOT=Path(__file__).resolve().parents[1]

def extended(path):
    value=str(Path(path).resolve())
    if os.name=='nt' and not value.startswith('\\\\?\\'):
        return '\\\\?\\UNC\\'+value[2:] if value.startswith('\\\\') else '\\\\?\\'+value
    return value

def select_fresh_environment(root):
    return Path(root)/'.lab-environments'/('env-'+uuid.uuid4().hex)

def publish_runtime(root,executable):
    root=Path(root);config=root/'.runtime-python';identifier=uuid.uuid4().hex
    previous=None
    if config.exists():
        previous=root/('.runtime-python.backup-'+identifier)
        shutil.copy2(config,previous)
    pending=root/('.runtime-python.pending-'+identifier)
    with pending.open('x',encoding='utf-8') as handle:handle.write(str(executable)+'\n')
    os.replace(pending,config)
    return previous

def install_from_wheel(bootstrap,lock,wheelhouse=None):
    # Run the local standard-library ensurepip wheel in the fresh interpreter.
    # No downloaded build hooks, shell command strings, or existing site packages.
    sys.path.insert(0,str(bootstrap))
    sys.argv=['__main__.py','install','--no-cache-dir','--only-binary=:all:','--require-hashes','-r',str(lock)]
    if wheelhouse:sys.argv.extend(['--no-index','--find-links',str(wheelhouse)])
    runpy.run_module('pip',run_name='__main__')

def setup(root=ROOT,lock=None,wheelhouse=None,verify_packages=True):
    if sys.platform!='win32' or sys.version_info[:2]!=(3,12) or struct.calcsize('P')!=8:
        raise RuntimeError('This candidate setup requires 64-bit Windows Python 3.12.')
    import ensurepip
    bootstrap=next((Path(ensurepip.__file__).parent/'_bundled').glob('pip-*.whl'))
    root=Path(root);lock=Path(lock) if lock else ROOT/'lab/requirements-windows-cpu.lock'
    target=select_fresh_environment(root)
    if target.exists():raise RuntimeError('New environment path already exists.')
    target.parent.mkdir(parents=True,exist_ok=True)
    print('Creating a separate environment:',target,flush=True)
    venv.EnvBuilder(with_pip=False).create(extended(target))
    executable=extended(target/'Scripts/python.exe')
    command=[executable,'-I',str(Path(__file__).resolve()),'--install-wheels',str(bootstrap),'--lock',str(lock)]
    if wheelhouse:command.extend(['--wheelhouse',str(wheelhouse)])
    subprocess.run(command,check=True,timeout=1500)
    subprocess.run([executable,str(ROOT/'lab/remove_telegram_helper.py')],check=True,timeout=60)
    subprocess.run([executable,'-I','-m','pip','check'],check=True,timeout=60)
    if verify_packages:
        subprocess.run([executable,'-I','-c','import torch,transformers,peft,accelerate,numpy,sklearn,scipy,matplotlib,PIL;print("Learning package imports passed")'],check=True,timeout=240)
    previous=publish_runtime(root,executable)
    result={'created_at':datetime.now(timezone.utc).isoformat(),'environment':str(target),'python':executable,'requirements':str(lock),'previous_runtime_backup':str(previous) if previous else None,'existing_environments_preserved':True,'status':'ready'}
    report=target/'setup-report.json'
    report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-wheels',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--lock',type=Path)
    parser.add_argument('--wheelhouse',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.install_wheels:
        if not args.lock:parser.error('Lock required for internal install phase')
        install_from_wheel(args.install_wheels,args.lock,args.wheelhouse)
    else:setup(lock=args.lock,wheelhouse=args.wheelhouse)

if __name__=='__main__':main()
