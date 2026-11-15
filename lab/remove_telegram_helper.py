"""Remove only tqdm's optional Telegram integration, at the owner's request.

Run with the specific test environment interpreter after installing packages.
Core tqdm remains available. Reinstallation may restore the optional module.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

def main():
    dist=importlib.metadata.distribution('tqdm')
    contrib=Path(dist.locate_file('tqdm/contrib')).resolve()
    environment=Path(sys.prefix).resolve()
    if sys.prefix==sys.base_prefix or not contrib.is_relative_to(environment):
        raise SystemExit('Use the test virtual environment; no global environment will be changed.')
    paths=[contrib/'telegram.py',*(contrib/'__pycache__').glob('telegram.*.pyc')]
    removed=[]
    for path in paths:
        if path.is_file():
            if not path.resolve().is_relative_to(contrib):raise RuntimeError('Path escaped tqdm/contrib.')
            removed.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            path.unlink()
    print(json.dumps({'removed':removed,'core_tqdm_retained':True,'note':'Original installed RECORD still lists the removed helper.'},indent=2))

if __name__=='__main__':main()
