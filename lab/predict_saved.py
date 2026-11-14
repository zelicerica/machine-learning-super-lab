"""Reload models created by this lab and predict from a JSON input file.
Only load your own trusted artifacts: joblib and TorchScript are executable formats.
"""
import argparse
import json
from pathlib import Path
import numpy as np

def predict(folder,inputs):
    folder=Path(folder);manifest=json.loads((folder/'model-manifest.json').read_text())
    kind=manifest['format']
    if kind=='superlab-synthetic':
        from synthetic_data import predict as synthetic_predict
        return synthetic_predict(folder,inputs)
    if kind=='superlab-neural':
        from neural_builds import predict as neural_predict
        return neural_predict(folder,inputs)
    if kind=='superlab-rag':
        from rag_workshop import predict as rag_predict
        return rag_predict(folder,inputs)
    if kind=='joblib':
        import joblib
        models=joblib.load(folder/'model.joblib')
        x=np.asarray(inputs,dtype=float)
        if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Provide a finite two-dimensional feature array.')
        return {name:getattr(model,manifest['method'])(x).tolist() for name,model in models.items()}
    if kind=='torchscript':
        import torch
        torch.set_num_threads(2)
        model=torch.jit.load(str(folder/'model.pt'),map_location='cpu');model.eval()
        x=torch.as_tensor(inputs,dtype=torch.float32)
        if x.ndim!=2 or not torch.isfinite(x).all():raise ValueError('Provide a finite two-dimensional feature array.')
        with torch.no_grad():out=model(x)
        if manifest['output']=='class_index':out=out.argmax(1)
        return out.tolist()
    if kind=='ngram':
        data=json.loads((folder/'counts.json').read_text());size=data['order']-1
        words=inputs.split() if isinstance(inputs,str) else list(inputs)
        words=[w if w in data['vocabulary'] else '<unk>' for w in words]
        context=tuple((['<s>']*size+words)[-size:]) if size else ()
        rows={tuple(r['context']):r['counts'] for r in data['rows']};row=rows.get(context,{})
        denominator=sum(row.values())+len(data['vocabulary'])
        return {w:(row.get(w,0)+1)/denominator for w in data['vocabulary']}
    if kind=='action_table':
        table=np.asarray(json.loads((folder/'action_values.json').read_text()));states=np.asarray(inputs)
        if not np.issubdtype(states.dtype,np.integer) or np.any(states<0) or np.any(states>=16):raise ValueError('State identifiers must be integers from 0 through 15.')
        return table[states].argmax(-1).tolist()
    raise ValueError('Unknown artifact format.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=predict(a.model_dir,json.loads(Path(a.input).read_text()));output=Path(a.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as handle:json.dump({'predictions':result},handle,indent=2,allow_nan=False)
    print(json.dumps({'predictions':result},indent=2,allow_nan=False))

if __name__=='__main__':main()
