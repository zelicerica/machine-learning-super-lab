"""Tiny, real neural builds from random weights. No checkpoint download required.
Use only artifacts produced by your own lab. Every run refuses to overwrite a folder.
"""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SPECIAL = ['<pad>', '<bos>', '<eos>', '<unk>', '[MASK]']
ACTIONS = ['backup', 'inspect', 'restart', 'update']
KINDS = ('deep', 'recurrent', 'encoder', 'seq2seq', 'transfer')

def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def operations(seed=42):
    """Split whole action/service/zone groups, not paraphrases of the same group."""
    groups=list(itertools.product(ACTIONS,['amber','birch','cedar','delta','ember','flint'],['east','west','north','south']))
    rng=random.Random(seed); rng.shuffle(groups)
    result={name:[] for name in ('train','validation','test')}
    for i,(action,service,zone) in enumerate(groups):
        split='train' if i<56 else 'validation' if i<76 else 'test'
        for template in ('please {a} {s} in {z}', '{a} {s} in {z} now', 'can you {a} {s} in {z}'):
            result[split].append({'text':template.format(a=action,s=service,z=zone),
                'target':f'{action} {service} {zone}', 'label':ACTIONS.index(action),
                'group':f'{action}/{service}/{zone}'})
    return result

def vocabulary(rows):
    return SPECIAL+sorted({w for row in rows for text in (row['text'],row['target']) for w in text.split()})

def encode(texts, vocab, length=12):
    lookup={w:i for i,w in enumerate(vocab)}; result=[]
    for text in texts:
        words=text.split()
        if len(words)>length-2: raise ValueError(f'Use at most {length-2} whitespace-separated tokens.')
        ids=[1]+[lookup.get(w,3) for w in words]+[2]
        result.append(ids+[0]*(length-len(ids)))
    return torch.tensor(result,dtype=torch.long)

def decode(ids,vocab):
    result=[]
    for i in ids:
        if int(i)==2:break
        if int(i) not in (0,1):result.append(vocab[int(i)])
    return ' '.join(result)

class DenseClassifier(nn.Module):
    def __init__(self,depth=2,width=32):
        super().__init__();layers=[];features=2
        for _ in range(depth):layers.extend([nn.Linear(features,width),nn.ReLU()]);features=width
        self.net=nn.Sequential(*layers,nn.Linear(features,2))
    def forward(self,x):return self.net(x)

class RecurrentLM(nn.Module):
    def __init__(self,size,width=32):
        super().__init__();self.token=nn.Embedding(size,width,padding_idx=0)
        self.recurrent=nn.GRU(width,width,batch_first=True);self.head=nn.Linear(width,size)
    def forward(self,ids):return self.head(self.recurrent(self.token(ids))[0])

class MaskedEncoder(nn.Module):
    def __init__(self,size,width=32):
        super().__init__();self.token=nn.Embedding(size,width,padding_idx=0)
        self.position=nn.Embedding(12,width)
        self.layers=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*2,
            dropout=0,batch_first=True,norm_first=True),2,enable_nested_tensor=False)
        # TransformerEncoder clones its layer. Initialize each clone independently.
        for layer in self.layers.layers:
            for p in layer.parameters():
                if p.ndim>1:nn.init.xavier_uniform_(p)
        self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,size)
    def hidden(self,ids):
        x=self.token(ids)+self.position(torch.arange(ids.shape[1],device=ids.device))
        return self.norm(self.layers(x,src_key_padding_mask=ids.eq(0)))
    def forward(self,ids):return self.head(self.hidden(ids))
    def pooled(self,ids):
        h=self.hidden(ids);mask=ids.ne(0).unsqueeze(-1)
        return (h*mask).sum(1)/mask.sum(1)

class IntentClassifier(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder;self.head=nn.Linear(encoder.token.embedding_dim,4)
        # The vocabulary head is not used for classification; do not call it trainable.
        self.encoder.head.requires_grad_(False)
    def forward(self,ids):return self.head(self.encoder.pooled(ids))

class SequenceMapper(nn.Module):
    """A recurrent encoder-decoder; intentionally no attention or vendor architecture."""
    def __init__(self,size,width=32):
        super().__init__();self.token=nn.Embedding(size,width,padding_idx=0)
        self.encoder=nn.GRU(width,width,batch_first=True)
        self.decoder=nn.GRU(width,width,batch_first=True);self.head=nn.Linear(width,size)
    def state(self,source):
        lengths=source.ne(0).sum(1).cpu()
        packed=nn.utils.rnn.pack_padded_sequence(self.token(source),lengths,batch_first=True,enforce_sorted=False)
        return self.encoder(packed)[1]
    def forward(self,source,target_prefix):return self.head(self.decoder(self.token(target_prefix),self.state(source))[0])
    def generate(self,source,limit=6):
        h=self.state(source);last=torch.ones((len(source),1),dtype=torch.long,device=source.device);output=[]
        for _ in range(limit):
            values,h=self.decoder(self.token(last),h);last=self.head(values).argmax(-1);output.append(last)
        return torch.cat(output,1)

def masked(ids, generator):
    labels=torch.full_like(ids,-100);noisy=ids.clone()
    # At least one ordinary word per example; targets are the original words.
    for i in range(len(ids)):
        available=torch.where(ids[i]>=len(SPECIAL))[0]
        choice=available[torch.randperm(len(available),generator=generator)[:max(1,len(available)//4)]]
        labels[i,choice]=ids[i,choice];noisy[i,choice]=4
    return noisy,labels

def token_loss(logits, targets, ignore=0):
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=ignore)

def fit(model, steps, loss_fn, lr, seed, frozen=False):
    generator=torch.Generator().manual_seed(seed)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr)
    history=[];model.train()
    for step in range(steps):
        if frozen:model.encoder.eval()
        optimizer.zero_grad(set_to_none=True);loss=loss_fn(model,generator);loss.backward()
        norm=float(nn.utils.clip_grad_norm_(model.parameters(),1.0));optimizer.step()
        if step==0 or (step+1)%max(1,steps//20)==0:
            history.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm_before_clip':norm})
    model.eval();return history

def batch_indices(n,generator):return torch.randint(n,(min(24,n),),generator=generator)

def gradient_check():
    w=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    loss=(w*2-6).square();loss.backward();epsilon=1e-5
    finite=(((1+epsilon)*2-6)**2-((1-epsilon)*2-6)**2)/(2*epsilon)
    return {'prediction':2.,'target':6.,'loss':float(loss.detach()),'autograd':float(w.grad),
            'hand_derivative':-16.,'finite_difference':finite,'absolute_error':abs(float(w.grad)-finite)}

def build(kind,manifest):
    width=manifest.get('width',32);size=len(manifest.get('vocabulary',[]))
    if kind=='deep':return DenseClassifier(manifest['depth'],width)
    if kind=='recurrent':return RecurrentLM(size,width)
    if kind=='encoder':return MaskedEncoder(size,width)
    if kind=='seq2seq':return SequenceMapper(size,width)
    if kind=='transfer':return IntentClassifier(MaskedEncoder(size,width))
    raise ValueError('Unknown neural model kind.')

def predict(folder,inputs):
    folder=Path(folder);m=json.loads((folder/'model-manifest.json').read_text());kind=m['kind']
    model=build(kind,m);model.load_state_dict(torch.load(folder/'weights.pt',map_location='cpu',weights_only=True));model.eval()
    with torch.no_grad():
        if kind=='deep':
            values=np.asarray(inputs,dtype=float)
            if values.ndim!=2 or values.shape[1]!=2 or not np.isfinite(values).all():raise ValueError('Use rows of two finite sensor features.')
            x=torch.tensor((values-np.array(m['mean']))/np.array(m['scale']),dtype=torch.float32)
            return {'probabilities':model(x).softmax(-1).tolist(),'classes':model(x).argmax(-1).tolist()}
        if not isinstance(inputs,str) or not inputs.strip():raise ValueError('Supply a nonempty text string.')
        vocab=m['vocabulary']
        if kind=='recurrent':
            lookup={w:i for i,w in enumerate(vocab)};words=inputs.split()
            if len(words)>10:raise ValueError('Use a prefix of at most ten words.')
            ids=torch.tensor([[1]+[lookup.get(w,3) for w in words]])
            for _ in range(8):
                next_id=model(ids)[:,-1].argmax(-1,keepdim=True);ids=torch.cat([ids,next_id],1)
                if int(next_id)==2:break
            return {'completion':decode(ids[0,1:],vocab)}
        ids=encode([inputs],vocab)
        if kind=='encoder':
            locations=torch.where(ids[0]==4)[0]
            if len(locations)!=1:raise ValueError('Include exactly one literal [MASK] token.')
            probs=model(ids)[0,locations[0]].softmax(-1);values,indices=probs.topk(5)
            return {'masked_token_candidates':[{'token':vocab[int(i)],'probability':float(v)} for v,i in zip(values,indices)]}
        if kind=='seq2seq':return {'normalized_command':decode(model.generate(ids)[0],vocab)}
        return {'intent':ACTIONS[int(model(ids).argmax(-1))],'probabilities':model(ids).softmax(-1)[0].tolist()}

def run(a):
    if not 1<=a.steps<=2000 or not 0<=a.depth<=4 or not 0<a.lr<=.1:raise ValueError('Use 1–2000 steps, depth 0–4, and learning rate in (0, 0.1].')
    if not 1<=a.adapt_steps<=2000 or not 1<=a.labels_per_class<=100 or not 1<=a.threads<=32:raise ValueError('Invalid adaptation steps, labels per class, or thread count.')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);started=time.perf_counter()
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose a new empty output directory; previous results are preserved.')
    out.mkdir(parents=True,exist_ok=True)
    manifest={'format':'superlab-neural','kind':a.experiment,'width':32,'depth':a.depth}
    report={'schema':'superlab.report.v1','experiment':a.experiment,'split':a.split,'settings':vars(a),'metrics':{},
            'artifacts':['learning-curves.png'],'limitations':['Synthetic teaching data; results do not establish production capability.']}
    curves={}
    if a.experiment=='deep':
        x,y=make_moons(n_samples=600,noise=.18,random_state=a.seed)
        train,remaining=train_test_split(np.arange(len(x)),test_size=.4,random_state=a.seed,stratify=y)
        validation,test=train_test_split(remaining,test_size=.5,random_state=a.seed,stratify=y[remaining]);selected=validation if a.split=='validation' else test
        scaler=StandardScaler().fit(x[train]);values=torch.tensor(scaler.transform(x),dtype=torch.float32);labels=torch.tensor(y)
        manifest.update(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist())
        model=build('deep',manifest);before=float(F.cross_entropy(model(values[selected]),labels[selected]))
        def loss_fn(m,g):
            idx=torch.tensor(train)[batch_indices(len(train),g)];return F.cross_entropy(m(values[idx]),labels[idx])
        curves['training']=fit(model,a.steps,loss_fn,a.lr,a.seed)
        with torch.no_grad():prediction=model(values[selected]);accuracy=float(prediction.argmax(-1).eq(labels[selected]).float().mean())
        majority=int(np.bincount(y[train]).argmax())
        report['metrics']={'accuracy':accuracy,'majority_baseline_accuracy':float((y[selected]==majority).mean()),'loss_before':before,
            'loss_after':float(F.cross_entropy(prediction,labels[selected])),'parameter_count':sum(p.numel() for p in model.parameters())}
        report['gradient_check']=gradient_check();report['tensor_shapes']={'input':[len(train),2],'hidden_width':32,'hidden_layers':a.depth,'output':[len(train),2]}
        write(out/'data-splits.json',{'train':train.tolist(),'validation':validation.tolist(),'test':test.tolist()})
        example=[[.2,.5],[1.1,-.2]]
    else:
        data=operations(a.seed);vocab=vocabulary(data['train']);manifest['vocabulary']=vocab
        train=encode([r['text'] for r in data['train']],vocab);held=encode([r['text'] for r in data[a.split]],vocab)
        write(out/'data-splits.json',data);report['data']={'train_groups':56,'validation_groups':20,'test_groups':20,'vocabulary_fit':'training rows only'}
        model=build(a.experiment if a.experiment!='transfer' else 'encoder',manifest)
        if a.experiment=='recurrent':
            def objective(m,ids):return token_loss(m(ids[:,:-1]),ids[:,1:])
            before=float(objective(model,held).detach())
            curves['training']=fit(model,a.steps,lambda m,g:objective(m,train[batch_indices(len(train),g)]),a.lr,a.seed)
            with torch.no_grad():after=float(objective(model,held))
            report['metrics']={'loss_before':before,'loss_after':after,'perplexity':math.exp(after),'uniform_baseline_perplexity':len(vocab)}
            example='please restart amber'
        elif a.experiment in ('encoder','transfer'):
            noisy,targets=masked(held,torch.Generator().manual_seed(12345))
            before=float(token_loss(model(noisy),targets,-100).detach())
            def objective(m,g):
                batch=train[batch_indices(len(train),g)];masked_ids,labels=masked(batch,g);return token_loss(m(masked_ids),labels,-100)
            initial=copy.deepcopy(model)
            curves['pretraining']=fit(model,a.steps,objective,a.lr,a.seed)
            with torch.no_grad():logits=model(noisy);after=float(token_loss(logits,targets,-100));valid=targets.ne(-100)
            report['metrics']={'masked_loss_before':before,'masked_loss_after':after,
                'masked_accuracy':float(logits.argmax(-1)[valid].eq(targets[valid]).float().mean()),'uniform_baseline_loss':math.log(len(vocab))}
            example='please [MASK] amber in east'
            if a.experiment=='transfer':
                torch.save(model.state_dict(),out/'pretrained-encoder.pt')
                labels=torch.tensor([r['label'] for r in data['train']]);held_labels=torch.tensor([r['label'] for r in data[a.split]])
                # Same labeled examples and same new head for all branches; pretraining uses training groups only.
                selected=torch.cat([torch.where(labels==i)[0][:a.labels_per_class] for i in range(4)])
                torch.manual_seed(a.seed+1);initial_head=nn.Linear(32,4).state_dict();branches={};metrics={}
                for name,encoder,frozen in [('scratch',initial,False),('pretrained_finetune',model,False),('pretrained_frozen',model,True)]:
                    classifier=IntentClassifier(copy.deepcopy(encoder));classifier.head.load_state_dict(initial_head)
                    if frozen:
                        for p in classifier.encoder.parameters():p.requires_grad=False
                    snapshot={k:v.clone() for k,v in classifier.encoder.state_dict().items()}
                    loss_fn=lambda m,g:F.cross_entropy(m(train[selected]),labels[selected])
                    curves[name]=fit(classifier,a.adapt_steps,loss_fn,a.lr,a.seed+9,frozen)
                    with torch.no_grad():score=float(classifier(held).argmax(-1).eq(held_labels).float().mean())
                    unchanged=all(torch.equal(v,classifier.encoder.state_dict()[k]) for k,v in snapshot.items())
                    metrics[name]={'accuracy':score,'encoder_unchanged':unchanged,'labeled_examples':len(selected),'adaptation_steps':a.adapt_steps,
                        'pretraining_steps':0 if name=='scratch' else a.steps,'trainable_parameters':sum(p.numel() for p in classifier.parameters() if p.requires_grad)}
                    torch.save(classifier.state_dict(),out/f'{name}.pt');branches[name]=classifier
                model=branches['pretrained_finetune'];example='please restart birch in west';report['branches']=metrics
                report['metrics']['majority_baseline_accuracy']=float(held_labels.eq(labels[selected].bincount(minlength=4).argmax()).float().mean())
                report['limitations'].append('Equal downstream data and steps, but pretrained branches also consumed pretraining compute; one seed is not a universal ranking.')
        else:
            targets=encode([r['target'] for r in data['train']],vocab,length=6);expected=encode([r['target'] for r in data[a.split]],vocab,length=6)
            def objective(m,source,target):return token_loss(m(source,target[:,:-1]),target[:,1:])
            before=float(objective(model,held,expected).detach())
            def loss_fn(m,g):
                idx=batch_indices(len(train),g);return objective(m,train[idx],targets[idx])
            curves['training']=fit(model,a.steps,loss_fn,a.lr,a.seed)
            with torch.no_grad():after=float(objective(model,held,expected));generated=model.generate(held)
            results=[{'source':r['text'],'expected':r['target'],'predicted':decode(ids,vocab)} for r,ids in zip(data[a.split],generated)]
            report['predictions']=results
            report['metrics']={'loss_before':before,'loss_after':after,'generated_exact_match':sum(r['expected']==r['predicted'] for r in results)/len(results),
                'copy_input_baseline_exact_match':sum(r['text']==r['target'] for r in data[a.split])/len(data[a.split])}
            example='please backup cedar in south'
    model.eval();torch.save(model.state_dict(),out/'weights.pt');write(out/'model-manifest.json',manifest);write(out/'example-input.json',example)
    # Exact tensor verification on a new input, after reconstructing the architecture.
    rebuilt=build(a.experiment,manifest);rebuilt.load_state_dict(torch.load(out/'weights.pt',weights_only=True));rebuilt.eval()
    with torch.no_grad():
        if a.experiment=='deep':probe=torch.tensor([[0.,.5],[1.,-.2]]);left=model(probe);right=rebuilt(probe)
        elif a.experiment=='seq2seq':probe=encode([example],vocab);left=model.generate(probe);right=rebuilt.generate(probe)
        else:probe=encode([example],vocab);left=model(probe);right=rebuilt(probe)
    report['reload_verified']=bool(torch.equal(left,right));assert report['reload_verified']
    report['new_input']=example;report['reloaded_prediction']=predict(out,example)
    report['curves']=curves;report['history']=next(iter(curves.values()));report['seconds']=time.perf_counter()-started
    report['weights_sha256']=hashlib.sha256((out/'weights.pt').read_bytes()).hexdigest()
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name,curve in curves.items():ax.plot([p['step'] for p in curve],[p['loss'] for p in curve],label=name)
    ax.set(xlabel='Optimizer updates in this phase',ylabel='Batch training loss',title='Training behavior; use held-out metrics to judge quality');ax.legend();fig.tight_layout();fig.savefig(out/'learning-curves.png',dpi=140);plt.close(fig)
    write(out/'report.json',report);print(json.dumps({'output':str(out),'metrics':report['metrics'],'branches':report.get('branches'),
        'reload_verified':True,'new_input_prediction':report['reloaded_prediction'],'seconds':report['seconds']},indent=2));return report

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',choices=KINDS)
    p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=300);p.add_argument('--adapt-steps',type=int,default=80)
    p.add_argument('--labels-per-class',type=int,default=4);p.add_argument('--depth',type=int,default=2);p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--threads',type=int,default=2);p.add_argument('--split',choices=['validation','test'],default='validation')
    return p

if __name__=='__main__':run(parser().parse_args())
