"""Pretraining decisions, executable neural builds, and retrieval engineering."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def document(name):return (ROOT/'course'/name).read_text(encoding='utf-8')

def build_section(title):
    text=document('Neural-Builds.md');marker='## '+title+'\n'
    return marker+text.split(marker,1)[1].split('\n## ',1)[0]

SPECS=[
 ('training-route','Pretrained or from scratch: choose the route','Training choices','concept','Pretrained-vs-Scratch.md',None,[],
  'Can an instruction-tuned checkpoint also be pretrained?',
  ['Yes; it already has learned parameters','No; pretrained means base only','Only after retrieval'],0,
  'Pretrained describes prior learning. Base and instruction-tuned describe different stages or behaviors.'),
 ('deep-network','Deep learning: train and inspect every layer','Deep learning','implementation','Deep-Learning.md',None,
  ['python neural_builds.py deep --depth 2 --steps 300 --output outputs/deep-two-{run}',
   'python neural_builds.py deep --depth 0 --steps 300 --output outputs/deep-linear-{run}'],
  'Why do hidden nonlinear activations matter?',
  ['They allow more than a single linear transformation','They eliminate the need for validation','They guarantee higher accuracy'],0,
  'Without nonlinearities, a stack of linear transformations is still linear. More flexibility must still be evaluated.'),
 ('scratch-recurrent','Build a recurrent language model from scratch','From scratch','implementation',None,'Recurrent language model from scratch',
  ['python neural_builds.py recurrent --steps 300 --output outputs/recurrent-{run}',
   'python neural_builds.py recurrent --steps 100 --output outputs/recurrent-short-{run}'],
  'What carries earlier information between recurrent steps?',
  ['A learned numerical hidden state','An external search index','The future target word'],0,
  'The recurrent update carries a compressed hidden state. It must not read a future target to predict that target.'),
 ('scratch-encoder','Build a masked-token encoder from scratch','From scratch','implementation',None,'Masked-token encoder from scratch',
  ['python neural_builds.py encoder --steps 300 --output outputs/encoder-{run}',
   'python neural_builds.py encoder --steps 100 --output outputs/encoder-short-{run}'],
  'Why replace a target word with a mask before predicting it?',
  ['To prevent the encoder from reading the answer at that position','To make the model autoregressive','To remove all surrounding context'],0,
  'Bidirectional context is allowed, but the hidden target itself must not be supplied as input.'),
 ('scratch-sequence','Build an encoder–decoder from scratch','From scratch','implementation',None,'Encoder–decoder model from scratch',
  ['python neural_builds.py seq2seq --steps 400 --output outputs/sequence-{run}',
   'python neural_builds.py seq2seq --steps 100 --output outputs/sequence-short-{run}'],
  'Which evaluation catches errors caused by earlier generated mistakes?',
  ['Free-running generated sequence evaluation','Only teacher-forced training loss','Only parameter count'],0,
  'Generation uses the model’s own earlier outputs; teacher forcing instead supplies the correct previous targets.'),
 ('transfer-compare','Compare pretrained, frozen, and fresh models','Transfer learning','implementation',None,'Compare pretraining with a fresh start',
  ['python neural_builds.py transfer --steps 300 --adapt-steps 80 --labels-per-class 4 --output outputs/transfer-{run}',
   'python neural_builds.py transfer --steps 300 --adapt-steps 80 --labels-per-class 1 --output outputs/transfer-few-{run}'],
  'What must be counted in addition to equal downstream updates?',
  ['The pretrained branch’s earlier data and compute','Only the filename length','Nothing; the total cost is automatically equal'],0,
  'Equal downstream examples and updates do not erase pretraining cost. Measure outcomes rather than assuming transfer helps.'),
 ('rag-pipeline','Build and audit retrieval-augmented generation','Retrieval engineering','implementation','Retrieval-Augmented-Generation.md',None,
  ['python rag_workshop.py --mode retrieval --output outputs/retrieval-{run}',
   'python rag_workshop.py --chunk-size 8 --overlap 0 --top-k 1 --output outputs/retrieval-small-{run}',
   'python rag_workshop.py --include-stale --top-k 2 --output outputs/retrieval-stale-{run}',
   'python lab.py lock',
   'python rag_workshop.py --mode compare --output outputs/retrieval-generation-{run}'],
  'What can change an answer while the generator’s weights stay identical?',
  ['Different retrieved context in the prompt','Only a training update','Only a new architecture'],0,
  'Retrieved passages change inference inputs. Ordinary retrieval and generation do not update generator parameters.'),
 ('synthetic-data','Synthetic data: generate, audit, and compare','Data engineering','implementation','Synthetic-Data.md',None,
  ['python synthetic_data.py --output outputs/synthetic-full-{run}',
   'python synthetic_data.py --coverage narrow --output outputs/synthetic-narrow-{run}',
   'python synthetic_data.py --label-noise 1 --output outputs/synthetic-wrong-{run}',
   'python synthetic_data.py --samples 2000 --output outputs/synthetic-more-{run}'],
  'What does a high score on held-out simulated data establish?',
  ['Performance on that simulated distribution; real-world transfer still needs evaluation',
   'Guaranteed production accuracy','That generated data is automatically private'],0,
  'Held-out simulator draws can test a workflow, but they share the simulator’s assumptions. Real-world claims need representative independent evidence.'),
]

def additions(render,start):
    result=[]
    for number,spec in enumerate(SPECS,start):
        ident,title,topic,kind,file,section,commands,question,options,correct,explanation=spec
        text=document(file) if file else build_section(section)
        result.append({'id':ident,'number':number,'title':title,'description':{
            'training-route':'Understand checkpoints, training history, reuse, adaptation, and full cost.',
            'deep-network':'Train a nonlinear classifier, check a gradient, compare depth, and reload predictions.',
            'scratch-recurrent':'Train a word predictor with a compact recurrent state and no pretrained weights.',
            'scratch-encoder':'Learn missing-word prediction with bidirectional context and inspect probabilities.',
            'scratch-sequence':'Train a source-to-output model and evaluate its own generated sequences.',
            'transfer-compare':'Measure three matched downstream branches, including a frozen-weight check.',
            'rag-pipeline':'Evaluate chunks, ranking, stale documents, citations, and actual frozen generation.',
            'synthetic-data':'Learn when generated data helps, then measure coverage, wrong labels, and downstream utility.'}[ident],
            'topic':topic,'track':'Build and retrieve','kind':kind,'minutes':35 if kind=='implementation' else 20,
            'scope':'Design activity and decision assessment' if kind=='concept' else 'Tiny training from random weights, evaluation, and reload',
            'needs':'foundations' if kind=='implementation' else None,'html':render(text),'commands':commands,
            'quiz':{'question':question,'options':options,'correct':correct,'explanation':explanation}})
        if ident=='transfer-compare':result[-1]['scope']='Pretrain a tiny encoder, then compare three classification branches'
        if ident=='rag-pipeline':result[-1]['scope']='Retrieval evaluation plus real pretrained generation; no generator training'
        if ident=='synthetic-data':result[-1]['scope']='Fictional data generation, auditing, and three trained classifier comparisons'
    book='# Building Models and Retrieval Systems\n\nMachine Learning Super Lab • Expanded November 14, 2026\n\n'
    book+='\n\n'.join(document(f) for f in ['Pretrained-vs-Scratch.md','Deep-Learning.md','Neural-Builds.md','Retrieval-Augmented-Generation.md','Synthetic-Data.md'])
    (ROOT/'course/Building-Models-and-Retrieval.md').write_text(book,encoding='utf-8')
    return result

def references(render):
    return {key:{'title':title,'html':render(document(file))} for key,title,file in [
        ('pretrained','Pretrained versus from scratch','Pretrained-vs-Scratch.md'),
        ('deep-learning','Deep learning workshop','Deep-Learning.md'),
        ('rag','Retrieval-augmented generation','Retrieval-Augmented-Generation.md'),
        ('synthetic-data','Synthetic data workshop','Synthetic-Data.md')]}
