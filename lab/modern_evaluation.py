"""Judge calibration: explicit simulated bias, or an actual local language model."""
import argparse
import json
import re
import numpy as np

CASES=[
 {'question':'What is 2 + 3?','good':'5','bad':'After careful thought, the answer is definitely 8.'},
 {'question':'Does ordinary inference update model weights?','good':'No.','bad':'Yes, every generated word trains the weights automatically.'},
 {'question':'Which split should guide hyperparameter choices?','good':'Validation.','bad':'Always tune repeatedly on the final test set.'},
 {'question':'Does retrieval alone fine-tune weights?','good':'No. It supplies context.','bad':'Yes. Retrieved notes always become permanent model weights.'},
 {'question':'What is 4 + 4?','good':'8','bad':'The result, using advanced arithmetic and careful checking, is 9.'},
 {'question':'Can a tiny teaching test prove production reliability?','good':'No.','bad':'Yes, a single perfect tiny test proves reliability everywhere.'}
]


def elo(ratings,winner,loser,k=16):
    expected=1/(1+10**((ratings[loser]-ratings[winner])/400));change=k*(1-expected)
    ratings[winner]+=change;ratings[loser]-=change


def parse_choice(text):
    text=text.strip().upper()
    return text if text in {'A','B'} else None


def judging(a):
    generator=None;metadata={}
    if a.judge=='model':
        import lab
        args=argparse.Namespace(model='HuggingFaceTB/SmolLM2-135M-Instruct',adapter=None,device='cpu',threads=a.threads,seed=a.seed)
        model,tokenizer,device=lab.load(args)
        def generator(prompt):return lab.generate(model,tokenizer,device,prompt,'Compare factual correctness only. Return exactly A or B.',4,0)['text']
        metadata={'model':args.model,'revision':lab.resolve(args.model),'rubric':'Factual correctness only; exact A or B output'}
    rows=[];ratings={'correct concise':1000.,'wrong verbose':1000.}
    for case_id,case in enumerate(CASES):
        for swapped in [False,True]:
            first,second=(case['bad'],case['good']) if swapped else (case['good'],case['bad']);expected='B' if swapped else 'A'
            prompt=f"Question: {case['question']}\nA: {first}\nB: {second}\nWhich answer is more factually correct? Return exactly A or B."
            # Deliberately bad, transparent fixture: prefers the longer answer.
            raw=generator(prompt) if generator else ('A' if len(first)>len(second) else 'B')
            choice=parse_choice(raw);correct=choice==expected
            rows.append({'case':case_id,'swapped':swapped,'expected':expected,'raw_judge_output':raw,'parsed':choice,'correct':correct})
            if choice:elo(ratings,'correct concise' if correct else 'wrong verbose','wrong verbose' if correct else 'correct concise')
    consistency=[]
    for i in range(0,len(rows),2):
        pair=rows[i:i+2]
        if all(x['parsed'] for x in pair):consistency.append(pair[0]['correct']==pair[1]['correct'])
    import matplotlib.pyplot as plt
    metrics={'agreement_with_authored_answer_key':sum(r['correct'] for r in rows)/len(rows),'valid_choice_rate':sum(r['parsed'] is not None for r in rows)/len(rows),
             'swap_consistency_on_valid_pairs':float(np.mean(consistency)) if consistency else None,'elo_ratings':ratings,'judge_mode':a.judge,'cases':rows,**metadata}
    plt.figure(figsize=(6,4));plt.bar(['Agreement','Valid choice'],[metrics['agreement_with_authored_answer_key'],metrics['valid_choice_rate']],color=['#527b57','#c08a42']);plt.ylim(0,1);plt.title('A valid judge response can still be wrong');plt.tight_layout();plt.savefig(a.out/'judge-audit.png',dpi=130);plt.close()
    return {'metrics':metrics,'artifacts':['judge-audit.png'],'limitations':'Fixture mode is a deliberately length-biased simulated judge, not an LLM. Model mode uses a real small local language model, which may fail. Answer keys are author-specified, not a human-participant study. Twelve pair presentations do not establish judge reliability. Elo depends on opponents, order and rating settings; not a universal quality scale.'}
