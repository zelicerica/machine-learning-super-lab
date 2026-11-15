"""Progressive modern language-model engineering track, appended without changing IDs."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SPECS=[
 ('modern-map','A map of modern language-model engineering','concept','Orientation',[],
  'Which pair can describe the same training run?',['Supervised fine-tuning and low-rank adaptation','Training and an untouched final test used for tuning','Inference and automatic weight updates'],0,'The objective and the parameter-update method are different choices.'),
 ('attention-cache','Build attention and measure its cache','implementation','Attention',
  ['python modern_lab.py attention --steps 100 --output outputs/attention-{run}'],
  'With four query heads, what changes when key/value heads go from four to one?',['The stored key/value head count falls','All model memory falls by exactly four','The vocabulary becomes smaller'],0,'Only the affected cache component has this four-to-one head ratio. Parameters, activations and runtime overhead remain.'),
 ('position-normalization','Give a Transformer position, scale, and gates','implementation','Architecture',
  ['python modern_lab.py components --steps 100 --output outputs/components-{run}'],
  'Which experiment best isolates rotary positions?',['Change only position handling and disclose parameter differences','Also double width and training steps','Compare unrelated datasets'],0,'Control the data, seed, training budget and other components; report parameter-count differences.'),
 ('expert-routing','Route tokens and account for expert costs','implementation','Architecture',
  ['python modern_lab.py experts --steps 100 --output outputs/experts-{run}'],
  'If one of four experts runs per token, which weights still need storage?',['All four experts, unless separately offloaded','Only the last selected expert','No expert weights'],0,'Sparse execution does not remove stored parameters. Router and shared-layer work also remain.'),
 ('latent-depth','Explore latent caches and attention across depth','implementation','Research mechanisms',
  ['python modern_lab.py research --steps 100 --output outputs/research-{run}'],
  'What does the depth-mixture demonstration attend over?',['Earlier block representations','Only earlier token positions','An external document index'],0,'Depth and sequence position are different axes. This exercise is not a paper-exact Attention Residuals reproduction.'),
 ('adaptation-comparison','Compare full tuning and small parameter updates','implementation','Post-training',
  ['python modern_lab.py tuning --steps 150 --output outputs/tuning-{run}'],
  'Does the CPU packed-four-bit exercise reproduce the QLoRA paper?',['No; it uses uniform quantization of selected projections','Yes; any four-bit weight is NormalFloat4','Yes; low-rank adapters determine the quantizer'],0,'The real QLoRA extension uses NormalFloat4 and a supported quantization backend. The local exercise reports its narrower scope.'),
 ('quantized-cloud','Plan and run the optional NormalFloat4 extension','concept','Optional cloud',[],
  'When is a paid graphics processor justified?',['After a measured resource need and a cost plan','Whenever a model has an acronym','Before checking the local baseline'],0,'Record local memory/runtime, select hardware for the actual workload, preserve artifacts, and account for retained storage.'),
 ('preference-learning','Learn from preferred and rejected responses','implementation','Preference learning',
  ['python modern_lab.py preference --steps 100 --beta 0.1 --output outputs/preference-{run}',
   'python modern_lab.py preference --steps 100 --beta 0.5 --output outputs/preference-beta-{run}'],
  'What stays frozen in this direct preference optimization build?',['The reference model','The policy being trained','Both policy and reference'],0,'Policy response likelihoods change; the reference likelihoods provide a fixed comparison.'),
 ('reward-learning','Build and audit a learned reward model','implementation','Reward design',
  ['python modern_lab.py reward --steps 100 --output outputs/reward-{run}'],
  'Does a high reward score prove factual correctness?',['No; it reflects the learned scoring objective','Yes, if the score is positive','Only when the response is long'],0,'A style scorer can assign high reward to unsupported claims or repetition. Use independent outcome evaluation.'),
 ('policy-gradient','Move from grid-world actions to generated tokens','implementation','Policy gradients',
  ['python modern_lab.py reinforce --steps 100 --reward exact --output outputs/policy-{run}',
   'python modern_lab.py reinforce --steps 100 --reward zero-prefix --output outputs/policy-flawed-{run}'],
  'Which outcome exposes reward hacking?',['Reward rises while independent task accuracy falls','Both training and test accuracy rise','The model saves a checkpoint'],0,'The deliberately flawed reward prefers a leading zero rather than a correct arithmetic answer.'),
 ('policy-optimization','Compare clipped and group-relative policy updates','implementation','Reinforcement learning',
  ['python modern_lab.py ppo --steps 100 --output outputs/ppo-{run}',
   'python modern_lab.py grpo --steps 100 --output outputs/grpo-{run}',
   'python modern_lab.py gspo --steps 100 --output outputs/gspo-{run}'],
  'What happens to normalized reward advantages when a group has identical rewards?',['They are zero in this implementation','They become perfect learning signals','They always become negative'],0,'No within-group reward contrast means no normalized reward advantage; other regularizers can still contribute.'),
 ('benchmark-literacy','Choose a benchmark that answers your question','concept','Benchmark literacy',[],
  'Can a homemade arithmetic exercise be reported as an AIME score?',['No; benchmark identity and protocol must match','Yes, if it uses integers','Yes, if accuracy is high'],0,'Dataset version, split, prompts, tools and scoring protocol are part of a benchmark result.'),
 ('evaluation-engineering','Build a reproducible evaluation harness','implementation','Evaluation',
  ['python modern_lab.py evaluation --steps 100 --split validation --output outputs/evaluation-{run}'],
  'What does pass at four measure?',['At least one correct candidate among four attempts','Four consecutive correct tasks','Average response length'],0,'Sampling and the per-problem correctness test define the measure. It is not a deployed system selection policy.'),
 ('judge-audit','Calibrate judges and compare preferences','implementation','Judge reliability',
  ['python modern_lab.py judging --judge fixture --output outputs/judge-fixture-{run}',
   'python modern_lab.py judging --judge model --output outputs/judge-model-{run}'],
  'Why swap the order of two candidate answers?',['To detect order sensitivity','To train the reference answers','To increase the true sample size automatically'],0,'Order changes should not change the underlying preferred content. Repeated presentations are not independent new problems.'),
 ('streaming-service','Serve a model and measure when tokens arrive','implementation','Inference systems',
  ['python modern_lab.py serving --steps 100 --output outputs/serving-{run}'],
  'How is subsequent-token time defined for a one-token response?',['Undefined; report no value','Always zero latency','Equal to total request time'],0,'There is no interval between output tokens when only one token arrives.'),
 ('modern-capstone','Defend a complete language-model engineering decision','concept','Capstone',[],
  'What makes the final comparison credible?',['A locked test protocol and reproducible artifacts','The largest model name','Reporting only the best random seed'],0,'Disclose settings, data, failures, uncertainty, resource use and deployment limits; preserve the untouched final assessment.')
]


def chapters():
    text=(ROOT/'course/Modern-Language-Engineering.md').read_text(encoding='utf-8')
    chunks=text.split('\n## ')
    result={part.split('\n',1)[0].split(' | ',1)[0]: '## '+part.split(' | ',1)[1] for part in chunks[1:]}
    result['modern-map']=chunks[0]+'\n\n'+result['modern-map']
    return result


def additions(render,start):
    content=chapters();result=[]
    for i,(ident,title,kind,topic,commands,q,options,correct,explanation) in enumerate(SPECS):
        scope='Design activity and evidence assessment' if kind=='concept' else 'Local teaching implementation with measured outputs'
        if ident=='latent-depth':scope='Simplified research mechanisms; not vendor-model reproductions'
        if ident=='judge-audit':scope='Simulated bias audit plus optional real pretrained local judge'
        if ident=='quantized-cloud':scope='Hardware-specific extension; graphics-processor execution not validated in the CPU release'
        result.append({'id':ident,'number':start+i,'title':title,'description':topic+' through a real use case, a worked algorithm, and evidence.',
                       'topic':topic,'track':'Modern language engineering','kind':kind,'minutes':40,'scope':scope,
                       'needs':'foundations' if kind=='implementation' else None,'html':render(content[ident]),'commands':commands,
                       'saved_model':kind=='implementation' and ident!='judge-audit',
                       'quiz':{'question':q,'options':options,'correct':correct,'explanation':explanation}})
    return result


def references(render):
    return {key:{'title':title,'html':render((ROOT/'course'/file).read_text(encoding='utf-8'))} for key,title,file in [
        ('modern-path','Modern language-model learning path','Modern-Language-Engineering.md'),
        ('research-landmarks','Landmark research: ideas and evidence','Research-Landmarks.md'),
        ('modern-glossary','Modern language-model terminology','Modern-Glossary.md'),
        ('qlora-cloud','Quantized adaptation and cloud costs','QLoRA-and-Cloud.md')]}
