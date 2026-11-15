"""Compile the trusted course Markdown into portable, escaped lesson HTML."""
import html
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def inline(text):
    tokens={}
    def hold(value):
        key=f'@@TOKEN{len(tokens)}@@';tokens[key]=value;return key
    text=re.sub(r'`([^`]+)`',lambda m:hold('<code>'+html.escape(m.group(1))+'</code>'),text)
    text=html.escape(text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    text=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<a href="\2" target="_blank" rel="noreferrer">\1</a>',text)
    text=re.sub(r'\[([^\]]+)\]\((#[\w/-]+)\)',r'<a href="\2">\1</a>',text)
    for k,v in tokens.items():text=text.replace(k,v)
    return text

def render(md):
    lines=md.splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('```'):
            language=line[3:].strip();code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            if language=='mermaid':out.append('<div class="diagram"><pre class="mermaid">'+html.escape('\n'.join(code))+'</pre><details><summary>Read the diagram algorithm</summary><pre>'+html.escape('\n'.join(code))+'</pre></details></div>')
            else:out.append('<div class="code-block"><div class="code-label">'+html.escape(language or 'example')+'</div><pre><code>'+html.escape('\n'.join(code))+'</code></pre></div>')
            i+=1;continue
        m=re.match(r'!\[([^\]]*)\]\((images/[^)]+)\)',line)
        if m:
            out.append(f'<figure><a href="/{html.escape(m[2])}" target="_blank"><img loading="lazy" src="/{html.escape(m[2])}" alt="{html.escape(m[1])}"></a><figcaption>{html.escape(m[1])}. Select the image to enlarge it.</figcaption></figure>');i+=1;continue
        m=re.match(r'(#{1,6}) (.*)',line)
        if m:out.append(f'<h{min(len(m[1])+1,5)}>{inline(m[2])}</h{min(len(m[1])+1,5)}>');i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',x) for x in cells): rows.append(cells)
                i+=1
            out.append('<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in rows[0])+'</tr></thead><tbody>')
            for row in rows[1:]:out.append('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in row)+'</tr>')
            out.append('</tbody></table></div>');continue
        if re.match(r'^(\d+\. |[-*] )',line):
            ordered=bool(re.match(r'^\d+\.',line));tag='ol' if ordered else 'ul';out.append('<'+tag+'>')
            while i<len(lines) and re.match(r'^(\d+\. |[-*] )',lines[i]):
                out.append('<li>'+inline(re.sub(r'^(\d+\. |[-*] )','',lines[i]))+'</li>');i+=1
            out.append('</'+tag+'>');continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||```|!\[|\d+\. |[-*] )',lines[i]):paragraph.append(lines[i]);i+=1
        out.append('<p>'+inline(' '.join(paragraph))+'</p>')
    return '\n'.join(out)

def main():
    md=(ROOT/'course/SML-Lab-Zero-to-Hero.md').read_text(encoding='utf-8')
    chunks=re.split(r'^## ',md,flags=re.M)
    sections={x.split('\n',1)[0]:x.split('\n',1)[1] for x in chunks[1:]}
    def section(*starts):
        selected=[]
        for start in starts:
            for title,body in sections.items():
                if title.startswith(start):selected.append('## '+title+'\n'+body);break
        return render('\n'.join(selected))
    metadata=[
      ('setup','Build your workspace','Make your first environment check. No rented computer required.','Environment',15,'Lab 0',
       ['python lab.py doctor','python -m unittest discover -s tests -v'],
       'What does a virtual environment separate?',['This project’s installed Python packages','The physical processor','Training and test examples'],0,'A virtual environment gives a project its own Python package installation.'),
      ('inference','Run your first model','Turn a short prompt into a real model response.','Inference',20,'Lab 1',
       ['python lab.py lock','python lab.py infer --prompt "Explain overfitting in one sentence." --max-new-tokens 48','python lab.py tokenize --prompt "A model predicts the next token."'],
       'What changes during ordinary inference?',['All learned weights','The current activations and context','The training dataset'],1,'Inference changes the current computation. The learned weights remain fixed.'),
      ('math','See the numbers underneath','Connect tensors, probabilities, loss and gradients.','Foundations',25,'Lab 2',
       ['python mechanisms.py --demo math'],
       'What does a gradient describe?',['A parameter’s local effect on loss','The number of words in a prompt','The model’s factual accuracy'],0,'A gradient describes a local rate of change. The optimizer uses it when updating parameters.'),
      ('baseline','Measure a baseline','Inspect every prediction before changing the model.','Evaluation',30,'Lab 3',
       ['python -m unittest discover -s tests -v','python lab.py eval --split validation --output outputs/baseline-validation.json'],
       'Which split should guide tuning choices?',['The final test split','The validation split','Any examples the model memorizes'],1,'Use validation for design choices, then preserve an independent final test.'),
      ('adapt','Teach with a small adapter','Train low-rank updates and compare real outputs.','Fine-tuning',40,'Lab 4',
       ['python lab.py train --steps 5 --output outputs/smoke-{run}','python lab.py train --steps 60 --output outputs/adapter-{run}','python lab.py eval --adapter {adapter} --split validation --output outputs/adapter-validation.json','python compare_reports.py outputs/baseline-validation.json outputs/adapter-validation.json'],
       'What does low-rank adaptation update?',['Small added adapter matrices','Only the prompt text','Every original weight by default'],0,'The adapter matrices train while most original weights remain frozen. The base is still needed.'),
      ('retrieval','Give the model useful notes','See retrieval and generation as separate steps.','Retrieval',25,'Lab 5',
       ['python lab.py rag --prompt "How do I fix CUDA out of memory?" --retrieve-only','python lab.py rag --prompt "How do I fix CUDA out of memory?"'],
       'Does ordinary retrieval update model weights?',['Yes, every retrieved document trains it','No, retrieved evidence is added to context','Only when a citation is included'],1,'Retrieval supplies context. A separate training process would be required to update learned weights.'),
      ('scratch','Train from random weights','Build a tiny byte-level language model from the beginning.','Pretraining',40,'Lab 6',
       ['python scratch_lm.py --steps 100 --output outputs/scratch-{run}'],
       'Why does the toy model use a causal mask?',['To hide incorrect labels','To stop a position from seeing future tokens','To reduce the vocabulary to words'],1,'Future tokens must not leak into the prediction of what comes next.'),
      ('compression','Make the numbers smaller','Measure storage and reconstruction error.','Quantization',25,'Lab 7',
       ['python mechanisms.py --demo quantization','python mechanisms.py --demo quantization --qmax 7'],
       'Do four-bit weights imply four-bit total runtime memory?',['Yes','No; other tensors and overhead still exist','Only on a laptop'],1,'Weights are one part of the memory budget. Activations, caches and buffers also matter.'),
      ('distillation','Learn from a teacher','Understand teacher signals, student learning and pruning.','Distillation',30,'Lab 8',
       ['python mechanisms.py --demo distillation'],
       'What changes during student distillation?',['The selected student parameters','Only the teacher prompt','The original evaluation answers'],0,'The student learns from teacher-provided signals. Evaluation must remain independent.'),
      ('code','Evaluate a coding model','Use executable tests to judge generated programs.','Specialization',35,'Lab 9',
       ['python lab.py lock --model Qwen/Qwen2.5-Coder-0.5B-Instruct','python lab.py infer --model Qwen/Qwen2.5-Coder-0.5B-Instruct --prompt "Write a Python function that returns the maximum of a nonempty list." --max-new-tokens 128'],
       'What is stronger evidence of code correctness?',['A fluent explanation','Passing suitable independently written tests','A long generated function'],1,'Tests provide task evidence, though no small test suite proves correctness in every case.'),
      ('vision','Connect images and language','Upload your own image and inspect the actual response.','Vision',30,'Lab 10',
       ['python lab.py lock --model HuggingFaceTB/SmolVLM-256M-Instruct'],
       'Can a text-only model see pixels from a filename?',['Yes, if the filename is descriptive','No; an image-capable processing path is needed','Only with a longer prompt'],1,'A vision-language model needs an image representation and a way to connect it to language.'),
      ('architectures','Experiment with other families','Train tiny expert, state-space and hybrid models.','Architecture',40,'Lab 11',
       ['python mechanisms.py --demo architectures','python scratch_lm.py --architecture ssm --steps 20 --output outputs/ssm-{run}','python scratch_lm.py --architecture moe --steps 20 --output outputs/moe-{run}','python scratch_lm.py --architecture hybrid --steps 20 --output outputs/hybrid-{run}'],
       'What does active expert count tell you?',['All model weight storage','Which expert computation is used per token','The guaranteed speed on your computer'],1,'Active computation and total stored parameters are different resource budgets.'),
      ('cloud','Move an experiment to a graphics processor','Plan a rental, verify the device and preserve your work.','Optional cloud',30,'Lab 12',
       ['python lab.py doctor'],
       'Does stopping a rented instance always end storage charges?',['Yes','No; storage can remain billable','Only if the model finished training'],1,'Save your work and review the provider’s remaining billable resources, including retained storage.'),
      ('capstone','Prove your engineering decision','Bring evaluation, artifacts and limitations together.','Assessment',45,'Lab 13',
       ['python lab.py eval --split test --output outputs/baseline-test.json','python lab.py eval --adapter {adapter} --split test --output outputs/adapter-test.json','python compare_reports.py outputs/baseline-test.json outputs/adapter-test.json'],
       'When should you inspect final test performance?',['After every tuning change','After choosing settings using validation','Before creating training data, then tune to it'],1,'Keep a final test independent of the choices it is intended to assess.'),
    ]
    lessons=[]
    for index,row in enumerate(metadata):
        ident,title,desc,topic,minutes,prefix,commands,q,options,correct,explanation=row
        extra=[]
        if ident=='math': extra=['Attention without']
        lessons.append({'id':ident,'number':index+1,'title':title,'description':desc,'topic':topic,'minutes':minutes,
            'html':section(prefix+' ',*extra),'commands':commands,
            'quiz':{'question':q,'options':options,'correct':correct,'explanation':explanation}})
    from curriculum_expansion import additions, atlas_markdown, write_book
    from project_builds import language_project_section
    for lesson in lessons:
        lesson['html']+=render(language_project_section(lesson['id']))
        lesson.update(track='Language models',kind='implementation',scope='Local teaching implementation')
        if lesson['id'] in ('inference','code','vision'): lesson['scope']='Run and evaluate a pretrained model'
        if lesson['id'] in ('math','compression','distillation'): lesson['scope']='Small numerical mechanism demonstration'
        if lesson['id']=='cloud':
            lesson.update(kind='concept',scope='Cloud procedure and cost-planning activity; rental not executed in verification',commands=[],html=render((ROOT/'course/Vast-ai-Lab.md').read_text(encoding='utf-8')))
    lessons.extend(additions(render,len(lessons)+1))
    from learning_tracks import additions as build_additions, references as build_references
    lessons.extend(build_additions(render,len(lessons)+1))
    from modern_curriculum import additions as modern_additions, references as modern_references
    lessons.extend(modern_additions(render,len(lessons)+1))
    from transformer_lesson import additions as transformer_additions
    lessons.extend(transformer_additions(render,len(lessons)+1))
    # Number the complete learning path, not the order source modules were added.
    # Stable IDs keep existing links, completion records and evidence notes valid.
    track_order=['Foundations','Language models','Build and retrieve','Modern language engineering']
    lessons.sort(key=lambda lesson:track_order.index(lesson['track']))
    for number,lesson in enumerate(lessons,1):
        lesson['number']=number
    from lesson_references import REFERENCES, references_markdown
    assert set(REFERENCES)=={lesson['id'] for lesson in lessons}, 'Reference coverage differs from lessons'
    for lesson in lessons:
        lesson['references']=REFERENCES[lesson['id']]
        lesson['html']+='<section class="lesson-references">'+render(references_markdown(lesson['id']))+'</section>'
    write_book()
    # Keep a single definition for terms shared by the two original glossaries.
    glossary_sources=[body for title,body in sections.items() if title.startswith('Acronyms abbreviations')]
    glossary_sources.append((ROOT/'course/Foundation-Glossary.md').read_text(encoding='utf-8'))
    terms={}
    for source in glossary_sources:
        for line in source.splitlines():
            if line.startswith('|'):
                cells=[v.strip() for v in line.strip('|').split('|')]
                if len(cells)!=3 or all(re.fullmatch(r'[:\- ]+',c) for c in cells):continue
                key=cells[0].strip('`').casefold()
                if key in ('short form or term','term','abbreviation','short form','symbol or term'):continue
                terms.setdefault(key,cells)
    glossary='# Every abbreviation, explained\n\n| Term | Expanded name | Meaning in this lab |\n|---|---|---|\n'
    glossary+='\n'.join('| '+' | '.join(terms[k])+' |' for k in sorted(terms))
    data={'lessons':lessons,'references':{
        'lab-workflow':{'title':'Run, compare, save, and explain','html':render((ROOT/'course/Lab-Workflow.md').read_text(encoding='utf-8'))},
        'build-plans':{'title':'More model build plans','html':render((ROOT/'course/Concept-Build-Plans.md').read_text(encoding='utf-8'))},
        'ml-atlas':{'title':'Machine learning atlas','html':render(atlas_markdown())},
        'windows':{'title':'Windows installation','html':render((ROOT/'WINDOWS-SETUP.md').read_text(encoding='utf-8'))},
        'foundations':{'title':'Start with the concepts','html':section('Before the laboratories','A decision algorithm','The learning algorithm')},
        'families':{'title':'Language model atlas','html':render('**Concept** — A representative map of language modeling within machine learning. Implementation links state exactly which exercises run in this package.')+section('Language models are machine learning','Model families explained','How specialization','Reference map')},
        'glossary':{'title':'Every abbreviation, explained','html':render(glossary)},
        'troubleshooting':{'title':'Find the cause of an error','html':section('Troubleshooting','Guided understanding')},
        'sources':{'title':'Primary sources and further reading','html':section('Sources and further study')}},
        'welcome':render(chunks[0])}
    data['references'].update(build_references(render))
    data['references'].update(modern_references(render))
    # The standalone language workbook uses its own zero-based lab numbers.
    # Resolve its headings and cross-references to the website's final numbers.
    lesson_numbers={lesson['id']:lesson['number'] for lesson in lessons}
    workbook_numbers={int(row[5].split()[1]):lesson_numbers[row[0]] for row in metadata}
    def website_lab_numbers(content):
        return re.sub(r'\bLab (\d+)\b',lambda m:'Lab '+str(workbook_numbers.get(int(m[1]),int(m[1]))),content)
    for entry in [*lessons,*data['references'].values()]:
        entry['html']=website_lab_numbers(entry['html'])
    data['welcome']=website_lab_numbers(data['welcome'])
    (ROOT/'public/course.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    assert all(x['html'] for x in lessons)
    print('Built',len(lessons),'complete lessons and',len(data['references']),'reference chapters.')

if __name__=='__main__':main()
