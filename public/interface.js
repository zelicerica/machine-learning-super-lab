'use strict';
/* Local interface helpers. No remote fonts, editors, analytics, or CDN calls. */
const TRACKS = {
  'Foundations': {icon:'01', text:'Build intuition, choose a baseline, and learn the model families.', tone:'violet'},
  'Language models': {icon:'02', text:'Run a small model. Train, adapt, and inspect its predictions.', tone:'cyan'},
  'Build and retrieve': {icon:'03', text:'Build neural systems and connect them to useful evidence.', tone:'mint'},
  'Modern language engineering': {icon:'04', text:'Attention, post-training, evaluation, and model serving.', tone:'amber'}
};
let navQuery='', researchTopic='All', researchQuery='', resultReports=[], diagramScale=1;
function trackList(){return Object.keys(TRACKS).filter(t=>course.lessons.some(l=>l.track===t));}
function completed(items){return items.filter(l=>state.done.includes(l.id)).length;}
function nextLesson(items=course.lessons){return items.find(l=>l.id===state.last&&!state.done.includes(l.id))||items.find(l=>!state.done.includes(l.id))||items[0];}
function renderNavigation(){
  $('#lessonCount').textContent=course.lessons.length;
  const current=route(), active=course.lessons.find(l=>current==='lesson/'+l.id), q=navQuery.toLowerCase();
  $('#lessonNav').innerHTML=trackList().map(t=>{
    const all=course.lessons.filter(l=>l.track===t);
    const visible=all.filter(l=>(l.title+' '+l.topic+' '+l.html.replace(/<[^>]*>/g,' ')).toLowerCase().includes(q));
    if(!visible.length)return '';
    const open=q||active?.track===t||(state.navOpen||[]).includes(t)||(!active&&nextLesson()?.track===t);
    return '<details class="nav-group" data-group="'+esc(t)+'" '+(open?'open':'')+'><summary><span>'+esc(t)+'</span><small>'+completed(all)+'/'+all.length+'</small></summary>'+visible.map(l=>
      '<a href="#lesson/'+l.id+'" class="lesson-link '+(current==='lesson/'+l.id?'active ':'')+(state.done.includes(l.id)?'done':'')+'" '+(current==='lesson/'+l.id?'aria-current="page"':'')+'><span class="lesson-number">'+(state.done.includes(l.id)?'✓':String(l.number).padStart(2,'0'))+'</span><span>'+esc(l.title)+'<small class="nav-kind">'+(l.kind==='concept'?'Concept':'Runnable lab')+'</small></span></a>').join('')+'</details>';
  }).join('')||'<p class="search-empty" role="status">No matching lessons.</p>';
  document.querySelectorAll('[data-nav]').forEach(a=>a.classList.toggle('active',a.dataset.nav===current));
  $('#sidebar').classList.remove('open');$('#menuButton').setAttribute('aria-expanded','false');
  $('#releaseLabel').textContent=(session.workspace?.name||'Midnight')+' · '+course.lessons.length+' lessons';
  $('#releaseBadge').textContent=course.lessons.length+' lessons · '+(session.workspace?.name||'Midnight');
}
function lessonCard(l){
  return '<a class="course-card '+(state.done.includes(l.id)?'done':'')+'" href="#lesson/'+l.id+'"><span class="index">'+(state.done.includes(l.id)?'✓':String(l.number).padStart(2,'0'))+'</span><div><span class="kind-tag '+l.kind+'">'+kindLabel(l)+'</span><h3>'+esc(l.title)+'</h3><p>'+esc(l.description)+'</p></div><span class="card-meta">'+esc(l.topic)+'<span>↗</span></span></a>';
}
function renderHome(){
  const next=nextLesson(), done=completed(course.lessons), total=course.lessons.length;
  $('#breadcrumb').textContent='Learning workspace';
  const cards=trackList().map(t=>{const items=course.lessons.filter(l=>l.track===t),n=completed(items),info=TRACKS[t];
    return '<button class="track-card '+info.tone+'" data-action="track" data-track="'+esc(t)+'"><span class="track-number">'+info.icon+' / LEARNING TRACK</span><h3>'+esc(t)+'</h3><p>'+info.text+'</p><div class="track-bottom"><span>'+n+' of '+items.length+' complete</span><span aria-hidden="true">↗</span></div><progress max="'+items.length+'" value="'+n+'" aria-label="'+esc(t)+' completion"></progress></button>';}).join('');
  return '<section class="hero"><div><span class="eyebrow">YOUR MACHINE LEARNING WORKSPACE</span><h1>Build models.<br><em>Understand every step.</em></h1><p class="page-desc">From your first gradient to modern language engineering. Read the idea, run the experiment, and keep the evidence.</p><div class="hero-actions"><a class="button primary" href="#lesson/'+next.id+'">'+(done||state.last?'Continue learning':'Start your first lab')+' <span>↗</span></a><a class="button secondary" href="#ref/research-landmarks">Explore the research</a></div><div class="hero-meta"><span class="status-dot"></span> LOCAL PYTHON <span>·</span> '+total+' LESSONS <span>·</span> YOUR OWN EXPERIMENTS</div></div></section>'+
  '<section class="resume-strip"><div><span class="eyebrow">UP NEXT</span><span class="tag">Lesson '+String(next.number).padStart(2,'0')+'</span></div><div><h2>'+esc(next.title)+'</h2><p>'+esc(next.description)+'</p></div><a class="button secondary" href="#lesson/'+next.id+'">Open lesson →</a></section>'+
  '<section class="stats" aria-label="Course status"><div class="stat-card"><span class="stat-label">YOUR PROGRESS</span><div class="stat-value">'+done+' <small>/ '+total+' complete</small></div><progress class="course-progress" value="'+done+'" max="'+total+'"></progress></div><div class="stat-card"><span class="stat-label">RESEARCH & REFERENCE</span><div class="stat-value">'+Object.keys(course.references).length+' <small>connected guides</small></div><a class="text-link" href="#ref/research-landmarks">Trace the ideas →</a></div><div class="stat-card"><span class="stat-label">LOCAL RUNTIME</span><div class="stat-value">'+(session.runtime.ready?'Ready to run':'Setup needed')+'</div><p class="stat-note">Python '+esc(session.runtime.python||'')+' · CPU by default</p></div></section>'+
  '<section><div class="section-heading"><div><span class="eyebrow">CHOOSE YOUR NEXT SKILL</span><h2>Four paths. One connected lab.</h2></div><a class="text-link" href="#ref/ml-atlas">View the model atlas →</a></div><div class="track-grid">'+cards+'</div></section>'+
  '<details id="lessonCatalog" class="catalog" '+(trackFilter!=='All'?'open':'')+'><summary><span>Browse the lesson catalog</span><small>'+total+' lessons</small></summary><div class="track-filters">'+['All',...trackList()].map(t=>'<button class="button secondary small '+(t===trackFilter?'selected':'')+'" data-action="track" data-track="'+esc(t)+'">'+esc(t)+'</button>').join('')+'</div><div class="course-list">'+course.lessons.filter(l=>trackFilter==='All'||l.track===trackFilter).map(lessonCard).join('')+'</div></details>'+
  '<section class="home-resources"><a class="resource-card" href="#workbench"><span class="resource-icon cyan">&gt;_</span><div><h3>Your Python workbench</h3><p>Write, run, inspect, repeat.</p></div><span>↗</span></a><a class="resource-card" href="#ref/research-landmarks"><span class="resource-icon violet">⌕</span><div><h3>The ideas behind the models</h3><p>Landmark papers, connected to your lessons.</p></div><span>↗</span></a><a class="resource-card" href="#workspace"><span class="resource-icon mint">◇</span><div><h3>Know your workspace</h3><p>Release, source folder, and upgrade guidance.</p></div><span>↗</span></a></section>';
}
function workspaceView(){
  $('#breadcrumb').textContent='About this workspace';const w=session.workspace;
  if(!w)return '<article class="reading"><h1>Restart the updated server</h1><p>This frontend is connected to an older server. Stop the launcher with Ctrl+C and run start-lab.cmd from the newly extracted folder.</p></article>';
  return '<section class="reference-head"><span class="eyebrow">KNOW WHAT IS RUNNING</span><h1 class="page-title">About this workspace.</h1><p class="page-desc">This information comes from the Python process serving this page.</p></section><article class="reading"><div class="workspace-facts">'+[['Release',w.release],['Course',w.lessons+' lessons · '+w.references+' reference guides'],['Serving folder',w.project_folder],['Catalog file',w.catalog_file],['Python executable',w.python]].map(([k,v])=>'<div><span>'+k+'</span><code>'+esc(v)+'</code></div>').join('')+'</div>'+
  (!w.catalog_matches_release?'<div class="callout warning"><strong>Catalog and release differ.</strong><p>This release expects '+w.expected_lessons+' lessons. Run <code>python build_course.py</code> in the serving folder, then restart.</p></div>':'<div class="callout success"><strong>✓ The catalog matches this release.</strong></div>')+
  '<h2>Why would another copy show 34 lessons?</h2><p>Each extracted folder is a separate copy. GitHub updates and new ZIP downloads do not change an already-running Python server. A server still listening on port 8765 can continue to serve the old folder.</p><ol><li>Stop your old launcher window with <strong>Ctrl+C</strong>.</li><li>Extract the entire updated archive to a separate folder. Keep the old folder as your backup.</li><li>Open the new folder containing <code>server.py</code> and <code>start-lab.cmd</code>, then launch it. Reuse an existing environment with <code>SML_LAB_PYTHON</code> if needed.</li><li>Open the address printed by the launcher. Check the release and serving folder here.</li></ol><p>Your browser progress stays associated with the exact address and browser profile. Keep using <code>http://localhost:8765</code> for the same progress. Copy your existing <code>lab/outputs</code>, <code>lab/inputs</code>, and <code>runs</code> into a fresh folder only when the destination is empty; retain your original backup. Do not move a virtual environment: point to its existing interpreter instead.</p><p><a href="#ref/windows">Read the full Windows setup guide →</a></p></article>';
}
function highlightCode(text){
  const re=/(#[^\n]*|\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:False|True|None|false|true|null|def|class|return|import|from|as|if|elif|else|for|while|in|with|try|except|raise|async|await|const|let|function|new|and|or|not|pass|break)\b|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]\w*(?=\s*\())/g;
  let out='',last=0;for(const m of text.matchAll(re)){
    out+=esc(text.slice(last,m.index));const t=m[0];
    const cls=/^(#|\/\/)/.test(t)?'comment':/^["']/.test(t)?'string':/^\d/.test(t)?'number':/^\w+$/.test(t)&&/^\s*\(/.test(text.slice(m.index+t.length))?'function':'keyword';
    out+='<span class="syntax-'+cls+'">'+esc(t)+'</span>';last=m.index+t.length;
  }return out+esc(text.slice(last));
}
function updateEditor(){
  const input=$('#codeEditor'),paint=$('#editorHighlight');if(!input||!paint)return;
  paint.innerHTML=highlightCode(input.value)+'\n';
  $('#editorLines').textContent=Array.from({length:input.value.split('\n').length},(_,i)=>i+1).join('\n');
  paint.parentElement.scrollTop=input.scrollTop;paint.parentElement.scrollLeft=input.scrollLeft;$('#editorLines').scrollTop=input.scrollTop;
}
function enhanceWorkbench(){
  const editor=$('#codeEditor');if(!editor)return;
  const panel=editor.closest('.editor-panel'),grid=document.createElement('div');grid.className='workbench-grid';
  panel.before(grid);const column=document.createElement('section');column.className='workbench-code';grid.append(column);column.append(panel);
  const quick=$('.workbench-quick'),source=$('#sourceDetails');column.append(quick,source);
  const guide=document.createElement('aside');guide.className='workbench-guide';
  guide.innerHTML='<span class="eyebrow">YOUR EXPERIMENT LOOP</span><h2>One change at a time.</h2><ol class="build-steps"><li><strong>Predict</strong><p>Write down what you expect to change.</p></li><li><strong>Run</strong><p>Choose an example, edit the code, and run Python.</p></li><li><strong>Inspect</strong><p>Read the actual process output in the console below.</p></li><li><strong>Keep evidence</strong><p>Save the script and log. Compare model reports in Results.</p></li></ol><a class="button secondary full" href="#results">View experiment results →</a>';
  grid.append(guide);
  const wrap=document.createElement('div');wrap.className='code-editor';editor.before(wrap);
  wrap.innerHTML='<pre id="editorLines" class="editor-lines" aria-hidden="true"></pre><pre class="editor-paint" aria-hidden="true"><code id="editorHighlight"></code></pre>';
  wrap.append(editor);editor.setAttribute('wrap','off');editor.addEventListener('scroll',updateEditor);updateEditor();
  const copy=document.createElement('button');copy.className='button secondary small';copy.dataset.action='copy-editor';copy.textContent='Copy code';$('.editor-footer').prepend(copy);
}
function enhanceReading(){
  document.querySelectorAll('.reading .code-block').forEach((block,i)=>{
    const code=block.querySelector('code');if(!code)return;
    const raw=code.textContent;block.dataset.code=raw;block.id='code-example-'+i;
    code.innerHTML=raw.split('\n').map((rawLine,n)=>{const line=highlightCode(rawLine);return '<span class="code-line"><span class="line-number" aria-hidden="true">'+(n+1)+'</span>'+line+'</span>';}).join('\n');
    const label=block.querySelector('.code-label');if(label)label.insertAdjacentHTML('beforeend','<button class="text-button" data-action="copy-code" data-target="'+block.id+'">Copy</button>');
  });
  document.querySelectorAll('.reading p').forEach(p=>{
    const first=p.firstElementChild;if(first?.tagName!=='STRONG')return;
    const text=first.textContent.toLowerCase();
    if(/worked|numerical|example/.test(text))p.classList.add('learning-callout','example-callout');
    else if(/assessment|checkpoint|explain/.test(text))p.classList.add('learning-callout','evidence-callout');
    else if(/limitation|warning|scope|production/.test(text))p.classList.add('learning-callout','warning-callout');
    else if(/algorithm|build from|change one|inspect/.test(text))p.classList.add('learning-callout','build-callout');
  });
  const article=$('.lesson-layout .reading'),side=$('.lesson-side');
  if(article&&side){
    const headings=[...article.querySelectorAll('h2,h3,h4')];const first=article.querySelector('h2');
    if(first&&first===article.firstElementChild)first.classList.add('chapter-opening');
    if(headings.length){const outline=document.createElement('nav');outline.className='lesson-outline';outline.setAttribute('aria-label','On this page');outline.innerHTML='<span class="eyebrow">ON THIS PAGE</span>'+headings.slice(0,16).map((h,i)=>{h.id='reading-section-'+i;return '<button data-action="section" data-target="'+h.id+'">'+esc(h.textContent)+'</button>';}).join('');side.prepend(outline);}
  }
}
function configureDiagrams(){
  const light=document.documentElement.dataset.theme==='light';
  mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',
    themeVariables:{background:light?'#ffffff':'#10192c',primaryColor:light?'#ede9fe':'#28284d',primaryTextColor:light?'#17233a':'#e6edf7',primaryBorderColor:light?'#6d5acb':'#a78bfa',secondaryColor:light?'#e0f2fe':'#15364b',tertiaryColor:light?'#d1fae5':'#16382f',lineColor:light?'#4b617e':'#8da9c8',textColor:light?'#17233a':'#e6edf7',edgeLabelBackground:light?'#ffffff':'#131c31',fontFamily:'Segoe UI, system-ui, sans-serif',fontSize:'16px',clusterBkg:light?'#f3f6fc':'#131c31',clusterBorder:light?'#cad5e5':'#354563'},
    flowchart:{htmlLabels:false,useMaxWidth:true}});
}
function decorateDiagrams(){
  document.querySelectorAll('.diagram').forEach((box,i)=>{
    const svg=box.querySelector('svg');if(!svg)return;
    box.id='diagram-'+i;
    if(!box.querySelector('.diagram-tools')){const tools=document.createElement('div');tools.className='diagram-tools';tools.innerHTML='<span>INTERACTIVE DIAGRAM</span><button class="button secondary small" data-action="expand-diagram" data-target="'+box.id+'">Expand ↗</button>';box.prepend(tools);}
    svg.querySelectorAll('.node').forEach(n=>{
      const t=n.textContent.toLowerCase();let role='';
      if(/\b(input|data|tokens|queries|keys|values)\b/.test(t))role='input';
      else if(/\b(loss|gradient|reward|objective)\b/.test(t))role='signal';
      else if(/\b(output|prediction|answer|completion)\b/.test(t))role='output';
      if(role)n.setAttribute('data-role',role);
    });
  });
}
function openDiagram(id){
  const source=document.getElementById(id)?.querySelector('svg');if(!source)return;
  const copy=source.cloneNode(true),ids=[copy,...copy.querySelectorAll('[id]')].filter(n=>n.id).map(n=>n.id).sort((a,b)=>b.length-a.length);
  const rename=new Map(ids.map(id=>[id,id+'-expanded']));
  for(const node of [copy,...copy.querySelectorAll('*')]){
    for(const attr of [...node.attributes]){
      let value=attr.value;
      if(attr.name==='id')value=rename.get(value)||value;
      else for(const [oldId,newId] of rename){value=value.replaceAll('url(#'+oldId+')','url(#'+newId+')');if(value==='#'+oldId)value='#'+newId;}
      node.setAttribute(attr.name,value);
    }
    if(node.tagName.toLowerCase()==='style')node.textContent=node.textContent.replaceAll('#'+source.id,'#'+rename.get(source.id));
  }
  copy.style.maxWidth='none';copy.style.width='100%';copy.style.height='auto';
  $('#diagramCanvas').replaceChildren(copy);diagramScale=1;$('#diagramDialog').showModal();
}
function zoomDiagram(delta){diagramScale=delta===0?1:Math.min(4,Math.max(.5,diagramScale+delta));const svg=$('#diagramCanvas svg');if(svg)svg.style.width=diagramScale*100+'%';}
function renderResearch(r){
  $('#breadcrumb').textContent='Landmark research';
  const doc=new DOMParser().parseFromString(r.html,'text/html');const rows=[];let lastTable=null;
  doc.querySelectorAll('table').forEach((table,i)=>{
    if(!table.querySelector('a[href^="http"]'))return;
    let h=table.closest('.table-scroll')||table;while(h.previousElementSibling&&!/^H[1-4]$/.test(h.previousElementSibling.tagName))h=h.previousElementSibling;
    const topic=h.previousElementSibling?.textContent||'Research';lastTable=table.closest('.table-scroll')||table;
    [...table.querySelectorAll('tr')].slice(1).forEach(tr=>{const cells=tr.querySelectorAll('td');if(cells.length>=3)rows.push({topic,title:cells[0].innerHTML,idea:cells[1].innerHTML,connection:cells[2].innerHTML,text:tr.textContent.toLowerCase()});});
  });
  let tail='';for(let node=lastTable?.nextElementSibling;node;node=node.nextElementSibling)tail+=node.outerHTML;
  const topics=[...new Set(rows.map(r=>r.topic))];
  return '<section class="reference-head"><span class="eyebrow">THE IDEAS BEHIND THE MODELS</span><h1 class="page-title">Landmark research.</h1><p class="page-desc">Follow the questions that shaped machine learning. Each paper connects an idea to something you can study or build in the lab. This is a curated reading path; the small teaching exercises are not reproductions of full research results.</p><div class="research-controls"><label>Find a paper<input id="researchSearch" type="search" value="'+esc(researchQuery)+'" placeholder="Attention, LoRA, reinforcement…"></label><label>Research area<select id="researchTopic"><option>All</option>'+topics.map(t=>'<option '+(researchTopic===t?'selected':'')+'>'+esc(t)+'</option>').join('')+'</select></label></div><p id="researchCount" class="lesson-note" role="status"></p></section><div class="research-timeline">'+rows.map((r,i)=>'<article class="paper-card" data-topic="'+esc(r.topic)+'" data-search="'+esc(r.text)+'"><span class="timeline-dot"></span><span class="eyebrow">'+esc(r.topic)+'</span><h2>'+r.title+'</h2><p>'+r.idea+'</p><div class="paper-connection"><span>IN YOUR LAB</span><p>'+r.connection+'</p></div></article>').join('')+'</div><article class="reading research-tail">'+tail+'</article><p class="lesson-note"><a href="/download/research.md">Download the complete reference guide →</a></p>';
}
function filterResearch(){
  const cards=[...document.querySelectorAll('.paper-card')];let visible=0;
  for(const c of cards){c.hidden=(researchTopic!=='All'&&c.dataset.topic!==researchTopic)||!c.dataset.search.includes(researchQuery.toLowerCase());if(!c.hidden)visible++;}
  if($('#researchCount'))$('#researchCount').textContent=visible+' of '+cards.length+' references';
}
function numericMetrics(report){
  const values={};function walk(obj,prefix='',depth=0){if(depth>3||!obj||typeof obj!=='object'||Array.isArray(obj))return;
    for(const [k,v] of Object.entries(obj)){const key=prefix?prefix+'.'+k:k;if(typeof v==='number'&&Number.isFinite(v))values[key]=v;else if(v&&typeof v==='object')walk(v,key,depth+1);}}
  walk(report.metrics||{});for(const k of ['accuracy','schema_valid_rate','validation_loss_before','validation_loss_after'])if(Number.isFinite(report[k]))values[k]=report[k];return values;
}
function metricLabel(key){return key.replaceAll('_',' ').replaceAll('.',' · ');}
function formatMetric(v){return Number.isInteger(v)?v.toLocaleString():v.toLocaleString(undefined,{maximumSignificantDigits:5});}
function reportSummary(r){
  const metrics=Object.entries(numericMetrics(r.data)).slice(0,8);
  return '<div class="report-overview"><div><span class="eyebrow">'+esc(r.data.split||'Measured run')+'</span><h3>'+esc((r.data.experiment||'Language model evaluation').replaceAll('-',' '))+'</h3><p class="report-path">'+esc(r.path)+'</p></div><span class="status-tag '+(r.data.reload_verified===true?'success':r.data.reload_verified===false?'failure':'neutral')+'">'+(r.data.reload_verified===true?'✓ Reload verified':r.data.reload_verified===false?'! Reload check failed':'Recorded measurements')+'</span></div><div class="metric-grid">'+metrics.map(([k,v])=>'<div><span>'+esc(metricLabel(k))+'</span><strong>'+formatMetric(v)+'</strong></div>').join('')+'</div>'+(r.data.limitations?'<p class="result-limits"><strong>Interpretation limits:</strong> '+esc(r.data.limitations)+'</p>':'');
}
function comparisonControls(data){
  if(data.length<2)return '';
  return '<section class="comparison-panel"><div class="section-heading"><div><span class="eyebrow">COMPARE YOUR EVIDENCE</span><h2>Baseline and experiment.</h2></div></div><div class="compare-selects">'+['Baseline','Experiment'].map((label,n)=>'<label>'+label+'<select id="compare'+label+'">'+data.map((r,i)=>'<option value="'+i+'" '+(i===n?'selected':'')+'>'+esc(r.path)+'</option>').join('')+'</select></label>').join('')+'</div><div id="comparisonTable"></div></section>';
}
function updateComparison(){
  if(!$('#comparisonTable'))return;const a=resultReports[Number($('#compareBaseline').value)],b=resultReports[Number($('#compareExperiment').value)];
  if(!a||!b)return;
  if((a.data.experiment||'language')!==(b.data.experiment||'language')||a.data.split!==b.data.split){$('#comparisonTable').innerHTML='<p class="callout warning">Choose reports from the same experiment and evaluation split. Different tasks or partitions are not directly comparable.</p>';return;}
  const x=numericMetrics(a.data),y=numericMetrics(b.data),keys=Object.keys(x).filter(k=>k in y);
  $('#comparisonTable').innerHTML='<p class="lesson-note">A shared metric name does not guarantee a fair comparison. Check dataset, units, model, seed, sampling, and settings in both raw reports. Positive change means a larger number, not necessarily a better result.</p><div class="table-scroll"><table><thead><tr><th>Measurement</th><th>Baseline</th><th>Experiment</th><th>Change</th></tr></thead><tbody>'+keys.map(k=>'<tr><td>'+esc(metricLabel(k))+'</td><td>'+formatMetric(x[k])+'</td><td>'+formatMetric(y[k])+'</td><td class="metric-delta">'+(y[k]-x[k]>0?'+':'')+formatMetric(y[k]-x[k])+'</td></tr>').join('')+'</tbody></table></div>'+(keys.length?'':'<p>No shared numeric metrics.</p>');
}
function afterRender(){
  updateBackgroundRoute();
  enhanceReading();enhanceWorkbench();filterResearch();updateComparison();enhanceMotion();
  document.querySelectorAll('a[href^="http"]').forEach(a=>{a.target="_blank";a.rel="noopener noreferrer";});
  document.querySelectorAll('.practice-card').forEach((card,i)=>{card.classList.add('build-card');});
}
function setTheme(theme){
  document.documentElement.dataset.theme=theme;localStorage.setItem('super-lab-theme',theme);
  if($('#themeToggle')){$('#themeToggle').textContent=theme==='dark'?'Light theme':'Dark theme';$('#themeToggle').setAttribute('aria-label','Switch to '+(theme==='dark'?'light':'dark')+' theme');}
  configureDiagrams();
}
async function copyText(value){
  try{await navigator.clipboard.writeText(value);toast('Copied to clipboard.');}
  catch{toast('Clipboard is unavailable. Select the code and copy it manually.');}
}
function initInterface(){
  initBackgroundVideo();
  $('#main').addEventListener('keydown',e=>{if(e.target.getAttribute('role')!=='tab'||!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const tabs=[...document.querySelectorAll('[role=tab]')],i=tabs.indexOf(e.target),n=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[n].click();document.querySelectorAll('[role=tab]')[n]?.focus();});
  $('.skip-link').onclick=e=>{e.preventDefault();$('#main').focus();$('#main').scrollIntoView();};
  setTheme(localStorage.getItem('super-lab-theme')==='light'?'light':'dark');
  $('#themeToggle').onclick=async()=>{setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');await render();};
  $('#lessonSearch').addEventListener('input',e=>{navQuery=e.target.value;renderNavigation();if(innerWidth<=760){$('#sidebar').classList.add('open');$('#menuButton').setAttribute('aria-expanded','true');}});
  $('#lessonNav').addEventListener('toggle',e=>{if(!e.target.matches('.nav-group'))return;state.navOpen=[...$('#lessonNav').querySelectorAll('.nav-group[open]')].map(n=>n.dataset.group);save();},true);
  $('#consoleHeight').addEventListener('input',e=>document.documentElement.style.setProperty('--console-height',e.target.value+'px'));
  $('#diagramClose').onclick=()=>$('#diagramDialog').close();$('#diagramZoomIn').onclick=()=>zoomDiagram(.25);$('#diagramZoomOut').onclick=()=>zoomDiagram(-.25);$('#diagramZoomReset').onclick=()=>zoomDiagram(0);
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){$('#sidebar').classList.remove('open');$('#menuButton').setAttribute('aria-expanded','false');}});
  $('#main').addEventListener('input',e=>{if(e.target.id==='codeEditor')updateEditor();if(e.target.id==='researchSearch'){researchQuery=e.target.value;filterResearch();}});
  $('#main').addEventListener('change',e=>{if(e.target.id==='templateSelect')updateEditor();if(e.target.id==='researchTopic'){researchTopic=e.target.value;filterResearch();}if(e.target.id.startsWith('compare'))updateComparison();});
  $('#main').addEventListener('click',e=>{const b=e.target.closest('[data-action]');if(!b)return;
    if(b.dataset.action==='copy-code')copyText(document.getElementById(b.dataset.target).dataset.code);
    if(b.dataset.action==='copy-editor')copyText($('#codeEditor').value);
    if(b.dataset.action==='expand-diagram')openDiagram(b.dataset.target);
    if(b.dataset.action==='section')document.getElementById(b.dataset.target)?.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'start'});
  });
}
async function renderResults(version){
  $('#breadcrumb').textContent='Experiment results';$('#main').innerHTML='<p class="loading">Reading your experiment reports…</p>';
  const data=await api('/api/results');if(version!==routeVersion)return;resultReports=data;
  const training=data.find(r=>(r.data.loss_log||r.data.history)?.length);
  $('#main').innerHTML='<section class="reference-head"><span class="eyebrow">YOUR EXPERIMENT NOTEBOOK</span><h1 class="page-title">Make the evidence visible.</h1><p class="page-desc">Actual measurements from your own runs. Inspect the results, compare a change, and keep the limitations in view.</p></section>'+
    (data.length?'<div class="results-grid"><div class="result-card"><h3>Saved reports</h3><div class="metric">'+data.length+'</div><p>Most recent reports in this project</p></div><div class="result-card"><h3>Verified reloads</h3><div class="metric">'+data.filter(r=>r.data.reload_verified===true).length+'</div><p>Reports explicitly confirming model reload</p></div><div class="result-card"><h3>Next step</h3><div class="metric metric-label">Compare.</div><p>Look beyond the training loss.</p></div></div>'+comparisonControls(data)+(training?lossChart(training):'')+
    '<div class="section-heading"><h2>Saved measurements</h2><button class="button secondary small" data-action="refresh">Refresh ↻</button></div><div class="report-list">'+data.map(r=>'<section class="report-card">'+reportSummary(r)+reportPlots(r)+'<details class="report-item"><summary>Inspect settings and raw report</summary><pre>'+esc(JSON.stringify(r.data,null,2))+'</pre></details></section>').join('')+'</div>':
    '<div class="empty-state"><span class="resource-icon cyan">▥</span><h2>Your first result starts with a baseline.</h2><p>Run an experiment to produce measurements here. Course reference results remain separate from your own work.</p><a class="button primary" href="#lesson/baseline">Run a baseline →</a></div>');
}
