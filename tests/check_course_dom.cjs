const {JSDOM}=require(process.env.SML_JSDOM_PATH||'jsdom');
const fs=require('fs'),assert=require('assert/strict'),path=require('path');
const root=path.resolve(__dirname,'..');
const course=JSON.parse(fs.readFileSync(root+'/public/course.json'));
const html=fs.readFileSync(root+'/public/index.html','utf8').replace(/<script[^>]*src=[^>]+><\/script>/g,'');
const dom=new JSDOM(html,{url:'http://localhost:8765',runScripts:'outside-only',pretendToBeVisual:true});
const w=dom.window;w.TextEncoder=TextEncoder;w.TextDecoder=TextDecoder;w.scrollTo=()=>{};w.matchMedia=()=>({matches:false,addEventListener:()=>{}});w.HTMLMediaElement.prototype.play=function(){this.dataset.testPlaying='yes';this.dispatchEvent(new w.Event('play'));return Promise.resolve()};w.HTMLMediaElement.prototype.pause=function(){this.dataset.testPlaying='no';this.dispatchEvent(new w.Event('pause'))};w.HTMLElement.prototype.scrollIntoView=()=>{};w.HTMLDialogElement.prototype.showModal=function(){this.open=true};w.HTMLDialogElement.prototype.close=function(){this.open=false};
w.HTMLCanvasElement.prototype.getContext=()=>null;
w.localStorage.setItem('super-lab-v1',JSON.stringify({done:['setup'],notes:{math:'Earlier evidence remains.'}}));
w.fetch=async url=>({ok:true,json:async()=>url==='/course.json'?course:url==='/api/session'?{token:'test',runtime:{ready:true,foundations_ready:true,python:'3.12'},templates:{hello:'print(1)'},workspace:{name:'Transformer Explorer',release:'2026.11-transformer.1',lessons:59,expected_lessons:59,references:Object.keys(course.references).length,project_folder:'/test/lab',catalog_file:'/test/lab/public/course.json',catalog_matches_release:true,python:'/test/python'}}:[]});
const mermaidFile=fs.readFileSync(root+'/public/index.html','utf8').match(/src="([^"]*mermaid[^\"]*)"/)[1];
require('vm').runInContext(fs.readFileSync(path.join(root,'public',mermaidFile),'utf8'),dom.getInternalVMContext());
w.eval('mermaid.run=async({nodes})=>{for(const n of nodes){await mermaid.parse(n.textContent);n.dataset.syntaxValid="true"}}');
for(const file of ['background.js','motion.js','transformer.js','interface.js'])require('vm').runInContext(fs.readFileSync(root+'/public/'+file,'utf8'),dom.getInternalVMContext());
require('vm').runInContext(fs.readFileSync(root+'/public/app.js','utf8'),dom.getInternalVMContext());
(async()=>{
 await new Promise(r=>setTimeout(r,100));
 assert.equal(w.document.querySelectorAll('.course-card').length,59);
 assert.deepEqual(course.lessons.map(l=>l.number),Array.from({length:59},(_,i)=>i+1));
 assert.deepEqual([...new Set(course.lessons.map(l=>l.track))],['Foundations','Language models','Build and retrieve','Modern language engineering']);
 assert.equal(course.lessons[0].id,'learning-signals');
 assert.equal(w.document.querySelector('.hero-actions a').getAttribute('href'),'#lesson/learning-signals');
 assert.deepEqual([...w.document.querySelectorAll('.lesson-link')].map(a=>a.getAttribute('href')),course.lessons.map(l=>'#lesson/'+l.id));
 assert.equal(w.document.querySelector('.lesson-link[href="#lesson/setup"]').classList.contains('done'),true);
 w.eval("state.last='math'");assert.equal(w.eval('nextLesson().id'),'math');assert.equal(w.eval('nextLesson().number'),23);w.eval('delete state.last');
 assert.equal(w.document.querySelectorAll('#labBackgroundVideo').length,1);
 assert.ok(fs.existsSync(root+'/public'+w.document.querySelector('#labBackgroundVideo source').getAttribute('src')));
 const video=w.document.querySelector('#labBackgroundVideo');assert.equal(video.dataset.testPlaying,'yes');
 const tone=w.document.querySelector('#backgroundToneToggle');
 assert.equal(w.document.body.dataset.backgroundTone,'original');assert.equal(tone.textContent,'Dim video');assert.equal(tone.getAttribute('aria-pressed'),'false');
 tone.click();assert.equal(w.document.body.dataset.backgroundTone,'dimmed');assert.equal(tone.textContent,'Original video');assert.equal(tone.getAttribute('aria-pressed'),'true');assert.equal(w.localStorage.getItem('super-lab-background-tone'),'dimmed');assert.equal(video.dataset.testPlaying,'yes');
 w.eval("updateBackgroundTone(localStorage.getItem('super-lab-background-tone'))");assert.equal(w.document.body.dataset.backgroundTone,'dimmed');
 w.location.hash='lesson/math';w.eval('updateBackgroundRoute()');assert.equal(w.document.body.dataset.backgroundTone,'dimmed');
 tone.click();assert.equal(w.document.body.dataset.backgroundTone,'original');assert.equal(w.localStorage.getItem('super-lab-background-tone'),'original');
 w.location.hash='';w.eval('updateBackgroundRoute()');
 w.document.querySelector('#backgroundToggle').click();assert.equal(video.dataset.testPlaying,'no');assert.equal(w.localStorage.getItem('super-lab-background-motion'),'off');
 w.document.querySelector('#backgroundToggle').click();assert.equal(video.dataset.testPlaying,'yes');
 w.eval("backgroundPreference=null;backgroundMedia.matches=true;updateBackgroundPlayback()");assert.equal(video.dataset.testPlaying,'no');
 w.eval("backgroundMedia.matches=false;updateBackgroundPlayback()");assert.equal(video.dataset.testPlaying,'yes');

 w.document.querySelector('[data-track="Modern language engineering"]').click();await new Promise(r=>setTimeout(r,10));assert.equal(w.document.querySelectorAll('.course-card').length,17);
 assert.equal(w.document.documentElement.dataset.theme,'dark');
 w.document.querySelector('#themeToggle').click();await new Promise(r=>setTimeout(r,0));assert.equal(w.document.documentElement.dataset.theme,'light');
 w.document.querySelector('#themeToggle').click();await new Promise(r=>setTimeout(r,0));assert.equal(w.localStorage.getItem('super-lab-theme'),'dark');
 const search=w.document.querySelector('#lessonSearch');search.value='latent caches';search.dispatchEvent(new w.Event('input',{bubbles:true}));assert.ok(w.document.querySelectorAll('.lesson-link').length<59);assert.ok(w.document.querySelectorAll('.lesson-link').length>0);search.value='';search.dispatchEvent(new w.Event('input',{bubbles:true}));assert.equal(w.document.querySelectorAll('.lesson-link').length,59);
 assert.equal(w.document.querySelectorAll('.nav-group').length,4);
 const checks=[];
 for(const [route,body] of [...course.lessons.map(l=>['lesson/'+l.id,l.html]),...Object.entries(course.references).map(([id,r])=>['ref/'+id,r.html])]){
   w.location.hash=route;await new Promise(r=>setTimeout(r,0));await w.eval('render()');await w.eval('diagramChain');
   assert.equal(w.document.querySelectorAll('.diagram-error').length,0,route);
   assert.equal(w.document.querySelectorAll('.tx-explorer').length,route==='lesson/transformer-world'?1:0,route);
   const diagrams=w.document.querySelectorAll('[data-syntax-valid="true"]').length;
   assert.equal(diagrams,(body.match(/class="mermaid"/g)||[]).length,route);
   if(route.startsWith('lesson/')){
     const index=course.lessons.findIndex(l=>route==='lesson/'+l.id),lesson=course.lessons[index],next=course.lessons[index+1];
     assert.equal(w.document.querySelector('.lesson-top .eyebrow').textContent,'LESSON '+String(index+1).padStart(2,'0')+' / 59');
     assert.equal(w.document.querySelector('.lesson-next .button').getAttribute('href'),next?'#lesson/'+next.id:'#results',route);
     for(const heading of w.document.querySelectorAll('.reading h3')){
       const match=heading.textContent.match(/^Lab (\d+) /);if(match)assert.equal(Number(match[1]),lesson.number,route);
     }
   }
   checks.push({route,diagrams});
 }
 const valid=new Set([...course.lessons.map(l=>'#lesson/'+l.id),...Object.keys(course.references).map(r=>'#ref/'+r)]);
 for(const h of [...course.lessons.map(l=>l.html),...Object.values(course.references).map(r=>r.html)]){
  for(const m of h.matchAll(/href="(#[^"]+)"/g))if(/^#(lesson|ref)\//.test(m[1]))assert.ok(valid.has(m[1]),m[1]);
  for(const m of h.matchAll(/src="([^"#]+)"/g))if(!/^https?:/.test(m[1]))assert.ok(fs.existsSync(path.join(root,'public',m[1])),m[1]);
 }
 w.location.hash='lesson/attention-cache';await new Promise(r=>setTimeout(r,0));
 w.document.querySelector('[data-tab="practice"]').click();await new Promise(r=>setTimeout(r,0));assert.equal(w.document.querySelectorAll('[data-action="run"]').length,1);assert.ok(w.document.querySelector('[data-action="load-example"]'));
 w.location.hash='lesson/judge-audit';await new Promise(r=>setTimeout(r,0));w.document.querySelector('[data-tab="practice"]').click();await new Promise(r=>setTimeout(r,0));assert.equal(w.document.querySelectorAll('[data-action="load-example"]').length,0);assert.equal(w.document.querySelectorAll('[data-action="run"]').length,2);
 w.location.hash='ref/research-landmarks';await new Promise(r=>setTimeout(r,0));assert.equal(w.document.querySelectorAll('.paper-card').length,47);assert.ok(w.document.querySelectorAll('#researchTopic option').length>5);
 const papers=w.document.querySelector('#researchSearch');papers.value='LoRA';papers.dispatchEvent(new w.Event('input',{bubbles:true}));assert.ok(w.document.querySelectorAll('.paper-card:not([hidden])').length>0);assert.ok(w.document.querySelectorAll('.paper-card:not([hidden])').length<47);
 w.location.hash='workspace';await new Promise(r=>setTimeout(r,0));assert.ok(w.document.querySelector('#main').textContent.includes('/test/lab'));assert.ok(w.document.querySelector('#main').textContent.includes('59 lessons'));
 w.location.hash='workbench';await new Promise(r=>setTimeout(r,0));const editor=w.document.querySelector('#codeEditor');editor.value='print("<script>safe</script>")';editor.dispatchEvent(new w.Event('input',{bubbles:true}));assert.equal(w.document.querySelector('#editorHighlight').textContent,editor.value+'\n');assert.equal(w.document.querySelectorAll('#editorHighlight script').length,0);assert.ok(w.document.querySelector('.workbench-guide'));
 w.document.querySelector('#terminalButton').click();assert.ok(w.document.body.classList.contains('terminal-open'));w.document.querySelector('#closeTerminal').click();assert.ok(!w.document.body.classList.contains('terminal-open'));
 w.location.hash='lesson/expert-routing';await new Promise(r=>setTimeout(r,0));assert.ok(w.document.querySelector('.concept-motion'));w.document.querySelector('[data-motion=next]').click();assert.ok(w.document.querySelector('.motion-caption').textContent.includes('scores'));
 w.location.hash='results';await new Promise(r=>setTimeout(r,0));assert.ok(w.document.querySelector('.empty-state'));
 w.eval(`resultReports=[{path:'a',data:{experiment:'test',split:'test',metrics:{accuracy:0.5}}},{path:'b',data:{experiment:'test',split:'test',metrics:{accuracy:0.75}}}];document.querySelector('#main').innerHTML=comparisonControls(resultReports);updateComparison()`);assert.ok(w.document.querySelector('#comparisonTable').textContent.includes('+0.25'));
 w.eval("resultReports[1].data.split='validation';updateComparison()");assert.ok(w.document.querySelector('#comparisonTable').textContent.includes('not directly comparable'));
 const state=JSON.parse(w.localStorage.getItem('super-lab-v1'));assert.deepEqual(state.done,['setup']);assert.equal(state.notes.math,'Earlier evidence remains.');
 const report={routes:checks,diagram_syntax_instances:checks.reduce((n,r)=>n+r.diagrams,0),modern_filter_count:17,practice_controls:true,old_progress_retained:true,lesson_sequence_and_next_links:true,links_and_images:true,limits:'DOM simulation and actual bundled Mermaid parser; no browser layout or visual rendering verification.'};
 if(process.env.SML_DOM_REPORT)fs.writeFileSync(process.env.SML_DOM_REPORT,JSON.stringify(report,null,2));console.log(JSON.stringify({routes:checks.length,diagrams:report.diagram_syntax_instances,result:'passed'}));await new Promise(r=>setTimeout(r,30));w.close();
})().catch(e=>{console.error(e);w.close();process.exit(1)});
