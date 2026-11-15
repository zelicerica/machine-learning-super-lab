const {JSDOM}=require(process.env.SML_JSDOM_PATH||'jsdom');
const fs=require('fs'),assert=require('assert/strict'),path=require('path');
const root=path.resolve(__dirname,'..');
const course=JSON.parse(fs.readFileSync(root+'/public/course.json'));
const html=fs.readFileSync(root+'/public/index.html','utf8').replace(/<script[^>]*src=[^>]+><\/script>/g,'');
const dom=new JSDOM(html,{url:'http://localhost:8765',runScripts:'outside-only',pretendToBeVisual:true});
const w=dom.window;w.TextEncoder=TextEncoder;w.TextDecoder=TextDecoder;w.scrollTo=()=>{};
w.localStorage.setItem('super-lab-v1',JSON.stringify({done:['setup'],notes:{math:'Earlier evidence remains.'}}));
w.fetch=async url=>({ok:true,json:async()=>url==='/course.json'?course:url==='/api/session'?{token:'test',runtime:{ready:true,foundations_ready:true,python:'3.12'},templates:{hello:'print(1)'}}:[]});
const mermaidFile=fs.readFileSync(root+'/public/index.html','utf8').match(/src="([^"]*mermaid[^\"]*)"/)[1];
require('vm').runInContext(fs.readFileSync(path.join(root,'public',mermaidFile),'utf8'),dom.getInternalVMContext());
w.eval('mermaid.run=async({nodes})=>{for(const n of nodes){await mermaid.parse(n.textContent);n.dataset.syntaxValid="true"}}');
require('vm').runInContext(fs.readFileSync(root+'/public/app.js','utf8'),dom.getInternalVMContext());
(async()=>{
 await new Promise(r=>setTimeout(r,100));
 assert.equal(w.document.querySelectorAll('.course-card').length,58);
 w.document.querySelector('[data-track="Modern language engineering"]').click();await new Promise(r=>setTimeout(r,10));assert.equal(w.document.querySelectorAll('.course-card').length,16);
 const checks=[];
 for(const [route,body] of [...course.lessons.map(l=>['lesson/'+l.id,l.html]),...Object.entries(course.references).map(([id,r])=>['ref/'+id,r.html])]){
   w.location.hash=route;await new Promise(r=>setTimeout(r,0));await w.eval('render()');await w.eval('diagramChain');
   assert.equal(w.document.querySelectorAll('.diagram-error').length,0,route);
   const diagrams=w.document.querySelectorAll('[data-syntax-valid="true"]').length;
   assert.equal(diagrams,(body.match(/class="mermaid"/g)||[]).length,route);
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
 const state=JSON.parse(w.localStorage.getItem('super-lab-v1'));assert.deepEqual(state.done,['setup']);assert.equal(state.notes.math,'Earlier evidence remains.');
 const report={routes:checks,diagram_syntax_instances:checks.reduce((n,r)=>n+r.diagrams,0),modern_filter_count:16,practice_controls:true,old_progress_retained:true,links_and_images:true,limits:'DOM simulation and actual bundled Mermaid parser; no browser layout or visual rendering verification.'};
 if(process.env.SML_DOM_REPORT)fs.writeFileSync(process.env.SML_DOM_REPORT,JSON.stringify(report,null,2));console.log(JSON.stringify({routes:checks.length,diagrams:report.diagram_syntax_instances,result:'passed'}));w.close();
})().catch(e=>{console.error(e);w.close();process.exit(1)});
