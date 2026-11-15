// Numerical reference cases and interaction/lifecycle regressions without a browser or GPU.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {JSDOM}=require(process.env.SML_JSDOM_PATH||'jsdom');
const source=path.resolve(__dirname,'../public/transformer.js');
const {attention,project,inside,parts,model,sceneFor}=require(source);
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);
const unmasked=attention(1,false),masked=attention(1,true);
near(unmasked.weights[0],0.4011120926797859);near(unmasked.output[0],1.6044483707191437);near(unmasked.output[1],1.1977758146404283);
near(masked.weights[0],0.6697615493266569);assert.equal(masked.weights[2],0);near(masked.output[0],1.3395230986533138);near(masked.output[1],0.6604769013466862);
for(const q of [-2,-1,0,1,2])for(const causal of [false,true]){const r=attention(q,causal);near(r.weights.reduce((a,b)=>a+b),1);assert.ok(r.weights.every(p=>p>=0&&p<=1));assert.ok(r.output.every(v=>v>=0&&v<=2));}
attention(0,false).weights.forEach(p=>near(p,1/3));attention(0,false).output.forEach(v=>near(v,4/3));
assert.ok(inside([1,1],[[0,0],[2,0],[2,2],[0,2]]));assert.ok(!inside([3,1],[[0,0],[2,0],[2,2],[0,2]]));
const camera={yaw:0,pitch:0,zoom:1,panX:0,panY:0};assert.deepEqual(project([0,0,0],camera,800,600),[400,300,0]);
assert.ok(project([1,0,2],camera,800,600)[0]>project([1,0,-2],camera,800,600)[0]);
assert.equal(new Set(parts.map(p=>p.id)).size,63);
assert.equal(model.layers.length,12);assert.equal(sceneFor().nodes.length,63);
// Repeated complete layers recede in depth; only their internal operations rise.
const overview=sceneFor(),separated=sceneFor('',true);
for(const stack of ['encoder','decoder']){
 const layerZ=[];
 for(let layer=1;layer<=6;layer++){
  const nodes=overview.nodes.filter(p=>p.stack===stack&&p.layer===layer);
  assert.equal(new Set(nodes.map(p=>p.z)).size,1);
  layerZ.push(nodes[0].z);
  for(const p of nodes){
   const first=overview.nodes.find(n=>n.layer===1&&n.templateId===p.templateId);
   near(p.y,first.y);
   if(p.order)assert.ok(p.y>nodes[p.order-1].y);
   near(separated.nodes.find(n=>n.id===p.id).y,p.y);
  }
 }
 assert.equal(new Set(layerZ).size,6);assert.ok(layerZ.every((z,i)=>!i||z<layerZ[i-1]));
 const expanded=separated.nodes.filter(p=>p.stack===stack&&p.layer).map(p=>p.z);
 assert.ok(Math.max(...expanded)-Math.min(...expanded)>layerZ[0]-layerZ[5]);
 near(overview.nodes.find(p=>p.id===(stack==='encoder'?'source':'target')).z,layerZ[0]);
 if(stack==='decoder')near(overview.nodes.find(p=>p.id==='output').z,layerZ[5]);
}
assert.equal(model.edges.filter(e=>e.kind==='residual').length,30);
const memoryEdges=model.edges.filter(e=>e.kind==='memory');assert.equal(memoryEdges.length,6);
assert.ok(memoryEdges.every(e=>e.from==='encoder-6-enc-norm-2'));
assert.deepEqual(memoryEdges.map(e=>e.to),Array.from({length:6},(_,i)=>`decoder-${i+1}-cross`));
for(const group of model.layers){
 assert.equal(group.ids.length,group.stack==='encoder'?4:6);
 const scene=sceneFor(group.key),ids=new Set(scene.nodes.map(p=>p.id));
 assert.equal(ids.size,scene.nodes.length);assert.ok(scene.edges.every(e=>ids.has(e.from)&&ids.has(e.to)));
 assert.equal(scene.nodes.filter(p=>!p.context).length,group.ids.length);
 assert.equal(scene.edges.filter(e=>e.kind==='residual').length,group.stack==='encoder'?2:3);
 if(group.stack==='decoder'){assert.ok(scene.nodes.find(p=>p.id==='encoder-6-enc-norm-2').context);assert.equal(scene.edges.filter(e=>e.kind==='memory').length,1);}
}
const dom=new JSDOM('<main><div class="lesson-layout"><article class="reading"></article></div></main>',{url:'http://localhost/#lesson/transformer-world',runScripts:'outside-only'}),w=dom.window;
let available=true,width=950,frames=new Map(),frameId=0,draws=[],observerDisconnected=0;
const context={clearRect(){draws=[]},moveTo(...v){draws.push(['move',...v])},lineTo(...v){draws.push(['line',...v])},fillText(...v){draws.push(['text',...v])},measureText(s){return{width:s.length*6}},setTransform(){},beginPath(){},stroke(){},fill(){},closePath(){},setLineDash(){},fillRect(){}};
w.HTMLCanvasElement.prototype.getContext=()=>available?context:null;
w.HTMLElement.prototype.getBoundingClientRect=()=>({width,height:570,left:0,top:0});
w.requestAnimationFrame=fn=>{frames.set(++frameId,fn);return frameId};w.cancelAnimationFrame=id=>frames.delete(id);
w.ResizeObserver=class{constructor(fn){this.fn=fn}observe(){this.fn()}disconnect(){observerDisconnected++}};
const flush=()=>{for(const [id,fn] of [...frames]){frames.delete(id);fn()}};
const pointer=(canvas,type,x,y,id=1,extras={})=>{const e=new w.Event(type,{bubbles:true});Object.assign(e,{clientX:x,clientY:y,pointerId:id,button:0,...extras});canvas.dispatchEvent(e);flush()};
w.eval(fs.readFileSync(source,'utf8'));w.TransformerExplorer.mount();flush();
const $=s=>w.document.querySelector(s),canvas=$('canvas'),initial=JSON.stringify(draws);
assert.equal($('.tx-explorer').querySelectorAll('#tx-component option').length,63);assert.ok($('#tx-part-description').textContent.includes('sinusoidal'));
assert.equal(w.document.querySelectorAll('[data-layer]').length,12);
assert.ok(draws.some(d=>d[0]==='text'));assert.equal($('.tx-canvas-fallback').hidden,true);
function checkOverviewLabels(){
 const texts=draws.filter(d=>d[0]==='text').map(d=>d[1]);
 for(const label of ['ENCODER · 6 layers','DECODER · 6 layers','Self-attention','Masked attention','Cross-attention','Feed-forward','Add & Norm','Source + positions','Prefix + positions','Linear + softmax',...['E','D'].flatMap(s=>Array.from({length:6},(_,i)=>s+(i+1)))])assert.ok(texts.includes(label),`Missing default label: ${label}`);
 assert.equal(texts.filter(t=>t==='Add & Norm').length,5);
 assert.equal(texts.filter(t=>t==='Feed-forward').length,2);
 assert.ok(draws.filter(d=>d[0]==='text').every(d=>d[2]>0&&d[2]<width&&d[3]>0&&d[3]<570),'Default labels must fit inside the canvas');
}
assert.equal($('#tx-labels').checked,true);checkOverviewLabels();
pointer(canvas,'pointerdown',20,20);pointer(canvas,'pointermove',120,40);pointer(canvas,'pointerup',120,40);assert.notEqual(JSON.stringify(draws),initial);
$('[data-tx=reset]').click();flush();assert.equal(JSON.stringify(draws),initial);
pointer(canvas,'pointerdown',20,20,1,{shiftKey:true});pointer(canvas,'pointermove',70,20,1,{shiftKey:true});pointer(canvas,'pointerup',70,20);assert.notEqual(JSON.stringify(draws),initial);
$('[data-tx=reset]').click();flush();canvas.dispatchEvent(new w.KeyboardEvent('keydown',{key:'ArrowLeft',bubbles:true}));flush();assert.notEqual(JSON.stringify(draws),initial);
canvas.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Home',bubbles:true}));flush();assert.equal(JSON.stringify(draws),initial);
for(let i=0;i<30;i++)$('[data-tx=in]').click();flush();assert.equal($('#tx-zoom').textContent,'300%');
for(let i=0;i<60;i++)$('[data-tx=out]').click();flush();assert.equal($('#tx-zoom').textContent,'55%');
$('[data-tx=reset]').click();flush();
pointer(canvas,'pointerdown',50,50,1);pointer(canvas,'pointerdown',150,50,2);pointer(canvas,'pointermove',250,50,2);assert.equal($('#tx-zoom').textContent,'200%');pointer(canvas,'pointercancel',50,50,1);pointer(canvas,'pointerup',250,50,2);
$('[data-tx=front]').click();flush();const front=JSON.stringify(draws);$('#tx-spread').click();flush();assert.notEqual(JSON.stringify(draws),front);
$('#tx-labels').click();flush();assert.ok(!draws.some(d=>d[0]==='text'));$('#tx-labels').click();flush();
$('#tx-component').value='source';$('#tx-component').dispatchEvent(new w.Event('change'));assert.match($('#tx-part-description').textContent,/sinusoidal/);
$('#tx-mask').click();assert.equal($('[data-probability="2"]').textContent,'0.0%');assert.match($('.tx-result').textContent,/1.3395, 0.6605/);
$('#tx-mask').click();$('#tx-query').value='0';$('#tx-query').dispatchEvent(new w.Event('input'));assert.match($('.tx-result').textContent,/1.3333, 1.3333/);
// Every layer opens with its own identity, components, and correct external memory source.
for(const group of model.layers){
 $(`[data-layer="${group.key}"]`).click();flush();
 assert.equal($(`[data-layer="${group.key}"]`).getAttribute('aria-pressed'),'true');
 assert.ok($('.tx-stage-caption').textContent.toLowerCase().includes(group.stack+' layer '+group.number));
 assert.equal($('#tx-component').value,group.ids[0]);
 assert.ok(draws.some(d=>d[0]==='text'&&d[1]===(group.stack==='decoder'?'Final encoder memory':'Self-attention')));
}
$('[data-tx=all]').click();flush();assert.equal($('[data-tx=all]').getAttribute('aria-pressed'),'true');
$('#tx-component').value='encoder-1-enc-norm-2';$('#tx-component').dispatchEvent(new w.Event('change'));assert.match($('#tx-part-description').textContent,/not this intermediate/);
$('[data-tx=detail]').click();flush();assert.equal($('[data-layer="encoder:1"]').getAttribute('aria-pressed'),'true');
$('[data-tx=next]').click();flush();assert.equal($('[data-layer="encoder:2"]').getAttribute('aria-pressed'),'true');
$('#tx-component').value='decoder-3-cross';$('#tx-component').dispatchEvent(new w.Event('change'));assert.equal($('#tx-heads').hidden,false);assert.match($('#tx-part-description').textContent,/final encoder output/);
$('[data-tx=all]').click();$('#tx-component').value='source';$('#tx-component').dispatchEvent(new w.Event('change'));
for(let i=1;i<63;i++)$('[data-tx=next]').click();assert.equal($('#tx-component').value,'output');assert.equal($('[data-tx=next]').disabled,true);
const withConnections=JSON.stringify(draws);$('#tx-connections').click();flush();assert.notEqual(JSON.stringify(draws),withConnections);
w.TransformerExplorer.mount();flush();assert.equal(w.document.querySelectorAll('.tx-explorer').length,1);assert.equal(observerDisconnected,1);
w.TransformerExplorer.cleanup();assert.equal($('.tx-explorer'),null);assert.equal(frames.size,0);assert.equal(observerDisconnected,2);
available=false;w.TransformerExplorer.mount();assert.equal($('.tx-canvas-fallback').hidden,false);assert.equal($('[data-tx=in]').disabled,true);$('#tx-mask').click();assert.match($('.tx-result').textContent,/1.3395, 0.6605/);w.TransformerExplorer.cleanup();
available=true;width=375;w.TransformerExplorer.mount();flush();checkOverviewLabels();w.TransformerExplorer.cleanup();
w.location.hash='lesson/setup';w.TransformerExplorer.mount();assert.equal($('.tx-explorer'),null);dom.window.close();
console.log('Transformer: six depth planes per stack, upward operations within layers, default component/stack/layer labels at desktop and mobile widths, full 12-layer topology, final-encoder cross-attention, residual paths, layer drilldown, tour, numerical cases, projection, orbit/pan/zoom, touch gestures, keyboard, controls, fallback, and cleanup passed (DOM/canvas simulation).');
