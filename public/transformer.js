/* Local, dependency-free perspective renderer. All labels and geometry are course data. */
(function (root) {
  'use strict';
  const parts = [
    ['source', 'Source embeddings + positions', -2.05, -3.3, 'encoder', 'Token IDs become learned vectors; sinusoidal position vectors are added in the original model. The encoder receives the complete source sequence.'],
    ['enc-attention', 'Multi-head self-attention', -2.05, -1.8, 'encoder', 'Queries, keys, and values come from the encoder stream. Each source position can attend to all non-padding source positions. The original base model used eight heads with separate learned projections.'],
    ['enc-norm-1', 'Add & Norm', -2.05, -1.1, 'norm', 'Add the attention sublayer input to its output, then apply layer normalization. During training the original model applies dropout to the sublayer output before this addition. This is a residual connection with post-normalization.'],
    ['enc-ffn', 'Feed-forward network', -2.05, 0, 'encoder', 'Two learned linear transformations with a ReLU between them operate at every position. Within a layer the same feed-forward weights are applied to each position; different layers have their own parameters.'],
    ['enc-norm-2', 'Add & Norm → encoder memory', -2.05, 0.7, 'norm', 'The feed-forward sublayer has another residual addition and layer normalization. After all six encoder layers, the final encoder representations supply keys and values to cross-attention in every decoder layer.'],
    ['target', 'Output-prefix embeddings + positions', 2.05, -3.3, 'decoder', 'During training, target tokens are shifted so a position predicts the following token. At inference, the prefix contains tokens generated so far. Embeddings and position information enter the decoder.'],
    ['dec-attention', 'Masked multi-head attention', 2.05, -2.2, 'decoder', 'Queries, keys, and values come from the decoder stream. A causal mask prevents each position from seeing later target tokens. Training can process known target positions in parallel without revealing the future.'],
    ['dec-norm-1', 'Add & Norm', 2.05, -1.5, 'norm', 'A residual addition and layer normalization wrap masked self-attention. The causal restriction belongs to the attention computation, not to the normalization step.'],
    ['cross', 'Encoder–decoder cross-attention', 2.05, -0.5, 'cross', 'Queries come from the decoder. Keys and values come from the final encoder output. This lets the generated output condition on the source sequence. The orange connection denotes encoder memory, not a new self-attention block.'],
    ['dec-norm-2', 'Add & Norm', 2.05, 0.2, 'norm', 'A residual addition and layer normalization wrap cross-attention. The resulting vectors continue through the decoder feed-forward sublayer.'],
    ['dec-ffn', 'Feed-forward network', 2.05, 1.2, 'decoder', 'The decoder also transforms features independently at each position using a feed-forward network. Attention mixes information across positions; this sublayer transforms the resulting features.'],
    ['dec-norm-3', 'Add & Norm → next layer', 2.05, 1.9, 'norm', 'A third residual addition and layer normalization finish the decoder layer. Repeat the layer structure six times with different learned parameters, then send the final decoder output to the vocabulary projection.'],
    ['output', 'Linear projection + softmax', 2.05, 3, 'output', 'Project the final decoder representation into vocabulary-sized logits, then turn them into token probabilities. Choose a token and repeat for autoregressive generation. The scene does not implement training or generate language.']
  ].map(([id, label, x, y, family, description]) => ({id, label, x, y, family, description}));
  const palette = {encoder:'#efbd70', decoder:'#70dcc5', cross:'#9bb7ff', norm:'#8398b3', output:'#d5b4ff'};
  const defaultCamera = () => ({yaw:-0.27, pitch:0.22, zoom:1, panX:0, panY:0});
  function attention(q, causal) {
    const scores = [q / Math.sqrt(2), 0, causal ? -Infinity : q / Math.sqrt(2)];
    const top = Math.max(...scores), exp = scores.map(x => Math.exp(x-top));
    const total = exp.reduce((a,b)=>a+b,0), weights = exp.map(x=>x/total);
    return {scores, weights, output:[2*(weights[0]+weights[2]), 2*(weights[1]+weights[2])]};
  }
  function project(p, camera, width, height) {
    const x = p[0]*Math.cos(camera.yaw) + p[2]*Math.sin(camera.yaw);
    const depth = -p[0]*Math.sin(camera.yaw) + p[2]*Math.cos(camera.yaw);
    const y = p[1]*Math.cos(camera.pitch) - depth*Math.sin(camera.pitch);
    const z = p[1]*Math.sin(camera.pitch) + depth*Math.cos(camera.pitch);
    const scale = Math.min(width/10.7,height/10.5)*camera.zoom*16/(16-z);
    return [width/2+x*scale+camera.panX, height/2-y*scale+camera.panY,z];
  }
  function inside(point, polygon) {
    let hit=false;
    for(let i=0,j=polygon.length-1;i<polygon.length;j=i++) {
      const a=polygon[i],b=polygon[j];
      if((a[1]>point[1])!==(b[1]>point[1]) && point[0]<(b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0])hit=!hit;
    }
    return hit;
  }
  let dispose=null;
  function cleanup() { if(dispose)dispose(); dispose=null; }
  function mount() {
    cleanup();
    if(root.location.hash!=='#lesson/transformer-world' || !document.querySelector('.reading'))return;
    const host=document.createElement('section');host.className='tx-explorer';host.setAttribute('aria-label','Interactive Transformer explorer');
    host.innerHTML=`<div class="tx-heading"><div><span class="tx-kicker">INSIDE THE 2017 TRANSFORMER</span><h2>One architecture. A different way to learn.</h2><p>Turn the model. Follow a connection. See what each component does.</p></div><span class="tx-badge">Interactive 3D · local</span></div>
      <div class="tx-stage"><canvas tabindex="0" aria-label="3D Transformer schematic. Arrow keys rotate, Shift and arrows pan, plus and minus zoom, Home resets. Select components using the menu below." aria-describedby="tx-help"></canvas><p class="tx-canvas-fallback" hidden>Your browser cannot draw the 3D view. The component menu and numerical activity below contain the same teaching content.</p><div class="tx-stage-legend" aria-hidden="true"><span>● Encoder</span><span>● Decoder</span><span>◆ Cross-attention</span></div><div class="tx-stage-caption">One layer expanded per stack · each stack repeats ×6</div></div>
      <div class="tx-toolbar" aria-label="3D view controls"><button type="button" data-tx="in" aria-label="Zoom in">＋</button><button type="button" data-tx="out" aria-label="Zoom out">−</button><button type="button" data-tx="reset">Reset view</button><button type="button" data-tx="front">Front view</button><label><input type="checkbox" id="tx-spread"> Spread components</label><label><input type="checkbox" id="tx-labels" checked> Labels</label><output id="tx-zoom" aria-label="Zoom level">100%</output></div>
      <p class="tx-help" id="tx-help">Drag to rotate · Shift-drag or right-drag to pan · Scroll to zoom · Touch: one finger rotates, two pan and zoom. Keyboard: arrows rotate, Shift + arrows pan, + / − zoom, Home resets.</p>
      <div class="tx-inspect"><div><label for="tx-component">INSPECT A COMPONENT</label><select id="tx-component">${parts.map(p=>`<option value="${p.id}">${p.x<0?'Encoder':p.id==='output'?'Output':'Decoder'} · ${p.label}</option>`).join('')}</select><p class="tx-model-note">Schematic geometry; block sizes are not parameter counts. The arrow between stacks represents the final encoder memory. Residual paths are explained in Add & Norm.</p></div><div class="tx-detail" aria-live="polite"><span id="tx-family"></span><h3 id="tx-part-title"></h3><p id="tx-part-description"></p></div></div>
      <div class="tx-attention"><div><span class="tx-kicker">TRY ONE ATTENTION HEAD</span><h3>Which information gets through?</h3><p>Query at position B: <code>q = [<output id="tx-q-value">1.0</output>, 0]</code>. Keys and values stay fixed. These vectors are invented, not trained.</p><label for="tx-query">Query's first coordinate</label><input type="range" id="tx-query" min="-2" max="2" value="1" step="0.1"><label class="tx-mask"><input type="checkbox" id="tx-mask"> Apply a causal mask (hide future position C)</label></div><div><div class="tx-weights" aria-label="Attention weights">${['A · past','B · current','C · future'].map((s,i)=>`<div><span>${s}</span><div class="tx-meter"><i data-weight="${i}"></i></div><output data-probability="${i}"></output></div>`).join('')}</div><p class="tx-result" role="status" aria-live="polite"></p><p class="tx-model-note">Scores = qKᵀ / √2; softmax turns allowed scores into weights. Output = weighted sum of values. The worked vectors appear in the lesson below.</p></div></div>`;
    document.querySelector('.lesson-layout').before(host);
    const find=s=>host.querySelector(s), canvas=find('canvas'), stage=find('.tx-stage');
    let context=null;try{context=canvas.getContext('2d');}catch(_){};
    if(!context){canvas.hidden=true;find('.tx-canvas-fallback').hidden=false;host.querySelectorAll('.tx-toolbar button,.tx-toolbar input').forEach(e=>e.disabled=true);}
    let camera=defaultCamera(),width=800,height=560,selected='cross',spread=false,labels=true,frame=0,dead=false,hitFaces=[];
    const controller=new AbortController(), signal=controller.signal;
    function on(target,event,callback,options={}) {target.addEventListener(event,callback,{...options,signal});}
    function schedule(){if(!dead&&!frame&&context)frame=root.requestAnimationFrame(()=>{frame=0;draw();});}
    function choose(id){const part=parts.find(p=>p.id===id);if(!part)return;selected=id;find('#tx-component').value=id;find('#tx-family').textContent=part.x<0?'ENCODER STREAM':part.id==='output'?'OUTPUT DISTRIBUTION':'DECODER STREAM';find('#tx-part-title').textContent=part.label;find('#tx-part-description').textContent=part.description;schedule();}
    function zoom(factor){camera.zoom=Math.max(.55,Math.min(2.5,camera.zoom*factor));find('#tx-zoom').textContent=Math.round(camera.zoom*100)+'%';schedule();}
    const pos=p=>[p.x,p.y*(spread?1.17:1),0];
    const projectPoint=p=>project(p,camera,width,height);
    function line(points,color,dashed=false){context.beginPath();points.map(projectPoint).forEach((p,i)=>i?context.lineTo(p[0],p[1]):context.moveTo(p[0],p[1]));context.strokeStyle=color;context.lineWidth=1.6;context.setLineDash(dashed?[5,5]:[]);context.stroke();context.setLineDash([]);}
    function arrow(from,to,color){const a=projectPoint(from),b=projectPoint(to),angle=Math.atan2(b[1]-a[1],b[0]-a[0]);line([from,to],color);context.beginPath();context.moveTo(b[0],b[1]);context.lineTo(b[0]-8*Math.cos(angle-.45),b[1]-8*Math.sin(angle-.45));context.lineTo(b[0]-8*Math.cos(angle+.45),b[1]-8*Math.sin(angle+.45));context.closePath();context.fillStyle=color;context.fill();}
    function draw(){
      context.clearRect(0,0,width,height);hitFaces=[];
      for(let x=-5;x<=5;x++)line([[x,-4.2,-3],[x,-4.2,3]],'#1a3048');
      for(let z=-3;z<=3;z++)line([[-5,-4.2,z],[5,-4.2,z]],'#1a3048');
      for(const ids of [['source','enc-attention','enc-norm-1','enc-ffn','enc-norm-2'],['target','dec-attention','dec-norm-1','cross','dec-norm-2','dec-ffn','dec-norm-3','output']]){
        for(let i=1;i<ids.length;i++){const a=pos(parts.find(p=>p.id===ids[i-1])),b=pos(parts.find(p=>p.id===ids[i]));a[1]+=.22;b[1]-=.24;arrow(a,b,ids[0]==='source'?'#ad8d60':'#4cae9a');}
      }
      const memory=pos(parts.find(p=>p.id==='enc-norm-2')), cross=pos(parts.find(p=>p.id==='cross'));
      memory[0]+=1.4;cross[0]-=1.4;arrow(memory,cross,'#efbd70');
      const faces=[];
      for(const part of parts){
        const [x,y,z]=pos(part),dy=part.family==='norm'?.11:.22,dx=1.38,dz=.55;
        const vertices=[[-dx,-dy,-dz],[dx,-dy,-dz],[dx,dy,-dz],[-dx,dy,-dz],[-dx,-dy,dz],[dx,-dy,dz],[dx,dy,dz],[-dx,dy,dz]].map(v=>projectPoint([v[0]+x,v[1]+y,v[2]+z]));
        [[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2],[3,2,6,7],[0,1,5,4]].forEach((indices,index)=>{const points=indices.map(i=>vertices[i]);faces.push({part,points,depth:points.reduce((n,p)=>n+p[2],0)/4,index});});
      }
      faces.sort((a,b)=>a.depth-b.depth);
      for(const face of faces){const {points,part,index}=face;context.beginPath();points.forEach((p,i)=>i?context.lineTo(p[0],p[1]):context.moveTo(p[0],p[1]));context.closePath();context.fillStyle=palette[part.family];context.globalAlpha=[.29,.60,.39,.45,.75,.29][index];context.fill();context.globalAlpha=1;context.strokeStyle=part.id===selected?'#ffffff':palette[part.family];context.lineWidth=part.id===selected?1.8:.65;context.stroke();hitFaces.push(face);}
      if(labels){
        for(const part of [...parts].sort((a,b)=>projectPoint(pos(a))[2]-projectPoint(pos(b))[2])){
          const p=projectPoint([...pos(part).slice(0,2),.65]);const font=Math.max(9,Math.min(12,width/73));context.font=`600 ${font}px system-ui, sans-serif`;context.textAlign='center';context.textBaseline='middle';
          const shortLabels={'source':'Source + positions','enc-attention':'Self-attention','enc-norm-2':'Add & Norm → memory','enc-ffn':'Feed-forward','target':'Prefix + positions','dec-attention':'Masked attention','cross':'Cross-attention','dec-ffn':'Feed-forward','dec-norm-3':'Add & Norm','output':'Linear + softmax'};
          const label=width<650?(shortLabels[part.id]||part.label):part.label;
          const textWidth=context.measureText(label).width;context.fillStyle='#081423eb';context.fillRect(p[0]-textWidth/2-5,p[1]-font/2-4,textWidth+10,font+8);context.fillStyle=part.id===selected?'#ffffff':'#e0ebf4';context.fillText(label,p[0],p[1]);
        }
      }
    }
    function resize(){if(dead)return;const rect=stage.getBoundingClientRect();width=Math.max(1,rect.width||800);height=Math.max(1,rect.height||560);const ratio=Math.min(root.devicePixelRatio||1,2);canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);if(context)context.setTransform(ratio,0,0,ratio,0,0);schedule();}
    function updateAttention(){const q=Number(find('#tx-query').value),masked=find('#tx-mask').checked,result=attention(q,masked);find('#tx-q-value').textContent=q.toFixed(1);result.weights.forEach((p,i)=>{find(`[data-weight="${i}"]`).style.width=(p*100)+'%';find(`[data-probability="${i}"]`).textContent=(p*100).toFixed(1)+'%';});find('.tx-result').textContent=`Weighted output = [${result.output.map(x=>x.toFixed(4)).join(', ')}]. ${masked?'Future position C is blocked.':'All three positions are visible.'}`;}
    on(find('#tx-component'),'change',e=>choose(e.target.value));
    on(find('#tx-spread'),'change',e=>{spread=e.target.checked;schedule();});
    on(find('#tx-labels'),'change',e=>{labels=e.target.checked;schedule();});
    for(const input of [find('#tx-query'),find('#tx-mask')])on(input,'input',updateAttention);
    on(find('.tx-toolbar'),'click',e=>{const action=e.target.closest('button')?.dataset.tx;if(action==='in')zoom(1.15);if(action==='out')zoom(1/1.15);if(action==='reset'){camera=defaultCamera();zoom(1);}if(action==='front'){camera=defaultCamera();camera.yaw=0;camera.pitch=0;zoom(1);}});
    on(canvas,'keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','_','Home'].includes(e.key))return;e.preventDefault();if(e.key==='Home'){camera=defaultCamera();zoom(1);}else if(['+','='].includes(e.key))zoom(1.1);else if(['-','_'].includes(e.key))zoom(1/1.1);else{const horizontal=e.key==='ArrowLeft'||e.key==='ArrowRight',amount=['ArrowLeft','ArrowUp'].includes(e.key)?-1:1;if(e.shiftKey)camera[horizontal?'panX':'panY']+=amount*18;else camera[horizontal?'yaw':'pitch']+=amount*.08;camera.pitch=Math.max(-1.15,Math.min(1.15,camera.pitch));schedule();}});
    on(canvas,'wheel',e=>{e.preventDefault();zoom(Math.exp(-Math.max(-150,Math.min(150,e.deltaY))*.002));},{passive:false});
    on(canvas,'contextmenu',e=>e.preventDefault());
    const pointers=new Map();let gesture=null;
    function pair(){const p=[...pointers.values()];return {x:(p[0].x+p[1].x)/2,y:(p[0].y+p[1].y)/2,d:Math.hypot(p[0].x-p[1].x,p[0].y-p[1].y)};}
    on(canvas,'pointerdown',e=>{if(e.button!==0&&e.button!==2)return;canvas.focus({preventScroll:true});pointers.set(e.pointerId,{x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,moved:false,pan:e.shiftKey||e.button===2});canvas.setPointerCapture?.(e.pointerId);if(pointers.size>=2){pointers.forEach(p=>p.moved=true);gesture=pair();}});
    on(canvas,'pointermove',e=>{const p=pointers.get(e.pointerId);if(!p)return;const dx=e.clientX-p.x,dy=e.clientY-p.y;p.x=e.clientX;p.y=e.clientY;if(Math.hypot(p.x-p.startX,p.y-p.startY)>4)p.moved=true;if(pointers.size>=2){const next=pair();if(gesture){camera.panX+=next.x-gesture.x;camera.panY+=next.y-gesture.y;if(gesture.d>0)zoom(next.d/gesture.d);}gesture=next;}else if(p.pan||e.shiftKey){camera.panX+=dx;camera.panY+=dy;}else{camera.yaw+=dx*.007;camera.pitch=Math.max(-1.15,Math.min(1.15,camera.pitch+dy*.007));}schedule();});
    function release(e){const p=pointers.get(e.pointerId);if(p&&e.type==='pointerup'&&!p.moved&&!p.pan){const rect=canvas.getBoundingClientRect(),point=[e.clientX-rect.left,e.clientY-rect.top];const face=[...hitFaces].reverse().find(f=>inside(point,f.points));if(face)choose(face.part.id);}pointers.delete(e.pointerId);gesture=null;if(canvas.hasPointerCapture?.(e.pointerId))canvas.releasePointerCapture(e.pointerId);}
    for(const event of ['pointerup','pointercancel','lostpointercapture'])on(canvas,event,release);
    const observer=root.ResizeObserver?new ResizeObserver(resize):null;observer?.observe(stage);if(!observer)on(root,'resize',resize);
    dispose=()=>{dead=true;controller.abort();observer?.disconnect();if(frame)root.cancelAnimationFrame(frame);pointers.clear();host.remove();};
    choose(selected);updateAttention();resize();
  }
  root.TransformerExplorer={mount,cleanup};
  if(typeof module!=='undefined'&&module.exports)module.exports={attention,project,inside,parts};
})(typeof window!=='undefined'?window:globalThis);
