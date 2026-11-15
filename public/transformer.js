/* Local, dependency-free 3D schematic of all original Transformer layers. */
(function(root){
  'use strict';
  const templates = [
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
  const palette={encoder:'#efbd70',decoder:'#70dcc5',cross:'#9bb7ff',ffn:'#bf9beb',norm:'#8296b1',output:'#f2b1d1'};
  const pattern={encoder:['enc-attention','enc-norm-1','enc-ffn','enc-norm-2'],decoder:['dec-attention','dec-norm-1','cross','dec-norm-2','dec-ffn','dec-norm-3']};
  const layerKey=p=>p.layer?`${p.stack}:${p.layer}`:'';
  const layerName=key=>{const [stack,n]=key.split(':');return `${stack==='encoder'?'Encoder':'Decoder'} layer ${n}`};
  const shortNames={'source':'Source + positions','target':'Prefix + positions','enc-attention':'Self-attention','enc-norm-1':'Add & Norm','enc-norm-2':'Add & Norm','enc-ffn':'Feed-forward','dec-attention':'Masked attention','dec-norm-1':'Add & Norm','dec-norm-2':'Add & Norm','dec-norm-3':'Add & Norm','cross':'Cross-attention','dec-ffn':'Feed-forward','output':'Linear + softmax'};
  function buildModel(){
    const parts=[],edges=[],layers=[];
    const add=(templateId,stack,layer=0,order=0)=>{
      const t=templates.find(p=>p.id===templateId),id=layer?`${stack}-${layer}-${templateId}`:templateId;
      let label=t.label;
      if(templateId==='enc-norm-2')label=layer===6?'Add & Norm → encoder memory':'Add & Norm → next encoder layer';
      if(templateId==='dec-norm-3')label=layer===6?'Add & Norm → vocabulary head':'Add & Norm → next decoder layer';
      const p={...t,id,templateId,label,stack,layer,order,family:templateId.includes('ffn')?'ffn':t.family};parts.push(p);return p;
    };
    const edge=(from,to,kind='main')=>edges.push({from,to,kind});
    for(const stack of ['encoder','decoder']){
      let previous=add(stack==='encoder'?'source':'target',stack).id;
      for(let layer=1;layer<=6;layer++){
        const nodes=pattern[stack].map((id,i)=>add(id,stack,layer,i));
        layers.push({key:`${stack}:${layer}`,stack,number:layer,ids:nodes.map(p=>p.id)});
        for(let i=0;i<nodes.length;i++){
          edge(previous,nodes[i].id);
          if(i%2===0)edge(previous,nodes[i+1].id,'residual');
          previous=nodes[i].id;
        }
        if(stack==='decoder')edge('encoder-6-enc-norm-2',nodes[2].id,'memory');
      }
      if(stack==='decoder')edge(previous,add('output','output').id);
    }
    return {parts,edges,layers};
  }
  const model=buildModel(),parts=model.parts;
  function sceneFor(focus='',spread=false){
    let nodes,edges;
    if(!focus){
      nodes=parts.map(p=>{
        // A whole layer occupies one depth plane. Its internal operations run upward.
        const depthGap=spread?2.65:1.75,front=2.5*depthGap;
        const y=p.id==='output'?4.15:p.layer?(p.order-(pattern[p.stack].length-1)/2)*(p.stack==='encoder'?1.4:1):-4;
        const z=p.layer?front-(p.layer-1)*depthGap:p.id==='output'?-front:front;
        return {...p,x:p.stack==='encoder'?-3.7:3.7,y,z,dx:1.35,dy:p.family==='norm'?.08:.14,dz:.22};
      });
      edges=model.edges.filter(e=>e.kind!=='residual');
    }else{
      const group=model.layers.find(l=>l.key===focus);
      if(!group)return sceneFor('',spread);
      const gap=spread?1.22:.95;
      nodes=group.ids.map(id=>parts.find(p=>p.id===id)).map(p=>({...p,x:group.stack==='decoder'?.7:0,y:(p.order-(group.ids.length-1)/2)*gap,z:0,dx:1.55,dy:p.family==='norm'?.13:.22,dz:.85}));
      const bottom=nodes[0],top=nodes[nodes.length-1];
      const originalInput=model.edges.find(e=>e.kind==='main'&&e.to===bottom.id).from;
      const originalOutput=model.edges.find(e=>e.kind==='main'&&e.from===top.id)?.to||'context-encoder-memory';
      const outside=(id,label,x,y,family)=>({id,label,x,y,z:0,dx:1.25,dy:.17,dz:.5,family,context:true,stack:group.stack,layer:0,templateId:'context'});
      nodes.push(outside(originalInput,group.number===1?'Embeddings + positions':`From ${group.stack} layer ${group.number-1}`,bottom.x,bottom.y-1.15,group.stack));
      nodes.push(outside(originalOutput,group.number===6?(group.stack==='encoder'?'Final encoder memory':'To linear + softmax'):`To ${group.stack} layer ${group.number+1}`,top.x,top.y+1.15,group.stack));
      if(group.stack==='decoder')nodes.push(outside('encoder-6-enc-norm-2','Final encoder memory',-3.3,-.5,'encoder'));
      const ids=new Set(nodes.map(p=>p.id));edges=model.edges.filter(e=>ids.has(e.from)&&ids.has(e.to));
      if(originalOutput==='context-encoder-memory')edges.push({from:top.id,to:originalOutput,kind:'main'});
    }
    return {nodes,edges};
  }
  const defaultCamera=()=>({yaw:-.60,pitch:.31,zoom:1,panX:0,panY:0});
  function attention(q,causal){const scores=[q/Math.sqrt(2),0,causal?-Infinity:q/Math.sqrt(2)],top=Math.max(...scores),exp=scores.map(x=>Math.exp(x-top)),sum=exp.reduce((a,b)=>a+b,0),weights=exp.map(x=>x/sum);return{scores,weights,output:[2*(weights[0]+weights[2]),2*(weights[1]+weights[2])]};}
  function project(p,camera,width,height){
    const x=p[0]*Math.cos(camera.yaw)+p[2]*Math.sin(camera.yaw),depth=-p[0]*Math.sin(camera.yaw)+p[2]*Math.cos(camera.yaw);
    const y=p[1]*Math.cos(camera.pitch)-depth*Math.sin(camera.pitch),z=p[1]*Math.sin(camera.pitch)+depth*Math.cos(camera.pitch);
    const scale=Math.min(width/17,height/13)*camera.zoom*22/(22-z);
    return[width/2+x*scale+camera.panX,height/2-y*scale+camera.panY,z];
  }
  function inside(point,polygon){let hit=false;for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){const a=polygon[i],b=polygon[j];if((a[1]>point[1])!==(b[1]>point[1])&&point[0]<(b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0])hit=!hit;}return hit;}
  let dispose=null;
  function cleanup(){if(dispose)dispose();dispose=null;}
  function mount(){
    cleanup();if(root.location.hash!=='#lesson/transformer-world'||!document.querySelector('.reading'))return;
    const host=document.createElement('section');host.className='tx-explorer';host.setAttribute('aria-label','Interactive Transformer explorer');
    const componentOptions=()=>['source','target','output'].map(id=>parts.find(p=>p.id===id)).map(p=>`<option value="${p.id}">${p.label}</option>`).join('')+model.layers.map(l=>`<optgroup label="${layerName(l.key)}">${l.ids.map(id=>{const p=parts.find(p=>p.id===id);return `<option value="${id}">${p.label}</option>`}).join('')}</optgroup>`).join('');
    host.innerHTML=`<div class="tx-heading"><div><span class="tx-kicker">INSIDE THE 2017 TRANSFORMER</span><h2>The whole model. Layer by layer.</h2><p>Six encoder blocks and six decoder blocks extend into depth. Operations inside each block run bottom to top.</p></div><span class="tx-badge">12 layers · 63 components</span></div>
      <div class="tx-layer-nav"><button type="button" class="tx-overview" data-tx="all" aria-pressed="true">Whole model</button><div class="tx-layer-rows">${['encoder','decoder'].map(stack=>`<div><span>${stack==='encoder'?'Encoder':'Decoder'}</span>${model.layers.filter(l=>l.stack===stack).map(l=>`<button type="button" data-layer="${l.key}" aria-label="Inspect ${layerName(l.key).toLowerCase()}" aria-pressed="false">${l.number}</button>`).join('')}</div>`).join('')}</div><p>Select a layer to unfold its attention, feed-forward, and residual paths.</p></div>
      <div class="tx-stage"><canvas tabindex="0" aria-label="3D Transformer with six encoder and six decoder layers. Arrow keys rotate, Shift and arrows pan, plus and minus zoom, Home resets. Layer buttons and the component menu provide keyboard selection." aria-describedby="tx-help"></canvas><p class="tx-canvas-fallback" hidden>Your browser cannot draw the 3D view. Use the layer buttons, component menu, and numerical activity below.</p><div class="tx-stage-legend" aria-hidden="true"><span>● Encoder attention</span><span>● Decoder attention</span><span>● Cross-attention</span><span>● Feed-forward</span><span>● Add & Norm</span></div><div class="tx-stage-caption" role="status">Layer 1 at the front → layer 6 at the back · 6 blocks per stack</div></div>
      <div class="tx-toolbar" aria-label="3D view controls"><button type="button" data-tx="in" aria-label="Zoom in">＋</button><button type="button" data-tx="out" aria-label="Zoom out">−</button><button type="button" data-tx="reset">Reset view</button><button type="button" data-tx="front">Front view</button><label><input type="checkbox" id="tx-spread"> Separate depth layers</label><label><input type="checkbox" id="tx-labels" checked> Labels</label><label><input type="checkbox" id="tx-connections" checked> Connections</label><output id="tx-zoom" aria-label="Zoom level">100%</output></div>
      <p class="tx-help" id="tx-help">Drag to rotate · Shift-drag or right-drag to pan · Scroll to zoom · Touch: one finger rotates, two pan and zoom. Keyboard: arrows rotate, Shift + arrows pan, + / − zoom, Home resets. A head-on view overlaps the depth layers; rotate to see their separation.</p>
      <div class="tx-tour"><div><span class="tx-kicker">FOLLOW THE CALCULATION</span><p id="tx-tour-status" aria-live="polite"></p></div><button type="button" data-tx="previous" aria-label="Previous component">← Previous</button><button type="button" data-tx="next" aria-label="Next component">Next →</button></div>
      <div class="tx-inspect"><div><label for="tx-component">INSPECT A COMPONENT</label><select id="tx-component">${componentOptions()}</select><button type="button" data-tx="detail" class="tx-detail-button">Open selected layer</button><p class="tx-model-note">Schematic geometry, not real activations or parameter sizes. Orange links carry final encoder memory to every decoder layer. Dashed paths in detail views are residual additions.</p></div><div class="tx-detail" aria-live="polite"><span id="tx-family"></span><h3 id="tx-part-title"></h3><p id="tx-part-description"></p><div id="tx-heads" hidden><span>8 PARALLEL ATTENTION HEADS</span><div>${Array.from({length:8},(_,i)=>`<i>${i+1}</i>`).join('')}</div><small>Each head has its own query, key, and value projections. Concatenate their outputs, then project.</small></div></div></div>
      <div class="tx-attention"><div><span class="tx-kicker">TRY ONE ATTENTION HEAD</span><h3>Which information gets through?</h3><p>Query at position B: <code>q = [<output id="tx-q-value">1.0</output>, 0]</code>. Keys and values stay fixed. These vectors are invented, not trained.</p><label for="tx-query">Query's first coordinate</label><input type="range" id="tx-query" min="-2" max="2" value="1" step="0.1"><label class="tx-mask"><input type="checkbox" id="tx-mask"> Apply a causal mask (hide future position C)</label></div><div><div class="tx-weights" aria-label="Attention weights">${['A · past','B · current','C · future'].map((s,i)=>`<div><span>${s}</span><div class="tx-meter"><i data-weight="${i}"></i></div><output data-probability="${i}"></output></div>`).join('')}</div><p class="tx-result" role="status" aria-live="polite"></p><p class="tx-model-note">Scores = qKᵀ / √2; softmax turns allowed scores into weights. Output = weighted sum of values. The worked vectors appear in the lesson below.</p></div></div>`;
    document.querySelector('.lesson-layout').before(host);
    const find=s=>host.querySelector(s),canvas=find('canvas'),stage=find('.tx-stage');
    let context=null;try{context=canvas.getContext('2d');}catch(_){}
    if(!context){canvas.hidden=true;find('.tx-canvas-fallback').hidden=false;host.querySelectorAll('.tx-toolbar button,.tx-toolbar input').forEach(e=>e.disabled=true);}
    let camera=defaultCamera(),width=800,height=690,selected='source',focus='',spread=false,labels=true,connections=true,frame=0,dead=false,hitFaces=[],scene=sceneFor();
    const controller=new AbortController(),signal=controller.signal;
    const on=(target,event,callback,options={})=>target.addEventListener(event,callback,{...options,signal});
    function schedule(){if(!dead&&!frame&&context)frame=root.requestAnimationFrame(()=>{frame=0;draw();});}
    function resetCamera(){camera=defaultCamera();zoom(1);}
    function setFocus(key){focus=key;scene=sceneFor(focus,spread);host.querySelectorAll('[data-layer]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.layer===focus)));find('[data-tx=all]').setAttribute('aria-pressed',String(!focus));find('.tx-stage-caption').textContent=focus?`${layerName(focus)} · solid: main path · dashed: residual · orange: encoder memory`:'Layer 1 at the front → layer 6 at the back · 6 blocks per stack';resetCamera();}
    function choose(id){
      const p=parts.find(p=>p.id===id);if(!p)return;selected=id;
      if(focus&&layerKey(p)!==focus)setFocus(layerKey(p));
      find('#tx-component').value=id;find('#tx-family').textContent=p.layer?layerName(layerKey(p)).toUpperCase():p.stack==='output'?'OUTPUT DISTRIBUTION':`${p.stack.toUpperCase()} INPUT`;
      find('#tx-part-title').textContent=p.label;
      let description=p.description;
      if(p.templateId==='enc-norm-2')description=`Residual addition and layer normalization finish encoder layer ${p.layer}. `+(p.layer===6?'These final encoder representations provide keys and values to cross-attention in all six decoder layers.':`The output continues into encoder layer ${p.layer+1}. Cross-attention uses the final output of layer 6, not this intermediate representation.`);
      if(p.templateId==='dec-norm-3')description=`Residual addition and layer normalization finish decoder layer ${p.layer}. `+(p.layer===6?'The output goes to the vocabulary projection and softmax.':`The output continues into decoder layer ${p.layer+1}, whose learned parameters are separate.`);
      if(p.layer&&p.templateId!=='enc-norm-2'&&p.templateId!=='dec-norm-3')description+=` This is ${p.stack} layer ${p.layer}; parameters are not shared with other layers.`;
      find('#tx-part-description').textContent=description;find('#tx-heads').hidden=!['encoder','decoder','cross'].includes(p.family)||!p.layer;
      find('[data-tx=detail]').disabled=!p.layer;
      const index=parts.indexOf(p);find('#tx-tour-status').textContent=`${index+1} / ${parts.length} · ${p.layer?layerName(layerKey(p))+' · ':''}${shortNames[p.templateId]}`;
      find('[data-tx=previous]').disabled=index===0;find('[data-tx=next]').disabled=index===parts.length-1;schedule();
    }
    function zoom(factor){camera.zoom=Math.max(.55,Math.min(3,camera.zoom*factor));find('#tx-zoom').textContent=Math.round(camera.zoom*100)+'%';schedule();}
    const position=p=>[p.x,p.y,p.z];
    function projectPoint(p){const fit=focus?1.45:(spread?.59:.78)*(width<650?1.12:1);return project(p,{...camera,zoom:camera.zoom*fit,panY:camera.panY-(focus?0:height*.035)},width,height);}
    function line(points,color,dashed=false,lineWidth=1){context.beginPath();points.map(projectPoint).forEach((p,i)=>i?context.lineTo(p[0],p[1]):context.moveTo(p[0],p[1]));context.strokeStyle=color;context.lineWidth=lineWidth;context.setLineDash(dashed?[5,4]:[]);context.stroke();context.setLineDash([]);}
    function arrow(points,color,dashed=false){line(points,color,dashed,focus?1.6:1);const a=projectPoint(points[points.length-2]),b=projectPoint(points[points.length-1]),angle=Math.atan2(b[1]-a[1],b[0]-a[0]),size=focus?6:4;context.beginPath();context.moveTo(b[0],b[1]);context.lineTo(b[0]-size*Math.cos(angle-.5),b[1]-size*Math.sin(angle-.5));context.lineTo(b[0]-size*Math.cos(angle+.5),b[1]-size*Math.sin(angle+.5));context.closePath();context.fillStyle=color;context.fill();}
    function label(text,at,{color='#e5edf6',size=11,align='center'}={}){const p=projectPoint(at);context.font=`600 ${size}px system-ui, sans-serif`;context.textAlign=align;context.textBaseline='middle';const tw=context.measureText(text).width,x=align==='left'?p[0]:align==='right'?p[0]-tw:p[0]-tw/2;context.fillStyle='#081423e8';context.fillRect(x-4,p[1]-size/2-3,tw+8,size+6);context.fillStyle=color;context.fillText(text,p[0],p[1]);}
    function draw(){
      context.clearRect(0,0,width,height);hitFaces=[];
      const floor=focus?-4.9:-4.8,extent=focus?3:spread?8:6;
      for(let x=-6;x<=6;x++)line([[x,floor,-extent],[x,floor,extent]],'#162b40');
      for(let z=-extent;z<=extent;z++)line([[-6,floor,z],[6,floor,z]],'#162b40');
      // Enclose every complete layer so its four/six component plates form one block.
      if(!focus)for(const group of [...model.layers].reverse()){
        const first=scene.nodes.find(n=>n.id===group.ids[0]),last=scene.nodes.find(n=>n.id===group.ids[group.ids.length-1]);
        const x0=first.x-1.65,x1=first.x+1.65,y0=first.y-.43,y1=last.y+.43,z0=first.z-.32,z1=first.z+.32;
        const color=group.ids.includes(selected)?'#dceafa':group.stack==='encoder'?'#947449':'#3f8d81';
        line([[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],[x0,y0,z1]],color,false,group.ids.includes(selected)?1.8:.8);
        line([[x0,y1,z1],[x0,y1,z0],[x1,y1,z0],[x1,y1,z1]],color);
        line([[x1,y1,z0],[x1,y0,z0],[x1,y0,z1]],color);
      }
      if(connections)for(const edge of scene.edges){
        const a=scene.nodes.find(p=>p.id===edge.from),b=scene.nodes.find(p=>p.id===edge.to);if(!a||!b)continue;
        const start=position(a),end=position(b);let points,color;
        if(edge.kind==='memory'){
          start[0]+=a.dx;end[0]-=b.dx;
          const rail=focus?-1.5:0;
          points=[start,[rail,start[1],a.z],[rail,end[1],b.z],end];color='#d7a559';
        }else if(edge.kind==='residual'){
          start[0]+=a.dx;end[0]+=b.dx;const rail=b.x+b.dx+.45;
          points=[start,[rail,start[1],0],[rail,end[1],0],end];color='#aebdd0';
        }else{
          start[1]+=a.dy;end[1]-=b.dy;
          if(!focus&&a.layer&&b.layer&&a.layer!==b.layer){
            const rail=a.x+(a.stack==='encoder'?-1.95:1.95);
            points=[start,[rail,start[1]+.35,a.z],[rail,start[1]+.35,b.z],[rail,end[1]-.35,b.z],end];
          }else points=[start,end];
          color=b.stack==='encoder'?'#8f7958':'#4f9488';
        }
        arrow(points,color,edge.kind==='residual');
      }
      const faces=[];
      for(const p of scene.nodes){
        const {x,y,z,dx,dy,dz}=p;
        const vertices=[[-dx,-dy,-dz],[dx,-dy,-dz],[dx,dy,-dz],[-dx,dy,-dz],[-dx,-dy,dz],[dx,-dy,dz],[dx,dy,dz],[-dx,dy,dz]].map(v=>projectPoint([v[0]+x,v[1]+y,v[2]+z]));
        [[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2],[3,2,6,7],[0,1,5,4]].forEach((indices,index)=>{const points=indices.map(i=>vertices[i]);faces.push({part:p,points,depth:points.reduce((n,v)=>n+v[2],0)/4,index});});
      }
      faces.sort((a,b)=>a.depth-b.depth);
      for(const face of faces){
        const {points,part,index}=face,highlight=part.id===selected;
        context.beginPath();points.forEach((p,i)=>i?context.lineTo(p[0],p[1]):context.moveTo(p[0],p[1]));context.closePath();
        context.globalAlpha=1;context.fillStyle='#101d2e';context.fill();context.globalAlpha=part.context?.18:highlight?.9:[.16,.48,.25,.35,.68,.14][index];context.fillStyle=palette[part.family];context.fill();
        context.globalAlpha=1;context.strokeStyle=highlight?'#f6f8ff':palette[part.family];context.lineWidth=highlight?1.8:.55;context.stroke();if(!part.context)hitFaces.push(face);
        if(index===4&&part.layer&&['encoder','decoder','cross'].includes(part.family)){
          // Eight visible strips identify parallel attention heads, not measured activations.
          for(let i=1;i<8;i++){const x=part.x-part.dx+2*part.dx*i/8;line([[x,part.y+part.dy,part.z-part.dz],[x,part.y+part.dy,part.z+part.dz]],'#0a1828',false,.7);}
        }
      }
      if(labels){
        if(!focus){
          for(const group of model.layers){
            const p=scene.nodes.find(n=>n.id===group.ids[0]),last=scene.nodes.find(n=>n.id===group.ids[group.ids.length-1]);
            const left=group.stack==='encoder';label(`${left?'E':'D'}${group.number}`,[p.x-1.6,last.y+.68,p.z+.36],{color:left?'#efbd70':'#70dcc5',size:width<500?10:12,align:'left'});
          }
          for(const stack of ['encoder','decoder']){
            const front=scene.nodes.filter(p=>p.stack===stack&&p.layer===1),top=front[front.length-1];
            label(`${stack.toUpperCase()} · 6 layers`,[top.x,5.7,0],{color:palette[stack],size:width<650?10:14});
            // Label the complete foreground layer without requiring a click or hover.
            for(const p of front)label(shortNames[p.templateId],[p.x,p.y,p.z+p.dz+.06],{color:p.family==='norm'?'#e5edf6':palette[p.family],size:width<650?10:12});
          }
          for(const id of ['source','target','output']){const p=scene.nodes.find(n=>n.id===id);label(shortNames[id],[p.x,p.y+(id==='output'?.65:-.42),p.z+.3],{size:width<650?10:12});}
          const p=scene.nodes.find(n=>n.id===selected);if(p?.layer>1)label(`${p.stack==='encoder'?'E':'D'}${p.layer} · ${shortNames[p.templateId]}`,[p.x,p.y,p.z+p.dz+.06],{color:'#fff',size:width<650?10:12});
        }else for(const p of scene.nodes)label(p.context?p.label:shortNames[p.templateId],[p.x,p.y,p.z+p.dz+.05],{color:p.id===selected?'#fff':'#d1deeb',size:width<600?9:12});
      }
    }
    function resize(){if(dead)return;const rect=stage.getBoundingClientRect();width=Math.max(1,rect.width||800);height=Math.max(1,rect.height||690);const ratio=Math.min(root.devicePixelRatio||1,2);canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);if(context)context.setTransform(ratio,0,0,ratio,0,0);schedule();}
    function updateAttention(){const q=Number(find('#tx-query').value),masked=find('#tx-mask').checked,r=attention(q,masked);find('#tx-q-value').textContent=q.toFixed(1);r.weights.forEach((p,i)=>{find(`[data-weight="${i}"]`).style.width=(p*100)+'%';find(`[data-probability="${i}"]`).textContent=(p*100).toFixed(1)+'%';});find('.tx-result').textContent=`Weighted output = [${r.output.map(x=>x.toFixed(4)).join(', ')}]. ${masked?'Future position C is blocked.':'All three positions are visible.'}`;}
    on(find('#tx-component'),'change',e=>choose(e.target.value));
    on(find('#tx-spread'),'change',e=>{spread=e.target.checked;scene=sceneFor(focus,spread);schedule();});
    on(find('#tx-labels'),'change',e=>{labels=e.target.checked;schedule();});
    on(find('#tx-connections'),'change',e=>{connections=e.target.checked;schedule();});
    for(const input of [find('#tx-query'),find('#tx-mask')])on(input,'input',updateAttention);
    on(host,'click',e=>{
      const button=e.target.closest('button');if(!button)return;
      if(button.dataset.layer){setFocus(button.dataset.layer);choose(model.layers.find(l=>l.key===focus).ids[0]);return;}
      const action=button.dataset.tx;
      if(action==='all')setFocus('');if(action==='detail')setFocus(layerKey(parts.find(p=>p.id===selected)));
      if(action==='in')zoom(1.15);if(action==='out')zoom(1/1.15);if(action==='reset')resetCamera();if(action==='front'){camera=defaultCamera();camera.yaw=0;camera.pitch=0;zoom(1);}
      if(action==='previous'||action==='next'){const index=parts.findIndex(p=>p.id===selected)+(action==='next'?1:-1);if(parts[index])choose(parts[index].id);}
    });
    on(canvas,'keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','_','Home'].includes(e.key))return;e.preventDefault();if(e.key==='Home')resetCamera();else if(['+','='].includes(e.key))zoom(1.1);else if(['-','_'].includes(e.key))zoom(1/1.1);else{const horizontal=['ArrowLeft','ArrowRight'].includes(e.key),amount=['ArrowLeft','ArrowUp'].includes(e.key)?-1:1;if(e.shiftKey)camera[horizontal?'panX':'panY']+=amount*18;else camera[horizontal?'yaw':'pitch']+=amount*.08;camera.pitch=Math.max(-1.15,Math.min(1.15,camera.pitch));schedule();}});
    on(canvas,'wheel',e=>{e.preventDefault();zoom(Math.exp(-Math.max(-150,Math.min(150,e.deltaY))*.002));},{passive:false});on(canvas,'contextmenu',e=>e.preventDefault());
    const pointers=new Map();let gesture=null;
    function pair(){const p=[...pointers.values()];return{x:(p[0].x+p[1].x)/2,y:(p[0].y+p[1].y)/2,d:Math.hypot(p[0].x-p[1].x,p[0].y-p[1].y)};}
    on(canvas,'pointerdown',e=>{if(e.button!==0&&e.button!==2)return;canvas.focus({preventScroll:true});pointers.set(e.pointerId,{x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,moved:false,pan:e.shiftKey||e.button===2});canvas.setPointerCapture?.(e.pointerId);if(pointers.size>=2){pointers.forEach(p=>p.moved=true);gesture=pair();}});
    on(canvas,'pointermove',e=>{const p=pointers.get(e.pointerId);if(!p)return;const dx=e.clientX-p.x,dy=e.clientY-p.y;p.x=e.clientX;p.y=e.clientY;if(Math.hypot(p.x-p.startX,p.y-p.startY)>4)p.moved=true;if(pointers.size>=2){const next=pair();if(gesture){camera.panX+=next.x-gesture.x;camera.panY+=next.y-gesture.y;if(gesture.d>0)zoom(next.d/gesture.d);}gesture=next;}else if(p.pan||e.shiftKey){camera.panX+=dx;camera.panY+=dy;}else{camera.yaw+=dx*.007;camera.pitch=Math.max(-1.15,Math.min(1.15,camera.pitch+dy*.007));}schedule();});
    function release(e){const p=pointers.get(e.pointerId);if(p&&e.type==='pointerup'&&!p.moved&&!p.pan){const rect=canvas.getBoundingClientRect(),point=[e.clientX-rect.left,e.clientY-rect.top],face=[...hitFaces].reverse().find(f=>inside(point,f.points));if(face)choose(face.part.id);}pointers.delete(e.pointerId);gesture=null;if(canvas.hasPointerCapture?.(e.pointerId))canvas.releasePointerCapture(e.pointerId);}
    for(const event of ['pointerup','pointercancel','lostpointercapture'])on(canvas,event,release);
    const observer=root.ResizeObserver?new ResizeObserver(resize):null;observer?.observe(stage);if(!observer)on(root,'resize',resize);
    dispose=()=>{dead=true;controller.abort();observer?.disconnect();if(frame)root.cancelAnimationFrame(frame);pointers.clear();host.remove();};choose(selected);updateAttention();resize();
  }
  root.TransformerExplorer={mount,cleanup};
  if(typeof module!=='undefined'&&module.exports)module.exports={attention,project,inside,parts,model,sceneFor};
})(typeof window!=='undefined'?window:globalThis);
