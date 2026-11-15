'use strict';
/* Schematic teaching animations: no model results or measured weights are invented. */
let motionTimer=null,motionStep=0;
const MOTION_LESSONS={
  'attention-cache':{title:'See attention happen.',labels:['Query','Keys','Weights','Values','Output'],steps:[
    '1 / 4 — Choose a query: the representation asking which available tokens matter.',
    '2 / 4 — Compare the query with each allowed key. Scale the scores and apply the attention mask.',
    '3 / 4 — Softmax converts the allowed scores into weights that sum to one.',
    '4 / 4 — Use those weights to combine the value vectors into the output for this query.'
  ]},
  'expert-routing':{title:'Follow a token through the experts.',labels:['Token','Router','Expert 1','Expert 2','Expert 3','Expert 4','Mix'],steps:[
    '1 / 4 — A token representation enters the router.',
    '2 / 4 — The router scores the experts. In this top-1 teaching example, expert 1 is selected.',
    '3 / 4 — The selected expert processes the token. The other experts stay inactive for this token.',
    '4 / 4 — Weight the selected output by its routing probability. All experts still occupy model memory.'
  ]}
};
function motionNode(x,y,label,cls=''){return '<g class="motion-node '+cls+'"><rect x="'+x+'" y="'+y+'" width="98" height="44" rx="10"/><text x="'+(x+49)+'" y="'+(y+27)+'" text-anchor="middle">'+label+'</text></g>';}
function motionGraphic(id){
  if(id==='expert-routing')return '<path class="motion-edge" d="M108 114H148M246 114H265V31H300M265 114V87H300M265 114V143H300M265 114V199H300M398 31H430V114H462M398 87H430M398 143H430V114M398 199H430V114"/>'+motionNode(10,92,'Token','input-node')+motionNode(148,92,'Router','router-node')+[0,1,2,3].map((n)=>motionNode(300,9+n*56,'Expert '+(n+1),'expert expert-'+n)).join('')+motionNode(462,92,'Mix','output-node');
  return '<path class="motion-edge" d="M108 65H168M108 161H140V87H168M266 87H290M388 87H420V135H462M388 183H420V135"/>'+motionNode(10,43,'Query','input-node')+motionNode(10,139,'Keys','input-node')+motionNode(168,65,'Scores','router-node')+motionNode(290,65,'Weights','weights-node')+motionNode(290,161,'Values','value-node')+motionNode(462,113,'Output','output-node');
}
function enhanceMotion(){
  clearTimeout(motionTimer);motionTimer=null;motionStep=0;
  const id=route().split('/')[1],spec=MOTION_LESSONS[id],article=document.querySelector('.lesson-layout .reading');if(!spec||!article)return;
  const section=document.createElement('section');section.className='concept-motion';section.dataset.model=id;section.dataset.step='0';
  section.innerHTML='<div class="section-heading"><div><span class="eyebrow">A MOVING MENTAL MODEL</span><h2>'+spec.title+'</h2></div><span class="tag">Conceptual illustration</span></div><svg viewBox="0 0 580 240" role="img" aria-label="'+(id==='expert-routing'?'A token passes through a router to one of four experts; its output is weighted.':'A query and keys produce scores and weights; weighted values form an output.')+'">'+motionGraphic(id)+'</svg><p class="motion-caption" aria-live="polite"></p><div class="motion-controls"><button class="button secondary small" data-motion="play">Play explanation</button><button class="button secondary small" data-motion="next">Next step →</button><button class="text-button" data-motion="reset">Reset</button></div><p class="fineprint">Illustrative steps, not recorded model activations. '+(matchMedia('(prefers-reduced-motion: reduce)').matches?'Reduced motion is on. Use Next step to advance.':'Nothing plays until you choose Play.')+'</p>';
  article.prepend(section);
  const paint=()=>{section.dataset.step=String(motionStep);section.querySelector('.motion-caption').textContent=spec.steps[motionStep];};
  const stop=()=>{clearTimeout(motionTimer);motionTimer=null;section.querySelector('[data-motion=play]').textContent='Play explanation';};
  const advance=()=>{if(!section.isConnected){stop();return;}if(motionStep===3){stop();return;}motionStep++;paint();motionTimer=setTimeout(advance,3500);};
  section.querySelector('[data-motion=play]').onclick=()=>{if(motionTimer){stop();return;}if(matchMedia('(prefers-reduced-motion: reduce)').matches){motionStep=(motionStep+1)%4;paint();return;}motionStep=0;paint();section.querySelector('[data-motion=play]').textContent='Pause';motionTimer=setTimeout(advance,3500);};
  section.querySelector('[data-motion=next]').onclick=()=>{stop();motionStep=(motionStep+1)%4;paint();};
  section.querySelector('[data-motion=reset]').onclick=()=>{stop();motionStep=0;paint();};paint();
}
