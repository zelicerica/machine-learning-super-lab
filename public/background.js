'use strict';
/* One local MP4 instance persists across routes; no third-party player or requests. */
let backgroundPreference=null,backgroundMedia=null,backgroundBlocked=false;
function updateBackgroundTone(tone){
  const dimmed=tone==='dimmed',button=document.getElementById('backgroundToneToggle');
  document.body.dataset.backgroundTone=dimmed?'dimmed':'original';
  button.textContent=dimmed?'Original video':'Dim video';
  button.setAttribute('aria-pressed',String(dimmed));
  button.setAttribute('aria-label',dimmed?'Show original video brightness':'Dim background video for reading');
  button.title=dimmed?'Restore the original video brightness':'Use the dimmed reading background';
}
function backgroundWanted(){return backgroundPreference==='on'||(backgroundPreference!=='off'&&!backgroundMedia.matches);}
function backgroundButton(){
  const video=document.getElementById('labBackgroundVideo'),button=document.getElementById('backgroundToggle');
  const playing=backgroundWanted()&&!backgroundBlocked;
  button.textContent=playing?'Pause motion':'Play motion';
  button.setAttribute('aria-label',playing?'Pause background video':'Play background video');
  button.setAttribute('aria-pressed',String(playing));
  button.title=backgroundMedia.matches&&backgroundPreference!=='on'?'Reduced motion is enabled. Play to opt in.':'Pause or play the decorative video background';
  document.body.dataset.backgroundMotion=playing?'playing':'paused';
}
function updateBackgroundPlayback(){
  const video=document.getElementById('labBackgroundVideo');
  if(backgroundWanted()&&!document.hidden){
    video.muted=true;const request=video.play();
    if(request?.catch)request.catch(()=>{backgroundBlocked=true;backgroundButton();});
  }else video.pause();
  backgroundButton();
}
function updateBackgroundRoute(){
  const current=location.hash.replace(/^#\/?/,'').split('/')[0];
  document.body.dataset.view=(!current||current==='home')?'home':'lesson';
}
function initBackgroundVideo(){
  const video=document.getElementById('labBackgroundVideo'),button=document.getElementById('backgroundToggle');
  updateBackgroundTone(localStorage.getItem('super-lab-background-tone'));
  document.getElementById('backgroundToneToggle').addEventListener('click',()=>{
    const tone=document.body.dataset.backgroundTone==='dimmed'?'original':'dimmed';
    localStorage.setItem('super-lab-background-tone',tone);updateBackgroundTone(tone);
  });
  backgroundPreference=localStorage.getItem('super-lab-background-motion');
  backgroundMedia=matchMedia('(prefers-reduced-motion: reduce)');
  button.addEventListener('click',()=>{
    backgroundPreference=backgroundWanted()&&!backgroundBlocked?'off':'on';backgroundBlocked=false;
    localStorage.setItem('super-lab-background-motion',backgroundPreference);updateBackgroundPlayback();
  });
  video.addEventListener('play',()=>{backgroundBlocked=false;backgroundButton();});
  video.addEventListener('error',()=>{backgroundBlocked=true;backgroundButton();button.disabled=true;button.title='Video unavailable; the still background remains visible.';});
  document.addEventListener('visibilitychange',updateBackgroundPlayback);
  backgroundMedia.addEventListener?.('change',()=>{backgroundBlocked=false;updateBackgroundPlayback();});
  updateBackgroundRoute();updateBackgroundPlayback();
}
