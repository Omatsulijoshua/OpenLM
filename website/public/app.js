const phases = [
  {title:'Meet the devices.',detail:'The coordinator discovers paired workers on the same local network. Each device offers a memory budget.',states:['Ready','Paired','Paired'],answer:'Waiting for the cluster',active:[]},
  {title:'Give the weights a home.',detail:'An illustrative 18 GB of weights is placed across the three devices: 9 GB, 6 GB and 3 GB. Each holds a different part of the model.',states:['9 GB placed','6 GB placed','3 GB placed'],answer:'Weights loaded. Ready for a prompt.',active:[0,1,2]},
  {title:'Pass the computation along.',detail:'Layer groups compute on their assigned devices. Intermediate activations cross the network; the full model is not copied to every worker.',states:['Computing','Computing','Computing'],answer:'Processing “Why share a model?”',active:[0,1,2]},
  {title:'One conversation comes back.',detail:'The coordinator returns the next token to the chat. Generation repeats this process for each token. The speed depends on your hardware and network.',states:['Returning token','Connected','Connected'],answer:'So a model can use more memory than one device can offer.',active:[0]}
];
let step=0,timer=null;
const network=document.querySelector('.device-network');
const play=document.querySelector('#play');
const buttons=[...document.querySelectorAll('[data-step]')];
function showStep(next){
  step=next; const phase=phases[step];network.dataset.phase=step;
  document.querySelector('#phase-number').textContent=`STEP 0${step+1} / 04`;
  document.querySelector('#phase-title').textContent=phase.title;
  document.querySelector('#phase-detail').textContent=phase.detail;
  const answer=document.querySelector('#answer');answer.textContent=phase.answer;
  const cursor=document.createElement('span');cursor.className='cursor';cursor.textContent='▍';answer.append(cursor);
  document.querySelectorAll('.device').forEach((el,i)=>{el.classList.toggle('active',phase.active.includes(i));el.querySelector('.device-state').textContent=phase.states[i];});
  buttons.forEach((button,i)=>{button.classList.toggle('selected',i===step);button.setAttribute('aria-pressed',String(i===step));});
}
function pause(){clearInterval(timer);timer=null;play.textContent='▶ Play walkthrough';play.setAttribute('aria-pressed','false');}
buttons.forEach(button=>button.addEventListener('click',()=>{pause();showStep(Number(button.dataset.step));}));
play.addEventListener('click',()=>{
  if(timer){pause();return;}
  if(step===3)showStep(0);
  play.textContent='Ⅱ Pause walkthrough';play.setAttribute('aria-pressed','true');
  timer=setInterval(()=>{if(step<3)showStep(step+1);else pause();},4000);
});
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
const dialog=document.querySelector('#lightbox');let opener;
document.querySelectorAll('.shot').forEach(button=>button.addEventListener('click',()=>{
  opener=button;const img=document.querySelector('#large-image');img.src=button.dataset.image;img.alt=button.querySelector('img').alt;dialog.showModal();
}));
document.querySelector('#close-lightbox').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
dialog.addEventListener('close',()=>opener?.focus());
const storeDialog=document.querySelector('#store-preview');let storeOpener;
let publishedStores=0;
document.querySelectorAll('[data-store]').forEach(link=>{
  const url=window.OPENLM_STORES?.[link.dataset.store];
  const valid=link.dataset.store==='googlePlay'?/^https:\/\/play\.google\.com\/store\/apps\/details\?/.test(url||''):/^https:\/\/apps\.apple\.com\//.test(url||'');
  if(valid){link.href=url;link.querySelector('span').textContent='DOWNLOAD';link.setAttribute('aria-label',link.dataset.store==='googlePlay'?'Download OpenLM on Google Play':'Download OpenLM on the App Store');publishedStores++;}
  else link.addEventListener('click',event=>{event.preventDefault();storeOpener=link;storeDialog.showModal();});
});
if(publishedStores===2)document.querySelector('#store-status').textContent='Download OpenLM for Android or iOS.';
else if(publishedStores===1)document.querySelector('#store-status').textContent='One store is available. The other is coming soon.';
document.querySelector('#close-store-preview').addEventListener('click',()=>storeDialog.close());
storeDialog.addEventListener('close',()=>storeOpener?.focus());
