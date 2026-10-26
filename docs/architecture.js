const canvas=document.querySelector('#diagram-canvas');
const fit=document.querySelector('#diagram-fit'),zoom=document.querySelector('#diagram-zoom');
function setZoom(on){canvas.classList.toggle('is-zoomed',on);fit.setAttribute('aria-pressed',String(!on));zoom.setAttribute('aria-pressed',String(on));canvas.scrollTo(0,0)}
fit.addEventListener('click',()=>setZoom(false));zoom.addEventListener('click',()=>setZoom(true));
