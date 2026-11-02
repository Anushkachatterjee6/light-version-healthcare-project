/* Studio controller only: computation and visualization are shared. */
(() => {
  'use strict';
  const run=document.getElementById('cnnStudioRun');
  const status=document.getElementById('cnnStudioStatus');
  const results=document.getElementById('cnnStudioResults');
  let busy=false;
  run.addEventListener('click',async()=>{
    if(busy)return;
    busy=true;run.disabled=true;results.hidden=true;results.replaceChildren();
    results.setAttribute('aria-busy','true');
    status.className='status';status.textContent='Training the compact CNN and evaluating held-out MRI slices…';
    try{
      const response=await fetch('/api/cnn/demo',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
      const result=await response.json();
      if(!response.ok)throw Error(typeof result.detail==='string'?result.detail:JSON.stringify(result.detail||'CNN analysis failed'));
      if(!result||result.workflow!=='training_then_validation_inference'||!result.tables?.predictions?.length){
        throw Error('CNN response has no completed predictions. Restart the project backend and try again.');
      }
      globalThis.HERResults.render(results,result,true);
      results.hidden=false;
      status.className='status ok';status.textContent='CNN training and validation inference completed. Results are shown below.';
    }catch(error){
      results.replaceChildren();results.hidden=true;
      status.className='status err';
      status.textContent=error instanceof TypeError?'Cannot connect to the backend. Start run_server.bat and reopen the studio.':
        error instanceof SyntaxError?'CNN backend returned an unreadable response. Restart the project backend and try again.':error.message;
    }finally{
      busy=false;run.disabled=false;results.setAttribute('aria-busy','false');
    }
  });
})();
