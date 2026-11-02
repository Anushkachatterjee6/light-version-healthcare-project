/* Actual studio scripts + live HTTP, with a DOM adapter (not a browser). */
const fs=require('node:fs');const vm=require('node:vm');const assert=require('node:assert/strict');
const base=process.env.HER_TEST_BASE_URL||'http://127.0.0.1:8000';const nativeFetch=global.fetch;
class Element{
  constructor(tag='div'){this.tagName=tag;this.children=[];this.listeners={};this.attributes={};this.value='';this.hidden=false;this.disabled=false;this.textContent='';this.className='';this.classes=new Set();this.classList={remove:c=>this.classes.delete(c)};}
  append(...els){for(const el of els){this.children.push(el);if(this.tagName==='select'&&!this.value)this.value=el.value;}}
  appendChild(el){this.append(el);}replaceChildren(...els){this.children=els;}
  setAttribute(key,value){this.attributes[key]=value;}
  addEventListener(type,fn){this.listeners[type]=fn;}click(){return this.listeners.click?.();}
  scrollIntoView(){}
  set innerHTML(value){this.html=value;this.children=[];if(this.tagName==='select')this.value='';}
  get innerHTML(){return this.html;}
}
const html=fs.readFileSync('frontend/image-analysis.html','utf8');
const elements={};
for(const match of html.matchAll(/<([a-z0-9]+)[^>]*\bid="([^"]+)"[^>]*>/gi)){
  const el=new Element(match[1]);el.hidden=/\shidden(?:\s|>)/.test(match[0]);elements[match[2]]=el;
}
elements.results.classes.add('hidden');
global.document={getElementById:id=>{assert.ok(elements[id],`missing studio id ${id}`);return elements[id];},createElement:tag=>new Element(tag)};
let requests=[];let cnnOutput;
global.fetch=async(path,opts)=>{
  requests.push({path,opts});const r=await nativeFetch(base+path,opts);
  if(path==='/api/cnn/demo'&&r.ok)cnnOutput=await r.clone().json();return r;
};
const waitFor=async(fn)=>{const start=Date.now();while(!fn()){if(Date.now()-start>90000)throw Error('Timed out: '+elements.cnnStudioStatus.textContent+' / '+elements.statusMsg.textContent);await new Promise(r=>setTimeout(r,50));}};
const descendants=el=>[el,...el.children.flatMap(descendants)];
async function sift(){
  elements.runBtn.click();await waitFor(()=>!elements.runBtn.disabled);
  assert.equal(elements.results.classes.has('hidden'),false);
  for(const id of ['orig1','orig2','sift1','sift2','matches','ransac'])assert.match(elements[id].src,/^data:image\/png;base64,/);
  assert.ok(elements.stats.innerHTML.includes('RANSAC Inliers'));
}
(async()=>{
  assert.ok(html.includes('href="#cnn-analysis"')&&html.includes('href="#sift-analysis"'));
  vm.runInThisContext(fs.readFileSync('frontend/results-renderer.js','utf8'));
  vm.runInThisContext(fs.readFileSync('frontend/image-analysis.js','utf8'));
  vm.runInThisContext(fs.readFileSync('frontend/studio-cnn.js','utf8'));
  assert.equal(elements.cnnStudioResults.hidden,true);
  assert.equal(requests.filter(r=>r.path==='/api/cnn/demo').length,0,'Page load must not train CNN');
  await waitFor(()=>elements.statusMsg.textContent.startsWith('Ready.'));
  await sift();const siftImage=elements.ransac.src;
  elements.cnnStudioRun.click();elements.cnnStudioRun.click();
  assert.ok(elements.cnnStudioRun.disabled);assert.ok(elements.cnnStudioResults.hidden);
  assert.equal(elements.cnnStudioResults.attributes['aria-busy'],'true');
  await waitFor(()=>!elements.cnnStudioRun.disabled);
  const cnnCalls=requests.filter(r=>r.path==='/api/cnn/demo');
  assert.equal(cnnCalls.length,1,'Repeated clicks while processing must not retrain');
  assert.equal(cnnCalls[0].opts.method,'POST');assert.equal(cnnCalls[0].opts.body,'{}');
  assert.equal(elements.cnnStudioResults.hidden,false);
  assert.ok(elements.cnnStudioStatus.className.includes('ok'),elements.cnnStudioStatus.textContent);
  assert.equal(elements.ransac.src,siftImage,'CNN must not replace SIFT results');
  const nodes=descendants(elements.cnnStudioResults);
  const previews=nodes.filter(n=>n.className==='cnn-sample');assert.equal(previews.length,cnnOutput.previews.length);
  for(const preview of cnnOutput.previews){
    assert.ok(nodes.some(n=>n.tagName==='img'&&n.src===preview.input_image));
    if(preview.ground_truth_mask)assert.ok(nodes.some(n=>n.tagName==='img'&&n.src===preview.ground_truth_mask));
    assert.ok(nodes.some(n=>n.textContent.includes(preview.predicted_class)));
  }
  assert.ok(nodes.some(n=>n.className==='cnn-score'));
  for(const plot of cnnOutput.plots){
    assert.ok(nodes.some(n=>n.tagName==='img'&&n.src==='data:image/png;base64,'+plot.png));
    assert.ok(nodes.some(n=>n.tagName==='figcaption'&&n.textContent===plot.title));
  }
  for(const name of Object.keys(cnnOutput.tables))assert.ok(nodes.some(n=>n.tagName==='summary'&&n.textContent.startsWith(name.replaceAll('_',' '))));
  const cnnChildren=elements.cnnStudioResults.children.slice();await sift();
  assert.deepEqual(elements.cnnStudioResults.children,cnnChildren,'SIFT must not clear CNN results');
  const liveFetch=global.fetch;
  for(const error of [{status:409,detail:'CNN demo already running'},{status:503,detail:'No labeled cases'}]){
    global.fetch=async()=>({ok:false,json:async()=>({detail:error.detail})});
    await elements.cnnStudioRun.click();assert.ok(elements.cnnStudioResults.hidden);assert.equal(elements.cnnStudioResults.children.length,0);
    assert.equal(elements.cnnStudioStatus.textContent,error.detail);assert.equal(elements.cnnStudioRun.disabled,false);
  }
  global.fetch=async()=>({ok:true,json:async()=>({tables:{predictions:[]}})});
  await elements.cnnStudioRun.click();assert.match(elements.cnnStudioStatus.textContent,/no completed predictions/);
  global.fetch=async()=>({ok:false,json:async()=>{throw new SyntaxError('invalid JSON');}});
  await elements.cnnStudioRun.click();assert.match(elements.cnnStudioStatus.textContent,/unreadable response/);
  global.fetch=async()=>{throw new TypeError('offline');};
  await elements.cnnStudioRun.click();assert.match(elements.cnnStudioStatus.textContent,/Cannot connect/);
  global.fetch=liveFetch;
  const css=fs.readFileSync('frontend/studio-cnn.css','utf8');
  assert.ok(css.includes('@media(max-width:800px)')&&css.includes('height:auto'));
  console.log('PASS: studio SIFT -> CNN -> SIFT, existing endpoint called once, real gallery/4 plots/5 tables, independent results, loading/duplicate-click guard, API/empty/network/JSON errors. DOM adapter only; browser layout unverified.');
})().catch(e=>{console.error(e);process.exitCode=1;});
