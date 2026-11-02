/* Run actual frontend JS against live HTTP with a minimal DOM test adapter.
 * This verifies frontend handlers and rendering, not browser layout/accessibility.
 */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const httpFetch = global.fetch;
const baseURL = process.env.HER_TEST_BASE_URL || 'http://127.0.0.1:8000';
const dynamicElements={};
class Element {
  constructor(tag='div'){this.tagName=tag;this.children=[];this.dataset={};this.listeners={};this.hidden=false;this.value='';this.files=[];this.textContent='';this.className='';this.classList={toggle:()=>{},remove:()=>{}};Object.defineProperty(this,'id',{get:()=>this._id,set:value=>{this._id=value;if(value)dynamicElements[value]=this;}});}
  append(...els){this.children.push(...els);}
  appendChild(el){this.append(el);}
  replaceChildren(...els){this.children=els;}
  setAttribute(name,value){this[name]=String(value);}
  removeAttribute(name){delete this[name];}
  addEventListener(type,fn){this.listeners[type]=fn;}
  querySelector(selector){if(selector==='small')return this.children.find(c=>c.tagName==='small');const match=selector.match(/data-id="(\d+)"/);return match?this.children.find(c=>c.dataset.id==match[1]):null;}
  click(){this.listeners.click?.();}
}
const html=fs.readFileSync('frontend/index.html','utf8');
const elements=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element()]));
Object.assign(dynamicElements,elements);
const formParent=new Element('section');formParent.insertBefore=(child,reference)=>{const index=formParent.children.indexOf(reference);formParent.children.splice(index<0?formParent.children.length:index,0,child);child.parentElement=formParent;};formParent.append(elements.analysisForm);elements.analysisForm.parentElement=formParent;
elements.clusters.value='3';elements.query.value='glioma surrounding edema';elements.clinicalText.value='Glioma in frontal lobe with edema measures 12 mm.';
global.document={getElementById:id=>{assert.ok(dynamicElements[id],`missing DOM id ${id}`);return dynamicElements[id];},createElement:tag=>new Element(tag)};
global.location={hash:''};
let latestTutorialResult=null;
global.fetch=async(path,opts)=>{
  const response=await httpFetch(baseURL+path,opts);
  if(/^\/api\/tutorials\/\d+\/run$/.test(path)&&response.ok)latestTutorialResult=await response.clone().json();
  return response;
};
const waitFor=async(predicate)=>{const start=Date.now();while(!predicate()){if(Date.now()-start>60000)throw Error('UI condition timed out: '+elements.status.textContent);await new Promise(r=>setTimeout(r,50));}};
(async()=>{
  vm.runInThisContext(fs.readFileSync('frontend/results-renderer.js','utf8'));
  vm.runInThisContext(fs.readFileSync('frontend/lab.js','utf8'));
  await waitFor(()=>elements.tutorialNav.children.length===15);
  assert.match(elements.connection.textContent,/Connected/);
  for(let id=1;id<=15;id++){
    elements.tutorialNav.children[id-1].click();
    assert.equal(elements.tutorialNumber.textContent,`TUTORIAL ${String(id).padStart(2,'0')}`);
    if(id===10)dynamicElements.t10Clustering.click();
    if([10,11,12].includes(id)){
      assert.equal(elements.fileControls.hidden,false,`T${id} must expose the synthetic text CSV controls`);
      assert.equal(elements.exampleLink.hidden,false,`T${id} must expose the synthetic text CSV download`);
      assert.equal(elements.exampleLink.href,`/api/tutorials/${id}/example`);
    } else {
      assert.equal(elements.fileControls.hidden,true,`T${id} must not expose text upload controls`);
      assert.equal(elements.exampleLink.hidden,true,`T${id} must not expose a synthetic text CSV download`);
      assert.equal(elements.exampleLink.href,undefined,`T${id} must not retain an unsupported example URL`);
    }
    if(id===11)elements.clinicalText.value='<img src=x onerror=alert(1)> Glioma in frontal lobe with edema measures 12 mm.';
    elements.analysisForm.listeners.submit({preventDefault(){}});
    await waitFor(()=>!elements.run.disabled);
    assert.equal(elements.runState.textContent,'Run completed',elements.status.textContent);
    assert.ok(elements.results.children.length>2);
    assert.equal(elements.download.hidden,false);
    const container=elements.results.children.find(el=>el.className==='plots');
    assert.equal(container.children.length,latestTutorialResult.plots.length,`T${id}: every returned plot must render`);
    for(const [index,plot] of latestTutorialResult.plots.entries()){
      const figure=container.children[index];
      if(plot.wide)assert.equal(figure.className,'plot-wide','Multi-panel MRI figures must span the result grid');
      assert.equal(figure.children.find(el=>el.tagName==='figcaption').textContent,plot.title);
      assert.equal(figure.children.find(el=>el.tagName==='img').src,'data:image/png;base64,'+plot.png);
      if(plot.caption)assert.equal(figure.children.find(el=>el.className==='plot-caption').textContent,plot.caption);
    }
    if(id===10){
      const terms=elements.results.children.find(el=>el.className==='cluster-terms');assert.ok(terms);
      assert.equal(terms.children.filter(el=>el.tagName==='article').length,(latestTutorialResult.tables.clusters||latestTutorialResult.tables.cluster_representatives).length);
    }
    if(id===11){
      const section=elements.results.children.find(el=>el.className==='text-annotations');assert.ok(section);
      const paragraph=section.children.find(el=>el.className==='annotated-source');
      assert.equal(paragraph.children.map(el=>el.textContent).join(''),latestTutorialResult.annotations.text);
      assert.ok(paragraph.children.some(el=>el.tagName==='mark'));
      assert.ok(paragraph.children.every(el=>['span','mark'].includes(el.tagName)),'Source text must not become HTML');
    }
    console.log(`T${id}: real API -> ${container.children.length} plots rendered with exact PNGs and captions`);
  }
  // Exercise the distinct CNN button; the regular T5 Run analysis uses SIFT.
  elements.tutorialNav.children[4].click();
  elements.cnn.click();
  await waitFor(()=>!elements.cnn.disabled);
  assert.equal(elements.runState.textContent,'Run completed',elements.status.textContent);
  const gallery=elements.results.children.find(el=>el.className==='cnn-visuals');
  assert.ok(gallery,'CNN images must have a prominent gallery');
  assert.ok(elements.results.children.indexOf(gallery)<elements.results.children.findIndex(el=>el.className==='metrics'));
  const samples=gallery.children.filter(el=>el.className==='cnn-sample');
  assert.equal(samples.length,4);
  for(const sample of samples){
    const comparison=sample.children.find(el=>el.className==='cnn-comparison');
    assert.equal(comparison.children.length,3);
    for(const card of comparison.children){
      const image=card.children.find(el=>el.tagName==='img');
      assert.match(image.src,/^data:image\/png;base64,/);
    }
    assert.match(comparison.children[2].children.find(el=>el.className==='cnn-score').textContent,/Predicted-class score:/);
  }
  const plotContainer=elements.results.children.find(el=>el.className==='plots');
  assert.equal(plotContainer.children.length,4,'CNN must render input/mask/prediction, accuracy, loss and confusion matrix plots');
  for(const figure of plotContainer.children){
    const image=figure.children.find(el=>el.tagName==='img');
    assert.match(image.src,/^data:image\/png;base64,/);
    assert.ok(Buffer.from(image.src.split(',')[1],'base64').subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])));
  }
  const tables=elements.results.children.filter(el=>el.tagName==='details').map(el=>el.children[0].textContent);
  for(const name of ['predictions','training history','classification report','confusion matrix','split']){
    assert.ok(tables.some(title=>title.startsWith(name)),`CNN missing rendered table: ${name}`);
  }
  console.log('CNN button: real training/inference -> 4 PNG plots and all 5 result tables rendered');
  // Verify graceful image/optional-mask/old-server and API-error states without
  // repeatedly training a model. All predictions above came from real HTTP.
  const liveFetch=global.fetch;
  const realCNN=await (await liveFetch('/api/cnn/demo',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})).json();
  const missingMask=structuredClone(realCNN);missingMask.previews[0].ground_truth_mask=null;
  global.fetch=async()=>({ok:true,json:async()=>missingMask});
  elements.cnn.click();await waitFor(()=>!elements.cnn.disabled);
  const missingGallery=elements.results.children.find(el=>el.className==='cnn-visuals');
  const firstSample=missingGallery.children.find(el=>el.className==='cnn-sample');
  const cards=firstSample.children.find(el=>el.className==='cnn-comparison').children;
  assert.match(cards[1].children.find(el=>el.className==='warning').textContent,/Image unavailable/);
  const broken=cards[0].children.find(el=>el.tagName==='img');broken.listeners.error();
  assert.equal(broken.hidden,true);assert.ok(cards[0].children.some(el=>el.textContent.includes('could not be decoded')));
  global.fetch=async()=>({ok:true,json:async()=>({metrics:{},tables:{},plots:[]})});
  elements.cnn.click();await waitFor(()=>!elements.cnn.disabled);
  const empty=elements.results.children.find(el=>el.className==='cnn-visuals');
  assert.ok(empty.children.some(el=>el.textContent.includes('Restart the project backend')));
  global.fetch=async()=>({ok:false,json:async()=>({detail:'CNN data unavailable'})});
  elements.cnn.click();await waitFor(()=>!elements.cnn.disabled);
  assert.equal(elements.runState.textContent,'Run failed');assert.equal(elements.results.children.length,0);
  global.fetch=liveFetch;
  // Generic visualization-decode and empty-result states also remain readable.
  elements.tutorialNav.children[12].click();
  global.fetch=async()=>({ok:true,json:async()=>({tutorial:13,tables:{},plots:[{title:'Missing plot'}]})});
  elements.analysisForm.listeners.submit({preventDefault(){}});await waitFor(()=>!elements.run.disabled);
  let generic=elements.results.children.find(el=>el.className==='plots');
  assert.ok(generic.children[0].children.some(el=>el.textContent.includes('Visualization unavailable')));
  global.fetch=async()=>({ok:true,json:async()=>({tutorial:13,tables:{},plots:[{title:'Decode failure',png:'bad'}]})});
  elements.analysisForm.listeners.submit({preventDefault(){}});await waitFor(()=>!elements.run.disabled);
  generic=elements.results.children.find(el=>el.className==='plots');
  const badImage=generic.children[0].children.find(el=>el.tagName==='img');badImage.listeners.error();
  assert.ok(badImage.hidden);assert.ok(generic.children[0].children.some(el=>el.textContent.includes('could not be decoded')));
  global.fetch=async()=>({ok:true,json:async()=>({tutorial:13,tables:{},plots:[]})});
  elements.analysisForm.listeners.submit({preventDefault(){}});await waitFor(()=>!elements.run.disabled);
  assert.ok(elements.results.children.some(el=>el.textContent.includes('No plot applies')));
  global.fetch=liveFetch;
  elements.tutorialNav.children[9].click();
  dynamicElements.t10Classification.click();
  const csv=await (await fetch('/api/tutorials/10/example')).text();
  elements.csv.files=[{name:'demo.csv',size:csv.length,text:async()=>csv}];
  elements.analysisForm.listeners.submit({preventDefault(){}});await waitFor(()=>!elements.run.disabled);
  assert.equal(elements.runState.textContent,'Run completed');
  elements.csv.files=[{name:'bad.exe',size:4,text:async()=>''}];
  elements.analysisForm.listeners.submit({preventDefault(){}});await waitFor(()=>!elements.run.disabled);
  assert.match(elements.status.textContent,/\.csv/);
  elements.csv.files=[];global.fetch=async()=>{throw new TypeError('Network down');};
  elements.analysisForm.listeners.submit({preventDefault(){}});await waitFor(()=>!elements.run.disabled);
  assert.match(elements.status.textContent,/Cannot connect/);
  console.log('PASS: 15 tutorial flows, real CSV upload, rejected file type, backend disconnect. DOM adapter only; no visual browser verification.');
})().catch(e=>{console.error(e);process.exitCode=1;});
