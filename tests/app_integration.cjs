/* Main application asset and runtime smoke test using its real HTTP API. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const base=process.env.HER_TEST_BASE_URL||'http://127.0.0.1:8000',nativeFetch=global.fetch;
class Element{constructor(){this.children=[];this.textContent='';this.innerHTML='';}append(child){this.children.push(child);}}
const ids=['connection','stats','completeness','limitations','models'];const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
global.document={getElementById:id=>elements[id],createElement:()=>new Element()};
global.fetch=path=>nativeFetch(base+path);
(async()=>{vm.runInThisContext(fs.readFileSync('frontend/app.js','utf8'));for(let elapsed=0;!elements.connection.textContent&&elapsed<10000;elapsed+=50)await new Promise(resolve=>setTimeout(resolve,50));
  assert.match(elements.connection.textContent,/Connected/);assert.equal(elements.stats.children.length,5);assert.match(elements.completeness.innerHTML,/%/);assert.equal(elements.models.children.length,2);
  console.log('PASS: main application JavaScript initialized with real application-summary API data.');
})().catch(error=>{console.error(error);process.exitCode=1;});
