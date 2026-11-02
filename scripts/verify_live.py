"""Exercise actual HTTP routes against the running local Windows server."""
import json
from pathlib import Path
import time
import httpx

base='http://127.0.0.1:8000'
results=[]
with httpx.Client(base_url=base,timeout=180) as c:
    for path in ['/','/image-analysis','/static/lab.css','/static/lab.js','/api/health','/api/tutorials']:
        r=c.get(path);r.raise_for_status();results.append({'route':path,'status':r.status_code})
    for i in range(1,16):
        started=time.time();r=c.post(f'/api/tutorials/{i}/run',json={});r.raise_for_status()
        data=r.json();assert data['tables'] or data['plots']
        results.append({'tutorial':i,'status':r.status_code,'seconds':round(time.time()-started,2),'metrics':data['metrics']})
    cases=c.get('/api/cases').json()['cases'];assert cases
    case=cases[0]; meta=c.get(f"/api/cases/{case['case_id']}/meta").json()
    z=meta['num_slices']//2
    r=c.post('/api/image-analysis/sift-ransac',json={'case_id':case['case_id'],'slice1':z,'slice2':z+2})
    r.raise_for_status();assert r.json()['ransac_visualization'];results.append({'route':'SIFT-RANSAC','metrics':{k:r.json()[k] for k in ['good_matches','ransac_inliers','inlier_ratio']}})
    bad=c.post('/api/image-analysis/sift-ransac',json={'case_id':case['case_id'],'slice1':z,'slice2':z})
    assert bad.status_code==400
    started=time.time();r=c.post('/api/cnn/demo',json={});r.raise_for_status()
    data=r.json();assert data['tables']['predictions']
    results.append({'route':'CNN training and inference','status':r.status_code,'seconds':round(time.time()-started,2),'metrics':data['metrics']})
out=Path(__file__).resolve().parents[1]/'artifacts';out.mkdir(exist_ok=True)
(out/'live_verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
