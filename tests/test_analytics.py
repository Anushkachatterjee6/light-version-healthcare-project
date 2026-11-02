"""Regression tests for the integrated local-MRI T1–T15 workspace."""
import json
import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app
from her_core.analytics import example_data, mri_case_features, run_analysis
from her_core.cnn import patient_level_split, _prepare_slice
from her_core.preprocess import clean_image, minmax_normalize, zscore_standardize


class BrainTumourAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results={i:run_analysis(i) for i in range(7,16)}

    def test_all_tutorials_are_serializable_and_brain_tumour_aligned(self):
        for tutorial,result in self.results.items():
            with self.subTest(tutorial=tutorial):
                self.assertEqual(result['tutorial'],tutorial); self.assertTrue(result['tables']); self.assertTrue(result['interpretation'])
                json.dumps(result,allow_nan=False)
                if tutorial in (10,11,12): self.assertIn('synthetic',result['dataset'].lower())
                else: self.assertIn('BraTS',result['dataset'])

    def test_local_features_do_not_invent_missing_labels_or_modalities(self):
        df=mri_case_features(); self.assertEqual(len(df),8); self.assertEqual(int(df.has_seg.sum()),6)
        unknown=df[df.has_seg==0]; self.assertTrue((unknown.tumour_label=='Unknown — no segmentation').all())
        self.assertTrue(unknown.tumour_fraction.isna().all())
        self.assertGreaterEqual(int(df.has_flair.sum()),6)

    def test_mri_workflow_outputs_are_from_case_features(self):
        self.assertIn('review_queue',self.results[7]['tables'])
        self.assertIn('MRI modality and segmentation availability',[p['title'] for p in self.results[8]['plots']])
        self.assertIn('MRI feature correlation heatmap',[p['title'] for p in self.results[9]['plots']])
        self.assertIn('PCA projection of MRI-derived case features',[p['title'] for p in self.results[13]['plots']])
        self.assertIn('Imaging feature and tumour-burden correlations',[p['title'] for p in self.results[14]['plots']])
        self.assertIn('heldout_tumour_burden_predictions',self.results[15]['tables'])
        self.assertIn('not tumour/no-tumour classification',' '.join(self.results[15]['warnings']))

    def test_synthetic_text_is_explicit_and_computed(self):
        r=self.results[10]
        self.assertTrue(0<=r['metrics']['accuracy']<=1); self.assertIn('confusion_matrix',r['tables'])
        entities=run_analysis(11,text='Glioma in frontal lobe with edema measures 12 mm.')['tables']['entities']
        self.assertEqual([e['entity'] for e in entities],['Glioma','frontal lobe','edema','12 mm'])
        ranking=run_analysis(12,query='glioma edema')['tables']['ranking']; self.assertGreaterEqual(ranking[0]['score'],ranking[-1]['score'])

    def test_no_legacy_operational_or_adult_csv_pipeline(self):
        for tutorial in (7,8,9,13,14,15):
            with self.assertRaises(ValueError): run_analysis(tutorial,[{'date':'2025-01-01','admissions':1}])
        self.assertNotIn('admissions',mri_case_features().columns); self.assertNotIn('BMI',mri_case_features().columns)

    def test_preprocessing_and_patient_split_remain_safe(self):
        x=np.array([[np.nan,np.inf],[1,3]],dtype=float); self.assertTrue(np.isfinite(clean_image(x)).all()); self.assertEqual(minmax_normalize(np.ones((2,2))).sum(),0)
        self.assertAlmostEqual(float(zscore_standardize(clean_image(x))[clean_image(x)!=0].mean()),0)
        train,val=patient_level_split(['a','b','c','d']); self.assertFalse(set(train)&set(val))
        mask=np.zeros((64,64));mask[20:40,20:40]=4;_,prepared,label=_prepare_slice(np.ones((64,64)),mask,(32,32));self.assertEqual(label,1);self.assertEqual(set(np.unique(prepared)),{0,1})


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.client=TestClient(app)

    def test_metadata_and_all_tutorial_routes(self):
        data=self.client.get('/api/tutorials').json()['tutorials']; self.assertEqual(len(data),15)
        self.assertTrue(all((x['id']<=6 or 'brain tumour' in x['title'].lower() or 'mri' in x['title'].lower()) for x in data))
        for tutorial in range(1,16):
            with self.subTest(tutorial=tutorial): self.assertEqual(self.client.post(f'/api/tutorials/{tutorial}/run',json={}).status_code,200)

    def test_csv_scope_and_validation(self):
        c=self.client
        self.assertEqual(c.get('/api/tutorials/7/example').status_code,404)
        self.assertEqual(c.post('/api/tutorials/7/upload',content=b'a',headers={'content-type':'text/csv'}).status_code,404)
        for tutorial in (10,11,12):
            example=c.get(f'/api/tutorials/{tutorial}/example'); self.assertEqual(example.status_code,200)
            response=c.post(f'/api/tutorials/{tutorial}/upload',content=example.content,headers={'content-type':'text/csv'}); self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(c.post('/api/tutorials/10/upload',content=b'bad',headers={'content-type':'application/octet-stream'}).status_code,415)
        self.assertEqual(c.post('/api/tutorials/10/run',json={'records':[]}).status_code,422)

    def test_frontend_and_missing_mri(self):
        for path in ['/','/image-analysis','/static/lab.js','/static/lab.css','/static/image-analysis.js']:
            self.assertEqual(self.client.get(path).status_code,200)
        from her_core.analytics import mri_case_features
        mri_case_features.cache_clear()
        with patch('her_core.analytics.discover_cases',return_value=[]): self.assertEqual(self.client.post('/api/tutorials/8/run',json={}).status_code,503)
        mri_case_features.cache_clear()

if __name__=='__main__': unittest.main(verbosity=2)
