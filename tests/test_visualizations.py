"""PNG/rendering regression checks for the integrated MRI tutorial workflow."""
import base64, json, unittest
from io import BytesIO
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app
from her_core.analytics import run_analysis

EXPECTED={
  1:['Original MRI middle slice','Available modality counts'],2:['Preprocessing'],3:['Missing MRI modalities'],
  4:['Image processing'],5:['SIFT + RANSAC correspondences'],6:['Same-patient multimodal MRI'],
  7:['MRI modality and segmentation availability','Case distribution by reference-label availability'],
  8:['MRI modality and segmentation availability','MRI case distribution by available modality count'],
  9:['MRI feature correlation heatmap'],10:['Synthetic text classification confusion matrix'],
  11:['Extracted brain-tumour report entities'],12:['Ranked report relevance (TF-IDF cosine similarity)'],
  13:['PCA projection of MRI-derived case features','PCA variance explained'],
  14:['Tumour burden versus FLAIR mean intensity','Imaging feature and tumour-burden correlations'],
  15:['Held-out predicted versus actual tumour fraction','MRI feature importance for tumour-burden model']}

class VisualizationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.client=TestClient(app);cls.results={}
  for tutorial in range(1,16):
   response=cls.client.post(f'/api/tutorials/{tutorial}/run',json={});
   if response.status_code!=200:raise AssertionError(f'T{tutorial}: {response.text}')
   cls.results[tutorial]=response.json()
  summary=[{'tutorial':i,'plots':[p['title'] for p in r['plots']],'tables':list(r['tables'])} for i,r in cls.results.items()]
  artifact=Path(__file__).resolve().parents[1]/'artifacts';artifact.mkdir(exist_ok=True);(artifact/'visualization_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

 def test_expected_mri_aligned_visuals_and_valid_pngs(self):
  for tutorial,result in self.results.items():
   with self.subTest(tutorial=tutorial):
    titles=[p['title'] for p in result['plots']];self.assertTrue(set(EXPECTED[tutorial])<=set(titles));self.assertEqual(len(titles),len(set(titles)))
    for plot in result['plots']:
     raw=base64.b64decode(plot['png'],validate=True);self.assertTrue(raw.startswith(b'\x89PNG\r\n\x1a\n'))
     with Image.open(BytesIO(raw)) as image:self.assertGreaterEqual(image.width,200);self.assertGreaterEqual(image.height,200);image.verify()

 def test_no_unrelated_metrics_or_unlabeled_synthetic_claims(self):
  for tutorial in (7,8,9,13,14,15):
   result=self.results[tutorial];structured=json.dumps({k:v for k,v in result.items() if k!='plots'}).lower();self.assertNotIn('admissions',structured);self.assertNotIn('bmi',structured);self.assertNotIn('staffing',structured)
  for tutorial in (10,11,12):self.assertIn('synthetic',self.results[tutorial]['dataset'].lower())

 def test_text_annotation_offsets(self):
  text='Glioma in frontal lobe with edema measures 12 mm.';result=run_analysis(11,text=text)
  for span,entity in zip(result['annotations']['spans'],result['tables']['entities']):self.assertEqual(text[span['start']:span['end']],entity['entity'])

if __name__=='__main__':unittest.main(verbosity=2)
