"""CNN response regression tests, including a small real labeled BraTS run."""
import base64
from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from sklearn.metrics import classification_report, confusion_matrix

from backend.main import app
from backend.tutorials import CNN_LOCK


class CNNOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=TestClient(app)
        cls.response=cls.client.post('/api/cnn/demo',json={})
        if cls.response.status_code!=200:
            raise AssertionError(cls.response.text)
        cls.output=cls.response.json()
        outdir=Path(__file__).resolve().parents[1]/'artifacts'
        outdir.mkdir(exist_ok=True)
        (outdir/'cnn_after.json').write_text(json.dumps(cls.output,indent=2),encoding='utf-8')

    def test_complete_serialized_response(self):
        self.assertEqual(int(self.response.headers['content-length']),len(self.response.content))
        json.dumps(self.output,allow_nan=False)
        self.assertEqual(self.output['workflow'],'training_then_validation_inference')
        self.assertEqual(set(self.output['tables']),{'predictions','split','training_history','classification_report','confusion_matrix'})
        self.assertEqual(len(self.output['plots']),4)
        self.assertEqual(self.output['model_task'],'classification')

    def test_individual_preview_images_match_real_model_inputs(self):
        from her_core.cnn import build_cnn_datasets
        data=build_cnn_datasets(target_size=(32,32),max_slices_per_case=8)
        previews=self.output['previews']
        self.assertEqual(len(previews),4)
        for preview in previews:
            rec=next(r for r in data['val_records'] if r.case_id==preview['case'] and r.slice_idx==preview['slice'])
            for key,array in [('input_image',rec.image),('ground_truth_mask',rec.mask)]:
                self.assertTrue(preview[key].startswith('data:image/png;base64,'))
                with Image.open(BytesIO(base64.b64decode(preview[key].split(',')[1],validate=True))) as img:
                    self.assertEqual(img.size,(32,32))
                    expected=np.clip(np.flipud(array.T)*255,0,255).astype(np.uint8)
                    np.testing.assert_array_equal(np.asarray(img),expected)
            self.assertNotIn('predicted_mask',preview)
            self.assertIn(preview['predicted_class'],['No Tumor','Tumor Present'])

    def test_png_conversion_and_plot_types(self):
        titles=[]
        for plot in self.output['plots']:
            png=base64.b64decode(plot['png'],validate=True)
            self.assertTrue(png.startswith(b'\x89PNG\r\n\x1a\n'))
            with Image.open(BytesIO(png)) as image:
                self.assertGreater(image.width,100)
                self.assertGreater(image.height,100)
                image.verify()
            titles.append(plot['title'].lower())
        self.assertTrue(any('accuracy' in title for title in titles))
        self.assertTrue(any('loss' in title for title in titles))
        self.assertTrue(any('confusion matrix' in title for title in titles))
        self.assertTrue(any('input' in title and 'mask' in title for title in titles))

    def test_actual_predictions_scores_and_split(self):
        rows=self.output['tables']['predictions']
        self.assertEqual(len(rows),self.output['metrics']['validation_slices'])
        self.assertGreater(len(rows),0)
        for row in rows:
            self.assertTrue(0<=row['probability']<=1)
            self.assertEqual(row['predicted'],int(row['probability']>=.5))
            self.assertAlmostEqual(row['probability']+row['no_tumor_probability'],1)
            self.assertAlmostEqual(row['predicted_class_score'],max(row['probability'],1-row['probability']))
            self.assertEqual(row['correct'],row['actual']==row['predicted'])
            self.assertEqual(row['predicted_class'],['No Tumor','Tumor Present'][row['predicted']])
            self.assertEqual(row['actual_class'],['No Tumor','Tumor Present'][row['actual']])
        split=self.output['tables']['split'][0]
        self.assertFalse(set(split['train_cases'].split(', ')) & set(split['validation_cases'].split(', ')))
        self.assertAlmostEqual(self.output['metrics']['accuracy'],np.mean([r['correct'] for r in rows]))

    def test_real_training_and_validation_history(self):
        history=self.output['tables']['training_history']
        self.assertEqual([row['epoch'] for row in history],[1,2,3])
        for row in history:
            self.assertTrue(0<=row['accuracy']<=1)
            self.assertTrue(0<=row['val_accuracy']<=1)
            self.assertTrue(np.isfinite(row['loss']) and row['loss']>=0)
            self.assertTrue(np.isfinite(row['val_loss']) and row['val_loss']>=0)
        self.assertAlmostEqual(history[-1]['val_accuracy'],self.output['metrics']['accuracy'],places=6)
        rows=self.output['tables']['predictions']
        actual=np.array([r['actual'] for r in rows])
        probabilities=np.clip([r['probability'] for r in rows],1e-7,1-1e-7)
        loss=-np.mean(actual*np.log(probabilities)+(1-actual)*np.log(1-probabilities))
        self.assertAlmostEqual(history[-1]['val_loss'],loss,places=5)

    def test_report_and_confusion_matrix_use_all_validation_labels(self):
        rows=self.output['tables']['predictions']
        actual=[r['actual'] for r in rows];predicted=[r['predicted'] for r in rows]
        expected=confusion_matrix(actual,predicted,labels=[0,1])
        received=np.array([[r['No Tumor'],r['Tumor Present']] for r in self.output['tables']['confusion_matrix']])
        np.testing.assert_array_equal(received,expected)
        self.assertEqual(int(received.sum()),len(rows))
        expected_report=classification_report(actual,predicted,labels=[0,1],target_names=['No Tumor','Tumor Present'],output_dict=True,zero_division=0)
        for row in self.output['tables']['classification_report']:
            for metric in ('precision','recall','f1-score','support'):
                self.assertAlmostEqual(row[metric],expected_report[row['class']][metric])

    def test_missing_labels_returns_error_and_releases_lock(self):
        with patch('her_core.cnn.build_cnn_datasets',side_effect=FileNotFoundError('No labeled cases')):
            response=self.client.post('/api/cnn/demo',json={})
        self.assertEqual(response.status_code,503)
        self.assertIn('No labeled cases',response.json()['detail'])
        self.assertFalse(CNN_LOCK.locked())

    def test_post_prediction_exception_returns_error_and_releases_lock(self):
        # Force a failure after prediction, in a newly added output-rendering step.
        with patch('her_core.analytics.figure_plot',side_effect=RuntimeError('PNG conversion failed')):
            response=self.client.post('/api/cnn/demo',json={})
        self.assertEqual(response.status_code,500)
        self.assertIn('CNN demonstration failed',response.json()['detail'])
        self.assertFalse(CNN_LOCK.locked())

    def test_concurrent_cnn_returns_conflict(self):
        with CNN_LOCK:
            response=self.client.post('/api/cnn/demo',json={})
        self.assertEqual(response.status_code,409)


if __name__=='__main__':unittest.main(verbosity=2)
