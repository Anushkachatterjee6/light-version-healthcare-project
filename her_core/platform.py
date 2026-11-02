"""Independent research services used by the main Brain MRI application.

These services intentionally use the local NIfTI inventory and reusable core
algorithms directly.  They do not call the Tutorial Lab router.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .io import discover_cases, load_case_modality
from .preprocess import clean_image, minmax_normalize, zscore_standardize


def _png(array: np.ndarray, *, binary: bool = False) -> str:
    pixels = np.asarray(array)
    if binary:
        pixels = (pixels > 0).astype(np.uint8) * 255
    else:
        pixels = (minmax_normalize(clean_image(pixels)) * 255).astype(np.uint8)
    ok, encoded = cv2.imencode('.png', pixels.T)
    if not ok:
        raise RuntimeError('PNG conversion failed')
    return 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')


def _plot(title: str, x, series: dict[str, Any], *, kind='line', xlabel='', ylabel='') -> dict:
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7, 3.8), layout='constrained')
    ax = fig.subplots()
    for label, values in series.items():
        if kind == 'bar':
            ax.bar(x, values, label=label)
        else:
            ax.plot(x, values, marker='o', label=label)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if len(series) > 1:
        ax.legend()
    ax.tick_params(axis='x', labelrotation=25)
    output = BytesIO(); fig.savefig(output, format='png', dpi=110)
    return {'title': title, 'png': base64.b64encode(output.getvalue()).decode('ascii')}


def case_inventory() -> list[dict]:
    rows = []
    for case in discover_cases():
        modalities = sorted(case['modalities'])
        preferred = 'flair' if 'flair' in modalities else modalities[0]
        volume, nii, _ = load_case_modality(case, preferred)
        rows.append({'case_id': case['case_id'], 'modalities': modalities,
                     'default_modality': preferred, 'shape': list(volume.shape),
                     'num_slices': int(volume.shape[2]), 'voxel_spacing': [float(v) for v in nii.header.get_zooms()[:3]],
                     'reference_mask_available': 'seg' in modalities,
                     'quality_flags': [] if np.isfinite(volume).all() and volume.ndim == 3 else ['non-finite values or invalid dimensionality']})
    return rows


def dataset_profile() -> dict:
    cases = case_inventory()
    modality_names = ['flair', 't1', 't1ce', 't2', 'seg']
    counts = {name: sum(name in row['modalities'] for row in cases) for name in modality_names}
    return {'cases': cases,
            'summary': {'total_cases': len(cases), 'cases_with_reference_masks': counts['seg'],
                        'modality_counts': counts, 'integrity_flags': sum(len(c['quality_flags']) for c in cases)},
            'plots': [_plot('Local MRI modality availability', list(counts), {'Cases': list(counts.values())}, kind='bar', xlabel='Asset', ylabel='Cases')],
            'limitations': 'Availability and integrity are derived from local NIfTI files. Missing modalities are reported, not imputed.'}


def radiomics(case_id: str, modality: str = 'flair') -> dict:
    case = next((c for c in discover_cases() if c['case_id'] == case_id), None)
    if not case or modality not in case['modalities']:
        raise ValueError('Requested case or modality is unavailable')
    volume, nii, _ = load_case_modality(case, modality)
    has_mask = 'seg' in case['modalities']
    mask = load_case_modality(case, 'seg')[0] > 0 if has_mask else np.ones(volume.shape, dtype=bool)
    values = clean_image(volume)[mask & np.isfinite(volume)]
    if not values.size:
        raise ValueError('No finite voxels are available in the selected region')
    scaled = np.clip(minmax_normalize(volume) * 31, 0, 31).astype(np.uint8)
    middle = scaled[:, :, scaled.shape[2] // 2]
    glcm = graycomatrix(middle, [1], [0], levels=32, symmetric=True, normed=True)
    features = {'mean_intensity': float(values.mean()), 'median_intensity': float(np.median(values)),
                'std_intensity': float(values.std()), 'p05_intensity': float(np.percentile(values, 5)),
                'p95_intensity': float(np.percentile(values, 95)),
                'entropy_bits': float(entropy(np.histogram(values, bins=64, density=True)[0] + 1e-12, base=2)),
                'energy': float(np.mean(values ** 2)), 'glcm_contrast_mid_slice': float(graycoprops(glcm, 'contrast')[0, 0]),
                'glcm_homogeneity_mid_slice': float(graycoprops(glcm, 'homogeneity')[0, 0])}
    return {'case_id': case_id, 'modality': modality, 'region': 'reference-mask region' if has_mask else 'whole image (no reference mask supplied)',
            'features': features, 'voxel_spacing': [float(v) for v in nii.header.get_zooms()[:3]],
            'provenance': 'Intensity and texture descriptors from supplied MRI voxels; not validated tumour biomarkers.'}


def quantify_reference_mask(case_id: str) -> dict:
    case = next((c for c in discover_cases() if c['case_id'] == case_id), None)
    if not case or 'seg' not in case['modalities']:
        raise ValueError('This case has no supplied reference mask')
    seg, nii, _ = load_case_modality(case, 'seg'); mask = seg > 0
    spacing = np.asarray(nii.header.get_zooms()[:3], dtype=float)
    per_slice = mask.sum(axis=(0, 1)).astype(int)
    voxel_volume = float(np.prod(spacing))
    labeled = int(mask.sum())
    return {'case_id': case_id, 'metrics': {'reference_mask_voxels': labeled,
            'reference_mask_volume_mm3': float(labeled * voxel_volume), 'reference_mask_volume_ml': float(labeled * voxel_volume / 1000),
            'mask_coverage_percent': float(mask.mean() * 100), 'slices_with_reference_region': int((per_slice > 0).sum())},
            'voxel_spacing': spacing.tolist(), 'slice_measurements': [{'slice_index': int(i), 'region_voxels': int(v), 'region_area_mm2': float(v * spacing[0] * spacing[1])} for i, v in enumerate(per_slice)],
            'plot': _plot('Reference-mask region distribution by slice', list(range(len(per_slice))), {'Region voxels': per_slice}, xlabel='Axial slice', ylabel='Reference-mask voxels'),
            'provenance': 'Measurements derive only from supplied reference segmentation; they are not CNN predictions.'}


def preprocess_preview(case_id: str, modality: str, slice_index: int) -> dict:
    case = next(c for c in discover_cases() if c['case_id'] == case_id)
    volume, _, _ = load_case_modality(case, modality); image = volume[:, :, slice_index]
    normalized = minmax_normalize(image)
    denoised = cv2.GaussianBlur(normalized, (3, 3), 0)
    standardized = zscore_standardize(image)
    return {'case_id': case_id, 'modality': modality, 'slice_index': slice_index,
            'images': {'input': _png(image), 'minmax_normalized': _png(normalized), 'denoised': _png(denoised), 'zscore_standardized_display': _png(standardized)},
            'parameters': {'denoise': 'GaussianBlur kernel 3x3', 'normalization': 'per-slice min-max', 'standardization': 'non-background z-score'}}


def discovery_features() -> dict:
    rows = []
    for item in case_inventory():
        if 'flair' not in item['modalities']:
            continue
        values = load_case_modality(next(c for c in discover_cases() if c['case_id'] == item['case_id']), 'flair')[0]
        finite = values[np.isfinite(values) & (values != 0)]
        rows.append({'case_id': item['case_id'], 'mean_intensity': float(finite.mean()), 'std_intensity': float(finite.std()), 'p95_intensity': float(np.percentile(finite, 95)), 'has_reference_mask': item['reference_mask_available']})
    return {'rows': rows, 'sample_count': len(rows), 'plots': [_plot('FLAIR mean intensity by local case', [r['case_id'] for r in rows], {'Mean intensity': [r['mean_intensity'] for r in rows]}, kind='bar', ylabel='Raw FLAIR intensity')], 'limitations': 'Exploratory descriptive analysis only; the small local cohort does not support diagnostic grouping claims.'}


def image_processing(case_id: str, modality: str, slice1: int, slice2: int) -> dict:
    """Direct application wrapper around the reusable SIFT/RANSAC core."""
    from .sift_ransac import analyze_mri_sift_ransac
    case = next((c for c in discover_cases() if c['case_id'] == case_id), None)
    if not case or modality not in case['modalities']:
        raise ValueError('Requested case or modality is unavailable')
    volume, _, _ = load_case_modality(case, modality)
    if not 0 <= slice1 < volume.shape[2] or not 0 <= slice2 < volume.shape[2] or slice1 == slice2:
        raise ValueError('Choose two distinct valid slice indices')
    result = analyze_mri_sift_ransac(volume[:, :, slice1], volume[:, :, slice2], return_base64=True)
    return {'case_id': case_id, 'modality': modality, 'slice1': slice1, 'slice2': slice2,
            'metrics': {key: result[key] for key in ('keypoints_image1', 'keypoints_image2', 'good_matches', 'ransac_inliers', 'ransac_outliers', 'inlier_ratio', 'homography_found', 'success', 'message')},
            'images': {key: result[f'{key}_data_url'] for key in ('original_image1', 'original_image2', 'sift_visualization_1', 'sift_visualization_2', 'matches_visualization', 'ransac_visualization')},
            'provenance': 'SIFT keypoints and RANSAC transformation are computed from the selected local MRI slices.'}


_SYNTHETIC_REPORTS = [
    ('glioma-related', 'Synthetic teaching report: infiltrative glioma with surrounding edema and enhancement.'),
    ('glioma-related', 'Synthetic teaching report: intra-axial lesion with edema and mass effect.'),
    ('meningioma-related', 'Synthetic teaching report: extra-axial enhancing meningioma along the falx.'),
    ('meningioma-related', 'Synthetic teaching report: dural based lesion with homogeneous enhancement.'),
    ('no-tumour-finding', 'Synthetic teaching report: no focal mass or abnormal enhancement identified.'),
    ('no-tumour-finding', 'Synthetic teaching report: no evidence of intracranial tumour or mass effect.')]


def clinical_nlp(workflow: str, text: str = '', query: str = '') -> dict:
    """Clearly labeled, separate synthetic-text workflows for product demonstration."""
    labels, documents = zip(*_SYNTHETIC_REPORTS)
    vectorizer = TfidfVectorizer(stop_words='english'); matrix = vectorizer.fit_transform(documents)
    if workflow == 'classify':
        if not text.strip(): raise ValueError('Enter text for synthetic-corpus classification')
        model = LogisticRegression(max_iter=300, random_state=42).fit(matrix, labels)
        probabilities = model.predict_proba(vectorizer.transform([text]))[0]
        return {'workflow': workflow, 'input_text': text, 'prediction': model.classes_[int(probabilities.argmax())],
                'scores': [{'class': c, 'probability': float(p)} for c, p in zip(model.classes_, probabilities)],
                'limitations': 'Training corpus is synthetic teaching text; this is not clinical NLP validation.'}
    if workflow == 'cluster':
        km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(matrix)
        terms = vectorizer.get_feature_names_out()
        return {'workflow': workflow, 'rows': [{'document': d, 'synthetic_label': l, 'cluster': int(k)} for d,l,k in zip(documents,labels,km.labels_)],
                'clusters': [{'cluster': int(i), 'top_terms': ', '.join(terms[np.argsort(center)[-4:][::-1]])} for i,center in enumerate(km.cluster_centers_)],
                'limitations': 'Clusters describe a six-document synthetic teaching corpus, not patient populations.'}
    if workflow == 'extract':
        source = text.strip() or documents[0]; lexicon = {'finding':['edema','enhancement','mass effect','lesion'], 'tumour_type':['glioma','meningioma','tumour'], 'anatomy':['falx','intracranial']}
        entities = [{'entity': term, 'category': category, 'start': source.lower().find(term)} for category,terms in lexicon.items() for term in terms if term in source.lower()]
        return {'workflow': workflow, 'input_text': source, 'entities': entities, 'limitations': 'Transparent keyword extraction, not validated named-entity recognition.'}
    if workflow == 'search':
        if not query.strip(): raise ValueError('Enter a search query')
        scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel(); order = np.argsort(-scores)
        return {'workflow': workflow, 'query': query, 'results': [{'rank': int(i+1), 'document': documents[j], 'synthetic_label': labels[j], 'score': float(scores[j])} for i,j in enumerate(order)], 'limitations': 'Search ranks synthetic teaching text with TF-IDF cosine similarity.'}
    raise ValueError('Unknown clinical NLP workflow')


def analytics_report() -> dict:
    data = discovery_features(); frame = pd.DataFrame(data['rows'])
    columns = ['mean_intensity','std_intensity','p95_intensity']
    x = StandardScaler().fit_transform(frame[columns]); pca = PCA(n_components=min(2, len(columns), len(frame))).fit(x)
    components = pca.transform(x)
    corr = frame[columns].corr().round(4).to_dict()
    pca_rows = [{'case_id': frame.case_id.iloc[i], 'PC1': float(components[i,0]), 'PC2': float(components[i,1]) if components.shape[1] > 1 else 0.0} for i in range(len(frame))]
    data.update({'correlation': corr, 'pca': pca_rows, 'explained_variance_ratio': [float(v) for v in pca.explained_variance_ratio_],
                 'plots': data['plots'] + [_plot('PCA explained variance', [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))], {'Explained variance': pca.explained_variance_ratio_}, kind='bar', ylabel='Proportion')]})
    return data


def _store_path() -> Path:
    path = Path(__file__).resolve().parent.parent / 'app_data' / 'experiments.json'; path.parent.mkdir(exist_ok=True); return path


def list_experiments() -> list[dict]:
    path = _store_path()
    if not path.exists(): return []
    try: return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError: return []


def save_experiment(experiment: dict) -> dict:
    records = list_experiments(); record = {'id': uuid4().hex, 'timestamp': datetime.now(timezone.utc).isoformat(), **experiment}
    records.insert(0, record); _store_path().write_text(json.dumps(records, indent=2, default=str), encoding='utf-8'); return record


def get_experiment(experiment_id: str) -> dict:
    for item in list_experiments():
        if item['id'] == experiment_id: return item
    raise ValueError('Experiment not found')
