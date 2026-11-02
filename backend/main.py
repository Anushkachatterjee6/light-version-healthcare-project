"""
HER Biomedical Image Tutorials — FastAPI backend.

Serves the existing tutorial UI and the T5 Image Analysis (SIFT + RANSAC) API.
"""

from __future__ import annotations

import sys
import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from her_core.io import (  # noqa: E402
    default_data_root,
    discover_cases,
    get_slice,
    list_modalities,
    load_case_modality,
)
from her_core.sift_ransac import analyze_mri_sift_ransac  # noqa: E402
from her_core.preprocess import clean_image, minmax_normalize  # noqa: E402
from her_core.platform import (  # noqa: E402
    analytics_report, clinical_nlp, dataset_profile, discovery_features, get_experiment,
    image_processing, list_experiments, preprocess_preview, quantify_reference_mask,
    radiomics, save_experiment,
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="Brain Tumor Detection Using MRI Images",
    description="T1–T15: MRI processing, operational analytics, clinical text and exploratory research",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory volume cache so repeated analysis clicks stay responsive.
_VOLUME_CACHE: dict[tuple[str, str], Any] = {}
_UPLOAD_MODEL: Any = None


class SiftRansacRequest(BaseModel):
    case_id: str = Field(..., description="Patient/case folder name")
    modality: str = Field("flair", description="MRI modality, e.g. flair, t1, t2")
    slice1: int = Field(..., ge=0, description="Index of first axial slice")
    slice2: int = Field(..., ge=0, description="Index of second axial slice")
    data_root: str | None = Field(
        None, description="Optional override for BraTS-style dataset root"
    )


class ExperimentRequest(BaseModel):
    experiment_type: str = Field(..., max_length=80)
    configuration: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)
    limitations: str = Field('', max_length=2000)


class UploadAnalysisReportRequest(BaseModel):
    """The latest client-held upload-analysis response; uploaded image bytes are never resent."""
    analysis: dict[str, Any]


def _get_case(case_id: str, data_root: str | None = None) -> dict:
    _validate_data_root(data_root)
    cases = discover_cases(data_root or default_data_root())
    for case in cases:
        if case["case_id"] == case_id:
            return case
    raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")


def _validate_data_root(data_root):
    if data_root is not None and Path(data_root).resolve() != default_data_root().resolve():
        raise HTTPException(400, "API dataset root must be the project's BraTS2020 folder")


def _load_volume_cached(case: dict, modality: str):
    key = (str(case['path'].resolve()), modality.lower())
    if key not in _VOLUME_CACHE:
        if len(_VOLUME_CACHE) >= 4:
            _VOLUME_CACHE.pop(next(iter(_VOLUME_CACHE)))
        volume, _, path = load_case_modality(case, modality.lower())
        _VOLUME_CACHE[key] = {"volume": volume, "path": str(path)}
    return _VOLUME_CACHE[key]


@app.get("/api/health")
def health():
    root = default_data_root()
    cases = discover_cases(root)
    return {
        "status": "ok",
        "data_root": str(root),
        "cases_found": len(cases),
    }


@app.get("/api/application-summary")
def application_summary():
    """Live, non-persistent dashboard facts derived from the supplied MRI inventory."""
    from her_core.analytics import mri_case_features
    features = mri_case_features()
    modality_columns = [column for column in features if column.startswith('has_') and column != 'has_seg']
    return {
        'total_cases': int(len(features)),
        'available_modalities': int(sum(int(features[column].sum()) > 0 for column in modality_columns)),
        'cases_with_segmentation': int(features.has_seg.sum()),
        'processed_scans': int(features.has_flair.sum()),
        'dataset_completeness_percent': float(features[modality_columns + ['has_seg']].to_numpy().mean() * 100),
        'available_models': ['SIFT + RANSAC MRI feature matching', 'Compact CNN slice classifier'],
        'recent_analyses': [],
        'limitations': 'No persistent analysis history or verified tumour-negative cohort is supplied locally.',
    }


@app.get('/api/app/dataset/cases')
def app_dataset_cases():
    return api_cases()


@app.get('/api/app/dataset/summary')
def app_dataset_summary():
    return application_summary()


@app.get('/api/app/dataset/profile')
def app_dataset_profile():
    """Application-level inventory and quality profile; no tutorial adapter."""
    return dataset_profile()


@app.get('/api/app/dataset/cases/{case_id}')
def app_dataset_case(case_id: str):
    case = _get_case(case_id)
    modalities = list_modalities(case)
    preferred = 'flair' if 'flair' in modalities else modalities[0]
    volume = _load_volume_cached(case, preferred)['volume']
    return {'case_id': case_id, 'modalities': modalities, 'reference_mask_available': 'seg' in modalities,
            'default_modality': preferred, 'shape': list(volume.shape), 'num_slices': int(volume.shape[2])}


@app.get('/api/app/mri/cases/{case_id}/slice')
def app_mri_slice(case_id: str, modality: str = 'flair', slice_index: int = 0):
    return api_case_slice(case_id, modality, slice_index)


@app.get('/api/app/mri/cases/{case_id}/metadata')
def app_mri_metadata(case_id: str, modality: str = 'flair'):
    return api_case_meta(case_id, modality)


@app.get('/api/app/radiomics/{case_id}')
def app_radiomics(case_id: str, modality: str = 'flair'):
    try:
        return radiomics(case_id, modality)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get('/api/app/quantification/{case_id}')
def app_quantification(case_id: str):
    try:
        return quantify_reference_mask(case_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get('/api/app/preprocessing/{case_id}')
def app_preprocessing(case_id: str, modality: str = 'flair', slice_index: int = 0):
    try:
        case = _get_case(case_id)
        if modality not in case['modalities']:
            raise HTTPException(404, f"Modality '{modality}' is unavailable")
        volume = _load_volume_cached(case, modality)['volume']
        if not 0 <= slice_index < volume.shape[2]:
            raise HTTPException(400, f"slice_index must be between 0 and {volume.shape[2] - 1}")
        return preprocess_preview(case_id, modality, slice_index)
    except HTTPException:
        raise
    except (ValueError, StopIteration) as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get('/api/app/research/discovery')
def app_research_discovery():
    return analytics_report()


@app.post('/api/app/upload-analysis')
async def app_upload_analysis(file: UploadFile = File(...), modality: str = Form(...)):
    """Saved-checkpoint inference for researcher-supplied FLAIR only; uploads are not retained."""
    import cv2, json, tempfile, time
    import numpy as np
    from her_core.cnn import preprocess_inference_slice
    if modality.lower() != 'flair': raise HTTPException(422, 'This saved model was trained only on FLAIR. Select FLAIR or do not submit the file.')
    name=(file.filename or '').lower()
    if not name.endswith(('.png','.jpg','.jpeg','.nii','.nii.gz')): raise HTTPException(415,'Supported formats: PNG, JPEG, NIfTI (.nii, .nii.gz).')
    blob=await file.read()
    if not blob or len(blob)>30*1024*1024: raise HTTPException(413,'Upload must be nonempty and at most 30 MB.')
    def data_url(a):
        pixels=(minmax_normalize(clean_image(a))*255).astype('uint8');ok,p=cv2.imencode('.png',pixels.T)
        if not ok: raise HTTPException(500,'Preview encoding failed')
        return 'data:image/png;base64,'+base64.b64encode(p).decode('ascii')
    try:
        if name.endswith(('.png','.jpg','.jpeg')):
            raw=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_UNCHANGED)
            if raw is None: raise ValueError('Upload bytes could not be decoded as an image.')
            decoded_shape=list(raw.shape); decoded_dtype=str(raw.dtype)
            if raw.ndim==2:
                image=raw; decoded_mode='L'
            elif raw.ndim==3 and raw.shape[2]==1:
                image=raw[:,:,0]; decoded_mode='L'
            elif raw.ndim==3 and raw.shape[2]==3:
                image=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY); decoded_mode='RGB/BGR converted to grayscale'
            elif raw.ndim==3 and raw.shape[2]==4:
                # Composite transparent pixels over black before grayscale conversion.
                alpha_max=np.iinfo(raw.dtype).max if np.issubdtype(raw.dtype,np.integer) else 1.0
                alpha=raw[:,:,3:4].astype(np.float32)/alpha_max
                image=cv2.cvtColor((raw[:,:,:3].astype(np.float32)*alpha).astype(raw.dtype),cv2.COLOR_BGR2GRAY); decoded_mode='LA/RGBA/BGRA alpha-composited then grayscale'
            else: raise ValueError(f'Unsupported decoded image layout {decoded_shape}; use L, LA, RGB, or RGBA PNG/JPEG.')
            volume=image[...,None];kind='2-D image'
        else:
            import nibabel as nib
            suffix='.nii.gz' if name.endswith('.nii.gz') else '.nii';tmp=tempfile.NamedTemporaryFile(suffix=suffix,delete=False);tmp.write(blob);tmp.close()
            try: volume=np.asarray(nib.load(tmp.name).get_fdata(),dtype=np.float32)
            finally: Path(tmp.name).unlink(missing_ok=True)
            if volume.ndim!=3: raise ValueError('NIfTI upload must contain one 3-D FLAIR volume.')
            kind='3-D NIfTI volume'
        if not np.isfinite(volume).any(): raise ValueError('Image contains no finite intensities.')
        global _UPLOAD_MODEL
        checkpoint=PROJECT_ROOT/'models'/'brain_tumor_cnn.keras'; metadata_path=PROJECT_ROOT/'models'/'model_metadata.json'
        if not checkpoint.exists() or not metadata_path.exists(): raise FileNotFoundError('Persistent checkpoint or metadata is missing. Run scripts/train_brain_tumor_cnn.py.')
        if _UPLOAD_MODEL is None:
            import tensorflow as tf
            _UPLOAD_MODEL=tf.keras.models.load_model(checkpoint)
        start=time.perf_counter(); eligible=[]
        for z in range(volume.shape[2]):
            raw=volume[:,:,z]
            if float(np.mean(clean_image(raw)>0))>=.02: eligible.append((z,preprocess_inference_slice(raw)))
        if not eligible: raise ValueError('No eligible non-background MRI slices were found.')
        batch=np.stack([x[1] for x in eligible]); probs=_UPLOAD_MODEL.predict(batch,verbose=0).ravel();mid=len(eligible)//2;z,processed=eligible[mid]
        meta=json.loads(metadata_path.read_text())
        input_info={'filename':Path(file.filename or 'upload').name,'kind':kind,'modality':'FLAIR','shape':list(volume.shape),'eligible_slices':len(eligible)}
        if kind=='2-D image': input_info.update(decoded_shape=decoded_shape,decoded_dtype=decoded_dtype,decoded_mode=decoded_mode)
        return {'model_version':meta['model_version'],'threshold':0.5,'input':input_info,'preview':{'slice_index':int(z),'original':data_url(volume[:,:,z]),'preprocessed':data_url(processed[:,:,0])},'predictions':[{'slice_index':int(i),'tumor_probability':float(p),'prediction':'Tumor Present' if p>=.5 else 'No Tumor'} for (i,_),p in zip(eligible,probs)],'duration_ms':round((time.perf_counter()-start)*1000,1),'preprocessing':meta['preprocessing'],'limitations':'Experimental research-only slice classifier. One held-out patient was used; negative predictions do not rule out tumour and this is not a patient-level diagnosis.'}
    except HTTPException: raise
    except (ValueError,FileNotFoundError) as exc: raise HTTPException(422,str(exc)) from exc
    except Exception as exc: raise HTTPException(500,'Upload inference failed; verify the FLAIR file is compatible.') from exc


@app.post('/api/app/upload-analysis/report.pdf')
def app_upload_analysis_pdf_report(payload: UploadAnalysisReportRequest):
    """Render a downloadable research PDF from the latest real upload-analysis response."""
    import base64
    import datetime as dt
    import re
    from html import escape
    from PIL import Image as PilImage
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PdfImage, PageBreak, KeepTogether
    from reportlab.graphics.shapes import Drawing, Line, Rect, String, Circle

    analysis = payload.analysis if isinstance(payload.analysis, dict) else {}
    if not analysis:
        raise HTTPException(422, 'A successful upload-analysis response is required to create a PDF report.')
    input_info = analysis.get('input') if isinstance(analysis.get('input'), dict) else {}
    preview = analysis.get('preview') if isinstance(analysis.get('preview'), dict) else {}
    predictions = analysis.get('predictions') if isinstance(analysis.get('predictions'), list) else []
    preprocessing = analysis.get('preprocessing') if isinstance(analysis.get('preprocessing'), list) else []
    timestamp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    def text(value: Any) -> str:
        return escape(str(value if value not in (None, '') else 'Not available'))

    def value(value: Any) -> str:
        return str(value if value not in (None, '') else 'Not available')

    def probability(value: Any) -> str:
        try:
            return f'{float(value) * 100:.1f}%'
        except (TypeError, ValueError):
            return 'Not available'

    def data_url_image(value: Any, caption: str):
        if not isinstance(value, str) or not value.startswith('data:image/') or ',' not in value:
            return [Paragraph(f'{caption}: not returned by inference.', styles['BodyText'])]
        try:
            encoded = value.split(',', 1)[1]
            raw = base64.b64decode(encoded, validate=True)
            with PilImage.open(BytesIO(raw)) as image:
                width, height = image.size
            display_width = 7.8 * cm
            display_height = min(10.5 * cm, display_width * height / max(width, 1))
            flowable = PdfImage(BytesIO(raw), width=display_width, height=display_height)
            return [flowable, Paragraph(caption, styles['Caption'])]
        except Exception:
            return [Paragraph(f'{caption}: preview could not be embedded.', styles['BodyText'])]

    def probability_chart(rows: list[dict]):
        points = []
        for index, item in enumerate(rows):
            try:
                points.append((float(item.get('slice_index', index)), max(0.0, min(1.0, float(item.get('tumor_probability'))))))
            except (AttributeError, TypeError, ValueError):
                continue
        if not points:
            return None
        width, height, left, bottom = 470, 190, 48, 35
        drawing = Drawing(width, height)
        chart_width, chart_height = width-left-18, height-bottom-38
        drawing.add(Rect(left, bottom, chart_width, chart_height, strokeColor=colors.HexColor('#b7d4cc'), fillColor=colors.white))
        drawing.add(String(left, height-12, 'Model-estimated tumour probability by slice (not diagnostic certainty)', fontSize=8, fillColor=colors.HexColor('#244d46')))
        for tick in (0.0, 0.5, 1.0):
            y = bottom + tick * chart_height
            drawing.add(Line(left, y, left+chart_width, y, strokeColor=colors.HexColor('#d8e6e2'), strokeWidth=.45))
            drawing.add(String(8, y-3, f'{tick:.1f}', fontSize=7))
        threshold_y = bottom + .5 * chart_height
        threshold = Line(left, threshold_y, left+chart_width, threshold_y, strokeColor=colors.HexColor('#bc6f00'), strokeWidth=1)
        threshold.strokeDashArray = [3, 2]
        drawing.add(threshold)
        drawing.add(String(left+chart_width-100, threshold_y+4, 'Threshold 0.5', fontSize=7, fillColor=colors.HexColor('#8a5300')))
        low, high = points[0][0], points[-1][0]
        span = max(high-low, 1)
        if len(points) == 1:
            _, actual = points[0]
            bar_width = chart_width * .22
            bar_x = left + (chart_width-bar_width)/2
            drawing.add(Rect(bar_x, bottom, bar_width, actual*chart_height, strokeColor=None, fillColor=colors.HexColor('#21856e')))
            drawing.add(String(bar_x, bottom+actual*chart_height+5, f'Actual probability {actual*100:.1f}%', fontSize=8, fillColor=colors.HexColor('#145f50')))
            drawing.add(String(bar_x, 13, f'Slice index {int(low)}', fontSize=7))
            drawing.add(String(left+chart_width/2-24, 2, 'Slice index', fontSize=7))
            drawing.add(String(2, bottom+chart_height/2, 'Probability', fontSize=7, fillColor=colors.HexColor('#54716e')))
            return drawing
        previous = None
        for slice_index, value in points:
            x = left + (slice_index-low)/span*chart_width
            y = bottom + value*chart_height
            if previous:
                drawing.add(Line(previous[0], previous[1], x, y, strokeColor=colors.HexColor('#21856e'), strokeWidth=1.2))
            drawing.add(Circle(x, y, 1.3, strokeColor=colors.HexColor('#21856e'), fillColor=colors.HexColor('#21856e')))
            previous = (x, y)
        drawing.add(String(left, 13, f'Slice index {int(low)}', fontSize=7))
        drawing.add(String(left+chart_width-62, 13, f'Slice index {int(high)}', fontSize=7))
        drawing.add(String(left+chart_width/2-24, 2, 'Slice index', fontSize=7))
        drawing.add(String(2, bottom+chart_height/2, 'Probability', fontSize=7, fillColor=colors.HexColor('#54716e')))
        return drawing

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CoverTitle', parent=styles['Title'], textColor=colors.HexColor('#145f50'), fontSize=24, leading=29, spaceAfter=10))
    styles.add(ParagraphStyle(name='ResearchOnly', parent=styles['BodyText'], textColor=colors.HexColor('#7a4d00'), backColor=colors.HexColor('#fff1cf'), borderColor=colors.HexColor('#e2bd69'), borderWidth=0.5, borderPadding=8, spaceAfter=14))
    styles.add(ParagraphStyle(name='Caption', parent=styles['BodyText'], fontSize=8, textColor=colors.HexColor('#54716e'), alignment=1, spaceAfter=8))
    styles.add(ParagraphStyle(name='TableLabel', parent=styles['BodyText'], fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor('#18333c')))
    styles.add(ParagraphStyle(name='TableValue', parent=styles['BodyText'], fontSize=8.5, leading=11, textColor=colors.HexColor('#18333c')))
    styles.add(ParagraphStyle(name='TableHeader', parent=styles['BodyText'], fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.white))
    def table_cell(content: Any, label: bool = False):
        return Paragraph(text(content), styles['TableLabel' if label else 'TableValue'])
    document_buffer = BytesIO()
    document = SimpleDocTemplate(document_buffer, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.7*cm, title='MRI Upload & AI Analysis Report')
    story = [Paragraph('BRAIN + MRI RESEARCH PLATFORM', styles['Heading3']), Paragraph('MRI Upload &amp; AI Analysis Report', styles['CoverTitle']), Paragraph(f'Generated: {timestamp}', styles['BodyText']), Spacer(1, 10), Paragraph('RESEARCH-ONLY OUTPUT — Experimental slice-level classifier; this document is not a clinical diagnosis or patient-level assessment.', styles['ResearchOnly'])]
    story += [Paragraph('Input information', styles['Heading2'])]
    dimensions = input_info.get('shape')
    dimension_text = ' × '.join(map(str, dimensions)) if isinstance(dimensions, list) else 'Not available'
    input_rows = [
        [table_cell('Uploaded filename', True), table_cell(input_info.get('filename'))], [table_cell('Input modality', True), table_cell(input_info.get('modality'))],
        [table_cell('Input dimensions', True), table_cell(dimension_text)], [table_cell('Eligible slices', True), table_cell(input_info.get('eligible_slices'))],
        [table_cell('Input format', True), table_cell(input_info.get('kind'))],
    ]
    input_table = Table(input_rows, colWidths=[4.7*cm, 12.3*cm]); input_table.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),colors.HexColor('#edf7f4')),('GRID',(0,0),(-1,-1),0.25,colors.HexColor('#c6ddd7')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)])); story += [input_table, Spacer(1, 14)]
    story += [Paragraph('Model and methodology', styles['Heading2'])]
    model_rows = [[table_cell('Model version', True), table_cell(analysis.get('model_version'))], [table_cell('Classification threshold', True), table_cell(analysis.get('threshold'))], [table_cell('Inference duration', True), table_cell(f"{analysis.get('duration_ms')} ms" if analysis.get('duration_ms') is not None else None)], [table_cell('Preprocessing methodology', True), table_cell(' → '.join(map(str, preprocessing)) if preprocessing else None)]]
    model_table = Table(model_rows, colWidths=[4.7*cm, 12.3*cm]); model_table.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),colors.HexColor('#edf7f4')),('GRID',(0,0),(-1,-1),0.25,colors.HexColor('#c6ddd7')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)])); story += [model_table, Spacer(1, 14)]
    visualization_section = [Paragraph('MRI visualizations', styles['Heading2'])]
    original = data_url_image(preview.get('original'), 'Original MRI preview')
    processed = data_url_image(preview.get('preprocessed'), 'Preprocessed CNN input')
    visualization_section += [Table([[original, processed]], colWidths=[8.5*cm, 8.5*cm], hAlign='CENTER'), Spacer(1, 14)]
    story += [KeepTogether(visualization_section)]
    representative = next((row for row in predictions if isinstance(row, dict) and row.get('slice_index') == preview.get('slice_index')), predictions[0] if predictions else {})
    interpretation = Paragraph(f"<b>Slice-level result:</b> {text(representative.get('prediction') if isinstance(representative, dict) else None)} · <b>Model-estimated probability:</b> {probability(representative.get('tumor_probability') if isinstance(representative, dict) else None)} · <b>Classification threshold:</b> {text(analysis.get('threshold'))}. This is not diagnostic certainty; one 2-D slice does not establish a patient-level diagnosis.", styles['BodyText'])
    chart = probability_chart(predictions)
    prediction_intro = [Paragraph('Prediction results', styles['Heading2']), interpretation]
    if chart: prediction_intro += [Spacer(1, 8), chart]
    story += [PageBreak(), KeepTogether(prediction_intro), Spacer(1, 8)]
    prediction_rows = [[Paragraph('Slice index', styles['TableHeader']), Paragraph('Tumour probability', styles['TableHeader']), Paragraph('Predicted class', styles['TableHeader'])]]
    for row in predictions:
        if isinstance(row, dict): prediction_rows.append([table_cell(row.get('slice_index')), table_cell(probability(row.get('tumor_probability'))), table_cell(row.get('prediction'))])
    if len(prediction_rows) == 1: prediction_rows.append(['Not available', 'Not available', 'No predictions returned'])
    prediction_table = Table(prediction_rows, repeatRows=1, colWidths=[3.2*cm, 5*cm, 8.8*cm])
    prediction_table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#165e50')),('GRID',(0,0),(-1,-1),0.25,colors.HexColor('#c6ddd7')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f9f7')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)])); story += [prediction_table, Spacer(1, 14)]
    story += [Paragraph('Research limitations', styles['Heading2']), Paragraph('This experimental model produces slice-level outputs only. Model-estimated probabilities are not diagnostic certainty, the model is not clinically validated, and this report does not establish a patient-level diagnosis. Only one held-out patient was used for the known evaluation result.', styles['ResearchOnly']), Paragraph(text(analysis.get('limitations')), styles['BodyText'])]

    def footer(canvas, doc):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#c6ddd7')); canvas.line(1.5*cm, 1.2*cm, A4[0]-1.5*cm, 1.2*cm); canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#54716e')); canvas.drawString(1.5*cm, .8*cm, 'Brain + MRI Research Platform · Research-only'); canvas.drawRightString(A4[0]-1.5*cm, .8*cm, f'Page {doc.page}'); canvas.restoreState()
    try:
        document.build(story, onFirstPage=footer, onLaterPages=footer)
    except Exception as exc:
        raise HTTPException(500, 'PDF report generation failed.') from exc
    stem = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(str(input_info.get('filename') or 'mri_upload')).stem).strip('._') or 'mri_upload'
    filename = f'{stem}_research_report_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    return Response(content=document_buffer.getvalue(), media_type='application/pdf', headers={'Content-Disposition': f'attachment; filename="{filename}"'})


@app.post('/api/app/upload-preview')
async def app_upload_preview(file: UploadFile = File(...), slice_index: int = Form(0)):
    """Temporary NIfTI preview only; never loads the CNN or runs inference."""
    import cv2, tempfile, numpy as np
    name=(file.filename or '').lower(); blob=await file.read()
    if not name.endswith(('.nii','.nii.gz')): raise HTTPException(415,'Preview is available for NIfTI volumes only.')
    if not blob or len(blob)>30*1024*1024: raise HTTPException(413,'Upload must be nonempty and at most 30 MB.')
    try:
        import nibabel as nib
        suffix='.nii.gz' if name.endswith('.nii.gz') else '.nii';tmp=tempfile.NamedTemporaryFile(suffix=suffix,delete=False);tmp.write(blob);tmp.close()
        try: volume=np.asarray(nib.load(tmp.name).get_fdata(),dtype=np.float32)
        finally: Path(tmp.name).unlink(missing_ok=True)
        if volume.ndim!=3 or not 0<=slice_index<volume.shape[2]: raise ValueError('Choose a valid slice in a 3-D NIfTI volume.')
        pixels=(minmax_normalize(clean_image(volume[:,:,slice_index]))*255).astype('uint8');ok,png=cv2.imencode('.png',pixels.T)
        if not ok: raise ValueError('Preview encoding failed')
        return {'shape':list(volume.shape),'slice_index':slice_index,'num_slices':int(volume.shape[2]),'image':'data:image/png;base64,'+base64.b64encode(png).decode('ascii'),'metadata':'Temporary local preview; no model inference was run.'}
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    except Exception as exc: raise HTTPException(422,'Invalid or corrupted NIfTI upload.') from exc


@app.get('/api/app/image-processing/{case_id}')
def app_image_processing(case_id: str, modality: str = 'flair', slice1: int = 60, slice2: int = 64):
    try: return image_processing(case_id, modality, slice1, slice2)
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc


@app.get('/api/app/clinical-nlp/{workflow}')
def app_clinical_nlp(workflow: str, text: str = '', query: str = ''):
    try: return clinical_nlp(workflow, text, query)
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc


@app.get('/api/app/experiments')
def app_experiments(): return {'experiments': list_experiments()}


@app.post('/api/app/experiments')
def app_save_experiment(body: ExperimentRequest): return save_experiment(body.model_dump())


@app.get('/api/app/experiments/{experiment_id}')
def app_get_experiment(experiment_id: str):
    try: return get_experiment(experiment_id)
    except ValueError as exc: raise HTTPException(404, str(exc)) from exc


@app.get('/api/app/reports/{experiment_id}')
def app_report(experiment_id: str):
    record = app_get_experiment(experiment_id)
    html = '<!doctype html><title>Brain MRI Research Report</title><h1>Brain MRI Research Report</h1>' + ''.join(f'<h2>{key.replace("_", " ").title()}</h2><pre>{value}</pre>' for key,value in record.items())
    from fastapi.responses import HTMLResponse
    return HTMLResponse(html, headers={'Content-Disposition': f'attachment; filename="experiment-{experiment_id}.html"'})


@app.get("/api/cases")
def api_cases(data_root: str | None = None):
    _validate_data_root(data_root)
    cases = discover_cases(data_root or default_data_root())
    payload = []
    for case in cases:
        modalities = list_modalities(case)
        depth = None
        # Prefer flair for default depth metadata.
        preferred = "flair" if "flair" in modalities else (modalities[0] if modalities else None)
        if preferred:
            try:
                cached = _load_volume_cached(case, preferred)
                depth = int(cached["volume"].shape[2])
            except Exception:
                depth = None
        payload.append(
            {
                "case_id": case["case_id"],
                "modalities": modalities,
                "default_modality": preferred,
                "num_slices": depth,
            }
        )
    return {"cases": payload, "data_root": str(data_root or default_data_root())}


@app.get("/api/cases/{case_id}/meta")
def api_case_meta(case_id: str, modality: str = "flair", data_root: str | None = None):
    case = _get_case(case_id, data_root)
    modalities = list_modalities(case)
    if modality not in modalities:
        raise HTTPException(
            status_code=404,
            detail=f"Modality '{modality}' not available. Options: {modalities}",
        )
    cached = _load_volume_cached(case, modality)
    vol = cached["volume"]
    return {
        "case_id": case_id,
        "modality": modality,
        "shape": list(vol.shape),
        "num_slices": int(vol.shape[2]),
        "modalities": modalities,
        "path": cached["path"],
    }


@app.get("/api/cases/{case_id}/slice")
def api_case_slice(case_id: str, modality: str = "flair", slice_index: int = 0):
    """Compact PNG previews for the local viewer; reference masks remain distinct."""
    import cv2
    case = _get_case(case_id)
    if modality not in case['modalities']:
        raise HTTPException(404, f"Modality '{modality}' not available. Options: {sorted(case['modalities'])}")
    cached = _load_volume_cached(case, modality)
    volume = cached['volume']
    if slice_index < 0 or slice_index >= volume.shape[2]:
        raise HTTPException(400, f"slice_index must be between 0 and {volume.shape[2]-1}")
    def png(array):
        pixels = (minmax_normalize(clean_image(array)) * 255).astype('uint8')
        ok, encoded = cv2.imencode('.png', pixels.T)
        if not ok: raise HTTPException(500, 'MRI PNG conversion failed')
        return 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')
    mask_png = None
    if 'seg' in case['modalities']:
        mask, _, _ = load_case_modality(case, 'seg')
        raw_mask = (mask[:, :, slice_index] > 0).astype('uint8') * 255
        ok, encoded = cv2.imencode('.png', raw_mask.T)
        if ok: mask_png = 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')
    return {'case_id': case_id, 'modality': modality, 'slice_index': slice_index,
            'num_slices': int(volume.shape[2]), 'shape': list(volume.shape),
            'image': png(volume[:, :, slice_index]), 'reference_mask': mask_png}


@app.post("/api/image-analysis/sift-ransac")
def api_sift_ransac(body: SiftRansacRequest):
    case = _get_case(body.case_id, body.data_root)
    modalities = list_modalities(case)
    modality = body.modality.lower()
    if modality not in modalities:
        raise HTTPException(
            status_code=404,
            detail=f"Modality '{modality}' not available for {body.case_id}. Options: {modalities}",
        )

    cached = _load_volume_cached(case, modality)
    volume = cached["volume"]
    n_slices = int(volume.shape[2])

    if body.slice1 >= n_slices or body.slice2 >= n_slices:
        raise HTTPException(
            status_code=400,
            detail=f"Slice indices out of range. Valid range: 0–{n_slices - 1}",
        )

    if body.slice1 == body.slice2:
        raise HTTPException(
            status_code=400,
            detail="Slice 1 and Slice 2 must be different for meaningful matching.",
        )

    slice_a = get_slice(volume, body.slice1)
    slice_b = get_slice(volume, body.slice2)

    analysis = analyze_mri_sift_ransac(slice_a, slice_b, return_base64=True)

    return {
        "case_id": body.case_id,
        "modality": modality,
        "slice1": body.slice1,
        "slice2": body.slice2,
        "num_slices": n_slices,
        "keypoints_image1": analysis["keypoints_image1"],
        "keypoints_image2": analysis["keypoints_image2"],
        "good_matches": analysis["good_matches"],
        "ransac_inliers": analysis["ransac_inliers"],
        "ransac_outliers": analysis["ransac_outliers"],
        "inlier_ratio": analysis["inlier_ratio"],
        "homography_found": analysis["homography_found"],
        "success": analysis["success"],
        "message": analysis["message"],
        "original_image1": analysis["original_image1"],
        "original_image2": analysis["original_image2"],
        "sift_visualization_1": analysis["sift_visualization_1"],
        "sift_visualization_2": analysis["sift_visualization_2"],
        "matches_visualization": analysis["matches_visualization"],
        "ransac_visualization": analysis["ransac_visualization"],
    }


@app.get("/")
def serve_home():
    return FileResponse(FRONTEND_DIR / "app.html")


@app.get("/tutorial-lab")
def serve_tutorial_lab():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/image-analysis")
def serve_image_analysis():
    return FileResponse(FRONTEND_DIR / "image-analysis.html")


@app.get("/app")
def serve_application():
    return FileResponse(FRONTEND_DIR / "app.html")


@app.get("/dataset-explorer")
@app.get("/mri-workspace")
@app.get("/upload-analysis")
@app.get("/radiomics")
@app.get("/quantification")
@app.get("/preprocessing")
@app.get("/image-processing")
@app.get("/segmentation")
@app.get("/model-center")
@app.get("/model-evaluation")
@app.get("/clinical-nlp")
@app.get("/research-analytics")
@app.get("/research-reports")
@app.get("/experiment-manager")
def serve_module():
    return FileResponse(FRONTEND_DIR / "module.html")


from backend.tutorials import router as tutorial_router
app.include_router(tutorial_router)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
