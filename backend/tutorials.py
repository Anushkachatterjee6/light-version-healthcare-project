"""Validated tutorial API; CPU work runs in FastAPI's worker thread pool."""
from io import BytesIO
import json
from pathlib import Path
import threading

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from her_core.analytics import TITLES, example_data, run_analysis, records
from her_core.io import discover_cases, default_data_root, load_case_modality
from her_core.preprocess import clean_image, minmax_normalize, zscore_standardize

router = APIRouter(prefix='/api')
CNN_LOCK = threading.Lock()


class AnalysisRequest(BaseModel):
    records: list[dict] | None = Field(None, max_length=5000)
    text: str = Field('', max_length=20000)
    query: str = Field('glioma surrounding edema', max_length=1000)
    clusters: int = Field(3, ge=2, le=10)
    operation: str = Field('classification', pattern='^(classification|clustering)$')
    workload: float = Field(8, gt=0, le=100, description='Legacy field; not used by MRI tutorials')


@router.get('/tutorials')
def metadata():
    cases = discover_cases()
    return {'tutorials':[{'id':i,'title':title,'status':'implemented',
        'dataset_status':f'{len(cases)} local MRI cases' if i not in (10,11,12) else 'Synthetic brain-tumour text example ready; CSV accepted',
        'source':'Local BraTS-style NIfTI files' if i not in (10,11,12) else 'Locally generated synthetic brain-tumour report templates',
        'columns':list(example_data(i).columns) if i in (10,11,12) else [],
        'required_columns':(['text','label'] if i==10 else ['text']) if i in (10,11,12) else [],
        'endpoint':f'/api/tutorials/{i}/run'} for i,title in enumerate(TITLES,1)]}


@router.get('/tutorials/{tutorial}/example')
def example(tutorial:int):
    if tutorial not in (10,11,12): raise HTTPException(404,'Synthetic text examples are available for T10–T12 only; all other tutorials use local MRI files')
    from fastapi.responses import Response
    return Response(example_data(tutorial).to_csv(index=False),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename=T{tutorial}_SYNTHETIC_example.csv'})


def execute(tutorial, body):
    try:
        if tutorial in range(1,7): return mri_tutorial(tutorial)
        return run_analysis(tutorial, body.records, text=body.text, query=body.query, clusters=body.clusters, operation=body.operation, workload=body.workload)
    except (ValueError,KeyError,TypeError) as exc:
        raise HTTPException(422,str(exc)) from exc
    except (FileNotFoundError,ImportError) as exc:
        raise HTTPException(503,str(exc)) from exc
    except HTTPException: raise
    except Exception as exc:
        # Do not expose input records or paths in unexpected error responses.
        raise HTTPException(500,'Analysis failed. Check dataset requirements or server console.') from exc


@router.post('/tutorials/{tutorial}/run')
async def run(tutorial:int, request:Request):
    if tutorial not in range(1,16): raise HTTPException(404,'Unknown tutorial')
    raw=await bounded_body(request)
    try: body=AnalysisRequest.model_validate_json(raw or b'{}')
    except ValueError as exc: raise HTTPException(422,'Invalid request schema or JSON') from exc
    return await run_in_threadpool(execute,tutorial,body)


async def bounded_body(request):
    chunks=[]; size=0
    async for chunk in request.stream():
        size+=len(chunk)
        if size>2*1024*1024: raise HTTPException(413,'Maximum request size is 2 MB')
        chunks.append(chunk)
    return b''.join(chunks)


@router.post('/tutorials/{tutorial}/upload')
async def upload(tutorial:int,request:Request):
    if tutorial not in (10,11,12): raise HTTPException(404,'CSV upload is supported only for T10–T12 clinical-text tutorials; other tutorials use local MRI files')
    if request.headers.get('content-type','').split(';')[0] not in ('text/csv','application/csv'):
        raise HTTPException(415,'Upload UTF-8 CSV with Content-Type: text/csv')
    raw=await bounded_body(request)
    try:
        df=pd.read_csv(BytesIO(raw),encoding='utf-8',nrows=5001)
        if len(df)>5000 or len(df.columns)>50: raise ValueError('Limit: 5000 rows and 50 columns')
        data=json.loads(df.to_json(orient='records'))
        body=AnalysisRequest(records=data,text=request.query_params.get('text',''),query=request.query_params.get('query','glioma surrounding edema'),clusters=request.query_params.get('clusters',3),operation=request.query_params.get('operation','classification'),workload=request.query_params.get('workload',8))
    except (ValueError,UnicodeError,pd.errors.ParserError) as exc:
        raise HTTPException(422,'Invalid CSV or options: '+str(exc)[:250]) from exc
    return await run_in_threadpool(execute,tutorial,body)


def mri_tutorial(tutorial):
    import cv2
    import numpy as np
    from her_core.analytics import figure_plot
    from her_core.sift_ransac import analyze_mri_sift_ransac
    from her_core.visualizations import (bars, heatmap, enhancement_comparison,
        segmentation_overlay, volume_projection, file_fingerprint)
    cases=discover_cases()
    if not cases: raise FileNotFoundError('No NIfTI files under BraTS2020; restore local MRI data')
    out=dict(tutorial=tutorial,title=TITLES[tutorial-1],dataset='Local BraTS-style MRI; educational analysis',metrics={},tables={},plots=[],interpretation=[],warnings=[])
    if tutorial in (1,3):
        out['tables']['inventory']=[{'case':c['case_id'],'modalities':', '.join(sorted(c['modalities'])), 'missing':', '.join(sorted(set(['flair','t1','t1ce','t2','seg'])-set(c['modalities'])))} for c in cases]
        out['metrics']['cases']=len(cases)
        out['interpretation'].append('Inventory groups unique files by case and modality. Missing modalities are reported, not fabricated or imputed. No files are deleted.')
        if tutorial==3:
            modalities=['flair','t1','t1ce','t2','seg']
            missing=np.array([[int(m not in c['modalities']) for m in modalities] for c in cases])
            out['plots'].append(heatmap('Missing MRI modalities',missing,[c['case_id'] for c in cases],modalities,
                caption='1 = missing, 0 = available. No missing scan is replaced by zero-valued data.',limits=(0,1)))
            files=[]
            for case_item in cases:
                for name,path in case_item['modalities'].items():
                    files.append({'case':case_item['case_id'],'modality':name,'filename':path.name,'sha256':file_fingerprint(path)})
            hashes=pd.DataFrame(files)
            duplicates=hashes[hashes.sha256.duplicated(keep=False)]
            out['tables']['exact_duplicate_files']=records(duplicates)
            out['metrics']['exact_duplicate_files']=len(duplicates)
            out['interpretation'].append(f'{len(duplicates)} files share a byte-identical SHA256 digest; no files are deleted. This does not test equality of decompressed image content.')
            return out
    case=next((c for c in cases if 'flair' in c['modalities']),cases[0])
    if tutorial==6: case=max(cases,key=lambda c:len(c['modalities']))
    modality='flair' if 'flair' in case['modalities'] else next(iter(case['modalities']))
    volume,_,_=load_case_modality(case,modality)
    z=volume.shape[2]//2
    image=clean_image(volume[:,:,z])
    out['metrics'].update(case=case['case_id'],slice=z,shape=str(volume.shape))
    def image_plot(title,images):
        from matplotlib.figure import Figure
        from her_core.analytics import PLOT_LOCK
        import base64
        with PLOT_LOCK:
            columns=min(3,len(images));rows=int(np.ceil(len(images)/columns))
            fig=Figure(figsize=(10,4*rows),layout='constrained')
            axes=fig.subplots(rows,columns,squeeze=False).ravel()
            for ax,(name,array) in zip(axes,images.items()):
                ax.imshow(array.T,cmap='gray',origin='lower'); ax.set_title(name); ax.axis('off')
            for ax in axes[len(images):]:ax.axis('off')
            buf=BytesIO(); fig.savefig(buf,format='png',dpi=100)
            return {'title':title,'png':base64.b64encode(buf.getvalue()).decode(),'wide':len(images)>2}
    if tutorial==1:
        out['plots'].append(image_plot('Original MRI middle slice',{'FLAIR' if modality=='flair' else modality:image}))
        mods=sorted({m for c in cases for m in c['modalities']})
        out['plots'].append(bars('Available modality counts',mods,[sum(m in c['modalities'] for c in cases) for m in mods],ylabel='Cases with modality'))
    elif tutorial==2:
        normalized=minmax_normalize(image); standardized=zscore_standardize(image)
        out['plots'].append(image_plot('Preprocessing',{'Original':image,'Min-max':normalized,'Z-score':standardized}))
        out['plots'][-1]['caption']='Invalid intensities are cleaned before display; the original-volume NaN/Inf counts are in the image-quality table. Standardization here is per slice on nonbackground voxels.'
        out['metrics'].update(normalized_min=float(normalized.min()),normalized_max=float(normalized.max()),nonbackground_mean=float(standardized[image!=0].mean()))
        out['interpretation'].append('NaN/Inf cleaned; z-score computed on nonbackground voxels, min-max on whole slice.')
        out['tables']['image_quality']=[{'NaN_voxels':int(np.isnan(volume).sum()),
            'infinite_voxels':int(np.isinf(volume).sum()),'background_voxels':int((volume==0).sum()),
            'total_voxels':int(volume.size)}]
    elif tutorial==4:
        u8=(minmax_normalize(image)*255).astype('uint8')
        enhanced=cv2.createCLAHE(2,(8,8)).apply(u8)
        denoised=cv2.GaussianBlur(u8,(5,5),0)
        threshold,seg=cv2.threshold(u8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ok,encoded=cv2.imencode('.jpg',u8,[cv2.IMWRITE_JPEG_QUALITY,50])
        decoded=cv2.imdecode(encoded,cv2.IMREAD_GRAYSCALE)
        _,bilateral=enhancement_comparison(image)
        out['plots'].append(image_plot('Image processing',{'Original 8-bit':u8,'CLAHE':enhanced,'Denoised':denoised,'Otsu threshold':seg,'JPEG':decoded,'Bilateral (notebook)':bilateral}))
        labeled=next((c for c in cases if {'flair','seg'}<=set(c['modalities'])),None)
        if labeled:
            out['plots'].append(segmentation_overlay(labeled,volume if labeled is case else None))
        else:out['warnings'].append('No paired FLAIR/reference mask is available; ground-truth overlay omitted. Otsu is not a tumor mask.')
        out['metrics'].update(otsu_threshold=float(threshold),raw_bytes=u8.nbytes,jpeg_bytes=len(encoded),jpeg_MSE=float(np.mean((u8.astype(float)-decoded)**2)))
        out['interpretation'].append('Otsu partitions intensity, not tumor identity. JPEG is a lossy teaching example and is unsuitable for preserving diagnostic measurements.')
    elif tutorial==5:
        analysis=analyze_mri_sift_ransac(image,volume[:,:,min(z+2,volume.shape[2]-1)])
        out['metrics'].update({k:analysis[k] for k in ['good_matches','ransac_inliers','inlier_ratio','success']})
        out['plots'].append({'title':'SIFT + RANSAC correspondences','png':analysis['ransac_visualization']})
        for title,key in [('Original MRI slice 1','original_image1'),('Original MRI slice 2','original_image2'),
            ('SIFT keypoints — slice 1','sift_visualization_1'),('SIFT keypoints — slice 2','sift_visualization_2'),
            ('Feature matches before RANSAC','matches_visualization')]:
            out['plots'].append({'title':title,'png':analysis[key]})
        out['metrics']['ransac_outliers']=analysis['ransac_outliers']
        out['interpretation'].append(analysis['message'])
    elif tutorial==6:
        images={name:load_case_modality(case,name)[0][:,:,z] for name in case['modalities'] if name!='seg'}
        out['plots'].append(image_plot('Same-patient multimodal MRI',images))
        hist,edges=np.histogram(image[image!=0],bins=50)
        out['plots'].append(figure_plot('Nonbackground intensity histogram',edges[:-1],{'voxels':hist},'bar',xlabel='MRI intensity (arbitrary units)',ylabel='Nonbackground voxels'))
        if 'seg' in case['modalities']:
            mask=load_case_modality(case,'seg')[0][:,:,z]
            out['plots'].append(image_plot('Segmentation mask',{'MRI':image,'Mask (labels)':mask}))
        else: out['warnings'].append('Selected case has no segmentation; labeled cases remain available in notebook.')
        if {'flair','seg'}<=set(case['modalities']):out['plots'].append(segmentation_overlay(case,volume))
        enhanced,restored=enhancement_comparison(image)
        out['plots'].append(image_plot('Notebook enhancement and restoration comparison',{'Original':image,'CLAHE (notebook)':enhanced,'Bilateral restored':restored}))
        projection=volume_projection(clean_image(volume))
        if projection:out['plots'].append(projection)
        else:out['warnings'].append('No positive-intensity voxels available for the 3D threshold view.')
    return out


@router.post('/cnn/demo')
def cnn_demo():
    """Bounded educational train + inference; never label an untrained model as trained."""
    if not CNN_LOCK.acquire(blocking=False): raise HTTPException(409,'CNN demo already running')
    try:
        import base64
        import numpy as np
        from matplotlib.figure import Figure
        from her_core.analytics import PLOT_LOCK, figure_plot
        from her_core.cnn import build_cnn_datasets,make_tumor_cnn
        import tensorflow as tf
        tf.keras.utils.set_random_seed(42)
        data=build_cnn_datasets(target_size=(32,32),max_slices_per_case=8)
        if not len(data['X_train']) or not len(data['X_val']):
            raise ValueError('CNN requires nonempty training and validation slices with segmentation labels')
        model=make_tumor_cnn((32,32,1))
        # train_on_batch avoids tf.data thread-pool exhaustion on Windows CPU.
        # Record genuine post-epoch train/validation measurements, not the running
        # metric accumulator shared by train_on_batch and test_on_batch.
        history=[]
        for epoch in range(1,4):
            model.train_on_batch(data['X_train'],data['y_train'])
            model.reset_metrics()
            train=model.test_on_batch(data['X_train'],data['y_train'],return_dict=True)
            model.reset_metrics()
            validation=model.test_on_batch(data['X_val'],data['y_val'],return_dict=True)
            history.append({'epoch':epoch,'accuracy':float(train['accuracy']),
                'val_accuracy':float(validation['accuracy']),'loss':float(train['loss']),
                'val_loss':float(validation['loss'])})
            model.reset_metrics()
        probabilities=model(data['X_val'],training=False).numpy().ravel()
        if len(probabilities)!=len(data['val_records']) or not np.isfinite(probabilities).all():
            raise ValueError('CNN returned invalid or incomplete validation probabilities')
        predicted=(probabilities>=.5).astype(int)
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
        classes=['No Tumor','Tumor Present']
        actual=data['y_val'].astype(int)
        cm=confusion_matrix(actual,predicted,labels=[0,1])
        report=classification_report(actual,predicted,labels=[0,1],target_names=classes,
            output_dict=True,zero_division=0)
        predictions=[{'case':r.case_id,'slice':r.slice_idx,'actual':r.label,'predicted':int(p),
            'actual_class':classes[r.label],'predicted_class':classes[int(p)],
            'probability':float(prob),'no_tumor_probability':float(1-prob),
            'predicted_class_score':float(prob if p else 1-prob),'correct':bool(p==r.label)}
            for r,p,prob in zip(data['val_records'],predicted,probabilities)]
        # Individual compact images let the website stack comparisons on mobile;
        # the existing composite notebook-style plot remains for compatibility.
        import cv2
        def image_url(array):
            # Match imshow(array.T, origin='lower') in the notebook previews.
            pixels=np.flipud(np.asarray(array).T)
            pixels=np.clip(pixels*255,0,255).astype(np.uint8)
            ok,png=cv2.imencode('.png',pixels)
            if not ok: raise ValueError('CNN preview PNG conversion failed')
            return 'data:image/png;base64,'+base64.b64encode(png).decode('ascii')
        preview_indices=np.linspace(0,len(predictions)-1,min(4,len(predictions))).astype(int)
        previews=[{**predictions[i], 'input_image':image_url(data['val_records'][i].image),
            'ground_truth_mask':image_url(data['val_records'][i].mask) if data['val_records'][i].mask is not None else None,
            'width':int(data['val_records'][i].image.shape[1]),
            'height':int(data['val_records'][i].image.shape[0])} for i in preview_indices]
        epochs=[row['epoch'] for row in history]
        plots=[figure_plot('CNN training and validation accuracy',epochs,
            {'Train (post-epoch)': [row['accuracy'] for row in history],
             'Validation': [row['val_accuracy'] for row in history]}),
            figure_plot('CNN training and validation loss',epochs,
            {'Train (post-epoch)': [row['loss'] for row in history],
             'Validation': [row['val_loss'] for row in history]})]
        # Use the same png field consumed by the existing frontend renderer.
        with PLOT_LOCK:
            def encode_figure(fig,title):
                buf=BytesIO();fig.savefig(buf,format='png',dpi=110)
                return {'title':title,'png':base64.b64encode(buf.getvalue()).decode('ascii')}
            fig=Figure(figsize=(6,4),layout='constrained');ax=fig.subplots()
            ax.imshow(cm,cmap='Blues',vmin=0)
            for (i,j),value in np.ndenumerate(cm):ax.text(j,i,str(value),ha='center',va='center')
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=classes,yticklabels=classes,
                xlabel='Predicted',ylabel='Actual',title='CNN validation confusion matrix')
            plots.append(encode_figure(fig,'CNN validation confusion matrix'))
            # Spread previews across the holdout set, without choosing by success.
            fig=Figure(figsize=(10,2.8*len(preview_indices)),layout='constrained')
            axes=fig.subplots(len(preview_indices),3,squeeze=False)
            for row,i in enumerate(preview_indices):
                rec=data['val_records'][i];prediction=predictions[i]
                for ax in axes[row]:ax.axis('off')
                axes[row,0].imshow(rec.image.T,cmap='gray',origin='lower',vmin=0,vmax=1)
                axes[row,0].set_title(f'{rec.case_id}\nSlice {rec.slice_idx}: input (32 x 32)',fontsize=9)
                axes[row,1].imshow(rec.mask.T,cmap='magma',origin='lower',vmin=0,vmax=1)
                axes[row,1].set_title('Mask-derived actual: '+prediction['actual_class'],fontsize=9)
                axes[row,2].imshow(rec.image.T,cmap='gray',origin='lower',vmin=0,vmax=1)
                axes[row,2].set_title(f"Predicted: {prediction['predicted_class']}\nClass score: {prediction['predicted_class_score']:.3f}; P(tumor): {prediction['probability']:.3f}",fontsize=9)
            plots.append(encode_figure(fig,'Validation input, ground-truth mask and CNN prediction (four evenly spaced previews at most)'))
        return dict(tutorial=5,title='CNN training and inference smoke demonstration',dataset='Local labeled BraTS MRI',
            workflow='training_then_validation_inference',
            metrics={'accuracy':float(accuracy_score(actual,predicted)),
                'precision':float(precision_score(actual,predicted,zero_division=0)),
                'recall':float(recall_score(actual,predicted,zero_division=0)),
                'F1':float(f1_score(actual,predicted,zero_division=0)),
                'epochs':len(history),'training_slices':len(data['X_train']),
                'validation_slices':len(data['X_val']),'input_width':32,'input_height':32},
            tables={'predictions':predictions,'training_history':history,
                'confusion_matrix':[{'actual':name,classes[0]:int(cm[i,0]),classes[1]:int(cm[i,1])} for i,name in enumerate(classes)],
                'classification_report':[{'class':name,**{k:float(v) for k,v in values.items()}}
                    for name,values in report.items() if isinstance(values,dict)],
                'split':[{'train_cases':', '.join(data['train_case_ids']),'validation_cases':', '.join(data['val_case_ids'])}]},
            plots=plots,previews=previews,model_task='classification',
            interpretation=['Training followed by inference: a fresh model is trained for three epochs (one full training batch per epoch) on 32×32 slices; no saved weights are loaded or overwritten. Use the notebook for the full 128×128, eight-epoch experiment.',
                'Accuracy/loss curves are measured on each complete split after each epoch with dropout disabled. Validation patients never enter training. Metrics and the classification report use every labeled validation slice, not just the image previews.',
                'Label 1 means tumor pixels are present in the prepared segmentation mask. probability is the sigmoid score P(Tumor Present); the other class score is 1 minus that value. The predicted-class score is not a calibrated clinical confidence.',
                'This endpoint requires segmentation labels to train and evaluate. Unlabeled inference accuracy, classification reports and confusion matrices cannot be calculated without ground truth. This is an educational experiment, not a validated diagnostic model.'],
            warnings=['Validation contains only one actual class; metrics for an unsupported class are set to zero, not evidence of clinical performance.'] if len(np.unique(actual))<2 else [])
    except (ImportError,FileNotFoundError) as exc: raise HTTPException(503,str(exc)) from exc
    except Exception as exc: raise HTTPException(500,'CNN demonstration failed; inspect local dependencies and labeled MRI files') from exc
    finally:
        try:
            if 'tf' in locals(): tf.keras.backend.clear_session()
        finally:
            CNN_LOCK.release()
