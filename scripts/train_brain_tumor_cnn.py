"""Reproducible patient-level BraTS FLAIR slice CNN training.

Run from the repository root: py -3 scripts/train_brain_tumor_cnn.py
Artifacts are written only after a real training/evaluation run completes.
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from her_core.cnn import build_case_slice_dataset, discover_labeled_cases, make_tumor_cnn, records_to_arrays

SEED=42; SIZE=(128,128); MAX_SLICES=40; OUT=ROOT/'models'
def jsonable(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)
def main():
    import tensorflow as tf
    random.seed(SEED);np.random.seed(SEED);tf.keras.utils.set_random_seed(SEED)
    cases=discover_labeled_cases(); ids=[c['case_id'] for c in cases]
    if len(ids)<6: raise RuntimeError('Need at least six labeled cases for fixed 4/1/1 patient split')
    rng=np.random.default_rng(SEED);rng.shuffle(ids); split={'test_case_ids':ids[:1],'validation_case_ids':ids[1:2],'training_case_ids':ids[2:]}
    by_id={c['case_id']:c for c in cases}; records={cid:build_case_slice_dataset(by_id[cid],target_size=SIZE,max_slices=MAX_SLICES) for cid in ids}
    arrays={};
    for name,key in [('train','training_case_ids'),('val','validation_case_ids'),('test','test_case_ids')]: arrays[name]=records_to_arrays([r for cid in split[key] for r in records[cid]])
    Xtr,ytr,_=arrays['train'];Xv,yv,_=arrays['val'];Xt,yt,test_records=arrays['test']
    if min(len(Xtr),len(Xv),len(Xt))==0 or len(np.unique(ytr))<2: raise RuntimeError('Split lacks usable slices or both training classes')
    counts=np.bincount(ytr.astype(int),minlength=2);weights={0:len(ytr)/(2*counts[0]),1:len(ytr)/(2*counts[1])}
    OUT.mkdir(exist_ok=True); checkpoint=OUT/'brain_tumor_cnn.keras'
    model=make_tumor_cnn((*SIZE,1)); callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True),tf.keras.callbacks.ModelCheckpoint(checkpoint,monitor='val_loss',save_best_only=True)]
    hist=model.fit(Xtr,ytr,validation_data=(Xv,yv),epochs=12,batch_size=16,class_weight=weights,callbacks=callbacks,verbose=2).history
    model=tf.keras.models.load_model(checkpoint);prob=model.predict(Xt,verbose=0).ravel();pred=(prob>=.5).astype(int)
    report={'test_accuracy':float((pred==yt).mean()),'precision_recall_f1':classification_report(yt,pred,labels=[0,1],target_names=['No Tumor','Tumor Present'],output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(yt,pred,labels=[0,1]).tolist(),'test_case_ids':split['test_case_ids'],'test_slice_count':len(yt),'test_class_counts':np.bincount(yt.astype(int),minlength=2).tolist()}
    if len(np.unique(yt))==2: report['roc_auc']=float(roc_auc_score(yt,prob))
    metadata={'model_version':'brats-flair-slice-cnn-v1','architecture':'make_tumor_cnn Conv32/64/128 sigmoid','input_shape':[128,128,1],'modality':'FLAIR','class_mapping':{'0':'No Tumor','1':'Tumor Present'},'preprocessing':['clean NaN/Inf','center crop 160x160 for >=160px slices','per-slice min-max normalization','resize 128x128','single channel'],'dataset':'local BraTS2020','split':split,'slice_counts':{'train':len(ytr),'validation':len(yv),'test':len(yt)},'train_class_counts':counts.tolist(),'class_weight':weights,'seed':SEED,'max_slices_per_case':MAX_SLICES,'epochs_requested':12}
    (OUT/'model_metadata.json').write_text(json.dumps(metadata,indent=2,default=jsonable));(OUT/'training_history.json').write_text(json.dumps(hist,indent=2,default=jsonable));(OUT/'evaluation_report.json').write_text(json.dumps(report,indent=2,default=jsonable))
    from matplotlib import pyplot as plt
    for keys,name in [(('accuracy','val_accuracy'),'training_accuracy.png'),(('loss','val_loss'),'training_loss.png')]:
        plt.figure();plt.plot(hist[keys[0]],label='train');plt.plot(hist[keys[1]],label='validation');plt.legend();plt.xlabel('epoch');plt.ylabel(keys[0].replace('val_',''));plt.tight_layout();plt.savefig(OUT/name,dpi=140);plt.close()
    plt.figure();plt.imshow(report['confusion_matrix'],cmap='Blues');plt.xticks([0,1],['No Tumor','Tumor Present']);plt.yticks([0,1],['No Tumor','Tumor Present']);plt.xlabel('Predicted');plt.ylabel('Actual');plt.title('Held-out confusion matrix');[plt.text(j,i,str(v),ha='center',va='center') for i,row in enumerate(report['confusion_matrix']) for j,v in enumerate(row)];plt.tight_layout();plt.savefig(OUT/'confusion_matrix.png',dpi=140);plt.close()
    if 'roc_auc' in report:
        from sklearn.metrics import roc_curve
        fpr,tpr,_=roc_curve(yt,prob);plt.figure();plt.plot(fpr,tpr,label=f"AUC={report['roc_auc']:.3f}");plt.plot([0,1],[0,1],'--');plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.legend();plt.tight_layout();plt.savefig(OUT/'roc_curve.png',dpi=140);plt.close()
    print(json.dumps({'checkpoint':str(checkpoint),'metadata':metadata,'evaluation':report},indent=2,default=jsonable))
if __name__=='__main__': main()
