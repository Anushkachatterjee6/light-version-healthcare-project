"""Data-derived T7–T15 brain-tumour MRI and educational clinical-text analyses."""
from __future__ import annotations
import base64, json, re, threading
from functools import lru_cache
from io import BytesIO
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_fscore_support, r2_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .io import discover_cases, load_case_modality

TITLES=['Understanding biomedical MRI data','MRI data standardization and cleaning','Redundant MRI data removal and missing-data handling','Biomedical image processing and analysis','SIFT, RANSAC and CNN-based brain tumour analysis','Biomedical MRI visualization and segmentation review','Predictive analysis for brain tumour detection workflow','Brain tumour MRI resource planning — Part 1','Statistical analysis for brain tumour MRI planning — Part 2','Classification and clustering of brain tumour clinical text','Information extraction for brain tumour reports','Ranking and word-sense disambiguation of brain tumour text','Dimensionality reduction for exploratory brain tumour MRI analysis','MRI imaging factors associated with tumour burden — Part 1','Multivariable MRI feature evaluation for tumour burden — Part 2']
PLOT_LOCK=threading.Lock()
MRI_FEATURES=['flair_mean','flair_std','flair_p95','nonzero_fraction','volume_slices','modalities_available']

def records(df): return json.loads(df.replace([np.inf,-np.inf],np.nan).to_json(orient='records',date_format='iso'))

def figure_plot(title,x,series,kind='line',*,xlabel='',ylabel='',caption=None):
    from matplotlib.figure import Figure
    with PLOT_LOCK:
        fig=Figure(figsize=(8,4),layout='constrained');ax=fig.subplots()
        for label,y in series.items():
            if kind=='scatter':ax.scatter(x,y,label=label,s=28,alpha=.8)
            elif kind=='bar':ax.bar(x,y,label=label)
            else:ax.plot(x,y,marker='o',label=label)
        ax.set_title(title);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
        if len(series)>1:ax.legend()
        ax.tick_params(axis='x',labelrotation=25);buf=BytesIO();fig.savefig(buf,format='png',dpi=110)
        result={'title':title,'png':base64.b64encode(buf.getvalue()).decode(),'x_label':xlabel,'y_label':ylabel}
        if caption:result['caption']=caption
        return result

def _plot(title,draw,*,xlabel='',ylabel='',caption=None):
    from matplotlib.figure import Figure
    with PLOT_LOCK:
        fig=Figure(figsize=(8,4),layout='constrained');ax=fig.subplots();draw(ax);ax.set_title(title);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.tick_params(axis='x',labelrotation=25)
        buf=BytesIO();fig.savefig(buf,format='png',dpi=110);result={'title':title,'png':base64.b64encode(buf.getvalue()).decode(),'x_label':xlabel,'y_label':ylabel}
        if caption:result['caption']=caption
        return result

@lru_cache(maxsize=1)
def mri_case_features():
    """One row per supplied case; never infer a missing modality, mask, or diagnosis."""
    rows=[]
    for case in discover_cases():
        mods=case['modalities'];row={'case':case['case_id'],'modalities_available':len(mods),**{f'has_{m}':int(m in mods) for m in ('flair','t1','t1ce','t2','seg')}}
        row.update(flair_mean=np.nan,flair_std=np.nan,flair_p95=np.nan,nonzero_fraction=np.nan,volume_slices=np.nan,tumour_voxels=np.nan,tumour_fraction=np.nan,tumour_slices=np.nan)
        if 'flair' in mods:
            volume,_,_=load_case_modality(case,'flair');finite=volume[np.isfinite(volume)];nonzero=finite[finite!=0]
            row.update(volume_slices=int(volume.shape[2]),nonzero_fraction=float(nonzero.size/volume.size),flair_mean=float(nonzero.mean()) if nonzero.size else np.nan,flair_std=float(nonzero.std()) if nonzero.size else np.nan,flair_p95=float(np.percentile(nonzero,95)) if nonzero.size else np.nan)
        if 'seg' in mods:
            seg,_,_=load_case_modality(case,'seg');mask=np.isfinite(seg)&(seg>0);row.update(tumour_voxels=int(mask.sum()),tumour_fraction=float(mask.mean()),tumour_slices=int(mask.any(axis=(0,1)).sum()));row['tumour_label']='Tumour present' if mask.any() else 'No tumour in supplied segmentation'
        else:row['tumour_label']='Unknown — no segmentation'
        rows.append(row)
    if not rows:raise FileNotFoundError('No NIfTI files under BraTS2020; restore local MRI data')
    return pd.DataFrame(rows)

def example_data(tutorial):
    """Explicitly synthetic educational report text only; other tutorials use local MRI."""
    if tutorial not in (10,11,12):return mri_case_features().copy()
    templates={'glioma-related':['MRI demonstrates infiltrative glioma with surrounding edema','Enhancing intra-axial glioma and necrotic component','Glioma in frontal lobe with mass effect'],'meningioma-related':['Extra-axial enhancing meningioma along falx','Dural based meningioma with homogeneous enhancement','Meningioma causes mild adjacent edema'],'no-tumour-finding':['No focal mass or abnormal enhancement is identified','MRI brain without evidence of intracranial tumour','No lesion, mass effect or midline shift']}
    return pd.DataFrame([{'text':f'{text}. {suffix}','label':label} for label,texts in templates.items() for text in texts for suffix in ('Educational synthetic report','Template example','Not a patient report','Teaching corpus')])

def _base(tutorial,df,dataset):return dict(tutorial=tutorial,title=TITLES[tutorial-1],dataset=dataset,metrics={},tables={},plots=[],interpretation=[],warnings=[],rows_input=len(df))
def _mri_base(tutorial):
    df=mri_case_features().copy();out=_base(tutorial,df,'Local BraTS-style MRI files; case-level imaging features computed at request time')
    out['tables']['case_feature_inventory']=records(df);out['tables']['missing_values']=records(df.isna().sum().rename('missing').rename_axis('column').reset_index());out['metrics'].update(mri_cases=len(df),cases_with_flair=int(df.has_flair.sum()),cases_with_segmentation=int(df.has_seg.sum()),cases_without_segmentation=int((1-df.has_seg).sum()));return df,out
def _modality_plot(df):
    return figure_plot('MRI modality and segmentation availability',['FLAIR','T1','T1ce','T2','Segmentation'],{'Cases':[int(df[f'has_{m}'].sum()) for m in ('flair','t1','t1ce','t2','seg')]},'bar',xlabel='Available local asset',ylabel='Cases',caption='Availability is read from local case folders; missing assets are not imputed.')
def _heatmap_plot(title,matrix,labels,caption):
    def draw(ax):
        image=ax.imshow(matrix,cmap='coolwarm',vmin=-1,vmax=1);ax.set_xticks(range(len(labels)),labels,rotation=30,ha='right');ax.set_yticks(range(len(labels)),labels)
        for i in range(len(labels)):
            for j in range(len(labels)):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8)
        ax.figure.colorbar(image,ax=ax,label='Pearson correlation')
    return _plot(title,draw,caption=caption)

def _run_mri_analysis(tutorial):
    df,out=_mri_base(tutorial);labeled=df[df.has_seg==1].copy()
    if tutorial==7:
        out['tables']['review_queue']=records(df[['case','tumour_label','has_flair','has_seg','modalities_available']]);counts=df.tumour_label.value_counts();out['plots']=[_modality_plot(df),figure_plot('Case distribution by reference-label availability',counts.index,{'Cases':counts.values},'bar',xlabel='Reference-label status',ylabel='MRI cases')];out['metrics']['cases_requiring_reference_review']=int((df.has_seg==0).sum());out['interpretation'].append('This workflow queue is based on supplied segmentation availability, not a tumour prediction. The local cases do not provide a verified tumour-negative cohort, so actual-versus-predicted classification metrics are not applicable.')
    elif tutorial==8:
        out['plots']=[_modality_plot(df),figure_plot('MRI case distribution by available modality count',df.case,{'Modalities available':df.modalities_available},'bar',xlabel='Case',ylabel='Available modalities')]
        if len(labeled):out['plots'].append(figure_plot('Reference tumour burden by labeled case',labeled.case,{'Tumour voxel fraction':labeled.tumour_fraction},'bar',xlabel='Case',ylabel='Segmented tumour voxels / volume voxels'))
        out['interpretation'].append('Planning summaries describe the supplied MRI and annotation inventory. Segmentation presence is an annotation attribute, not a diagnosis for unsegmented cases.')
    elif tutorial==9:
        available=df[MRI_FEATURES].dropna(axis=1,how='all');corr=available.corr().fillna(0);out['tables']['feature_summary']=records(available.describe().reset_index());out['plots']=[_heatmap_plot('MRI feature correlation heatmap',corr.to_numpy(),list(corr.columns),'Pairwise correlations of MRI-derived features. Small case count prevents confirmatory inference.'),_modality_plot(df)]
        if len(labeled)>=4 and labeled.tumour_fraction.nunique()>1:
            median=labeled.tumour_fraction.median();low=labeled[labeled.tumour_fraction<=median];high=labeled[labeled.tumour_fraction>median]
            if len(low) and len(high):
                test=stats.mannwhitneyu(low.flair_mean,high.flair_mean,alternative='two-sided');out['metrics'].update(tumour_burden_split_median=float(median),flair_mean_mann_whitney_U=float(test.statistic),flair_mean_mann_whitney_p=float(test.pvalue));out['tables']['burden_groups']=records(labeled.assign(burden_group=np.where(labeled.tumour_fraction<=median,'at/below median','above median'))[['case','tumour_fraction','flair_mean','burden_group']])
        out['warnings'].append('Any p-value is exploratory: the local set is tiny and burden groups are derived from these same segmentations.');out['interpretation'].append('Feature comparisons use real image-derived values. They do not compare tumour-positive with tumour-negative patients because a verified control cohort is not supplied.')
    elif tutorial==13:
        features=[c for c in MRI_FEATURES if df[c].notna().sum()>=2];x=make_pipeline(SimpleImputer(strategy='median'),StandardScaler()).fit_transform(df[features]);pca=PCA().fit(x);coords=pca.transform(x);out['tables']['explained_variance']=records(pd.DataFrame({'component':np.arange(1,len(features)+1),'variance':pca.explained_variance_ratio_,'cumulative':np.cumsum(pca.explained_variance_ratio_)}));out['tables']['pca_coordinates']=records(pd.DataFrame({'case':df.case,'PC1':coords[:,0],'PC2':coords[:,1],'segmentation_available':df.has_seg.astype(bool)}))
        def pca_draw(ax):
            work=pd.DataFrame({'x':coords[:,0],'y':coords[:,1],'seg':df.has_seg,'case':df.case})
            for avail,group in work.groupby('seg'):
                ax.scatter(group.x,group.y,label='Segmentation available' if avail else 'No segmentation',s=45)
                for row in group.itertuples():ax.annotate(row.case.replace('BraTS20_Training_',''),(row.x,row.y),fontsize=7)
            ax.legend()
        out['plots']=[_plot('PCA projection of MRI-derived case features',pca_draw,xlabel='PC1',ylabel='PC2'),figure_plot('PCA variance explained',np.arange(1,len(features)+1),{'Individual':pca.explained_variance_ratio_,'Cumulative':np.cumsum(pca.explained_variance_ratio_)},xlabel='Principal component',ylabel='Fraction of standardized variance')]
        if len(df)>=6:
            tsne=TSNE(n_components=2,perplexity=min(3,len(df)-1),init='pca',learning_rate='auto',random_state=42).fit_transform(x);out['plots'].append(figure_plot('t-SNE exploratory MRI feature projection',tsne[:,0],{'Dimension 2':tsne[:,1]},'scatter',xlabel='t-SNE dimension 1',ylabel='t-SNE dimension 2',caption='t-SNE is exploratory; distances and apparent clusters are unstable for this small local dataset.'))
        out['metrics']['first_two_variance']=float(pca.explained_variance_ratio_[:2].sum());out['interpretation'].append('PCA and t-SNE use standardized real MRI-derived features. Point color denotes segmentation availability, not a tumour class, because unsegmented cases have unknown reference status.')
    elif tutorial==14:
        if len(labeled)<2:raise ValueError('At least two cases with supplied segmentation are required')
        columns=[c for c in MRI_FEATURES+['tumour_fraction','tumour_slices'] if labeled[c].notna().sum()>=2];corr=labeled[columns].corr().fillna(0);out['tables']['labeled_case_summary']=records(labeled[['case']+columns]);out['plots']=[figure_plot('Tumour burden versus FLAIR mean intensity',labeled.flair_mean,{'Tumour fraction':labeled.tumour_fraction},'scatter',xlabel='FLAIR nonzero-voxel mean',ylabel='Segmented tumour fraction'),_heatmap_plot('Imaging feature and tumour-burden correlations',corr.to_numpy(),columns,'Pairwise complete correlations among supplied labeled cases; association is not causation.')];out['interpretation'].append('Only imaging-derived quantities and supplied segmentation masks are analyzed. No age, grade, location, or other missing clinical variable is invented.')
    elif tutorial==15:
        features=[c for c in MRI_FEATURES if labeled[c].notna().sum()>=2]
        if len(labeled)<5 or labeled.tumour_fraction.nunique()<2:raise ValueError('Held-out tumour-burden evaluation requires at least five varied labeled cases')
        train,test=train_test_split(labeled,test_size=max(2,int(np.ceil(len(labeled)*.33))),random_state=42);model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=1.0)).fit(train[features],train.tumour_fraction);predicted=model.predict(test[features]);actual=test.tumour_fraction.to_numpy();out['metrics'].update(MAE=float(mean_absolute_error(actual,predicted)),RMSE=float(np.sqrt(mean_squared_error(actual,predicted))),R2=float(r2_score(actual,predicted)) if len(actual)>1 else None,train_cases=len(train),heldout_cases=len(test));out['tables']['heldout_tumour_burden_predictions']=records(pd.DataFrame({'case':test.case.to_numpy(),'actual_tumour_fraction':actual,'predicted_tumour_fraction':predicted,'residual':actual-predicted}));out['tables']['feature_coefficients']=[{'feature':f,'coefficient_per_training_SD':float(v)} for f,v in zip(features,model[-1].coef_)];out['plots']=[figure_plot('Held-out predicted versus actual tumour fraction',actual,{'Predicted':predicted},'scatter',xlabel='Actual segmented tumour fraction',ylabel='Predicted tumour fraction'),figure_plot('MRI feature importance for tumour-burden model',features,{'Ridge coefficient':model[-1].coef_},'bar',xlabel='MRI-derived feature',ylabel='Coefficient per training SD')];out['warnings'].append('This is held-out regression of segmentation-derived tumour burden, not tumour/no-tumour classification. ROC, AUC, precision-recall and classification confusion matrices are invalid without a verified negative class.');out['interpretation'].append('The case-level split keeps each case wholly in train or test. With only a few labeled local cases, values demonstrate the pipeline, not a validated model result.')
    return out

def _t10(df,out,operation,clusters):
    if 'label' not in df: raise ValueError('T10 requires a label column')
    counts=df.label.value_counts()
    if len(df)<12 or len(counts)<2 or counts.min()<3: raise ValueError('Need 12 texts, two classes and at least three texts per class')
    vectorizer=TfidfVectorizer(stop_words='english',max_features=1000); matrix=vectorizer.fit_transform(df.text)
    if operation=='classification':
        tr,te=train_test_split(np.arange(len(df)),test_size=max(len(counts),int(np.ceil(.25*len(df)))),stratify=df.label,random_state=42); model=LogisticRegression(max_iter=500,random_state=42).fit(matrix[tr],df.label.iloc[tr]); pred=model.predict(matrix[te]); labels=model.classes_; cm=confusion_matrix(df.label.iloc[te],pred,labels=labels); p,r,f,_=precision_recall_fscore_support(df.label.iloc[te],pred,average='weighted',zero_division=0)
        from sklearn.metrics import classification_report
        out['operation']='classification';out['metrics'].update(accuracy=float(accuracy_score(df.label.iloc[te],pred)),precision=float(p),recall=float(r),F1=float(f),train_texts=len(tr),heldout_texts=len(te));out['tables']['confusion_matrix']=records(pd.DataFrame(cm,columns=labels).assign(actual=labels));out['tables']['sample_predictions']=records(pd.DataFrame({'text':df.text.iloc[te].to_numpy(),'actual':df.label.iloc[te].to_numpy(),'predicted':pred}));out['tables']['classification_report']=[{'class':k,**{a:float(b) for a,b in v.items()}} for k,v in classification_report(df.label.iloc[te],pred,output_dict=True,zero_division=0).items() if isinstance(v,dict)];out['plots']=[_heatmap_plot('Synthetic text classification confusion matrix',cm,labels,'Rows are actual labels; columns are predicted labels on the held-out synthetic teaching set.')];out['interpretation'].append('Supervised TF-IDF + Logistic Regression; metrics are only for held-out synthetic templates, not clinical validation.')
    else:
        k=min(clusters,len(df)-1); km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(matrix);terms=vectorizer.get_feature_names_out();out['operation']='clustering';out['tables']['cluster_assignments']=records(pd.DataFrame({'text':df.text,'synthetic_label':df.label,'cluster':km.labels_}));out['tables']['cluster_representatives']=[{'cluster':int(i),'size':int((km.labels_==i).sum()),'terms':', '.join(terms[np.argsort(c)[-5:][::-1]])} for i,c in enumerate(km.cluster_centers_)];out['metrics']['clusters']=k
        if len(set(km.labels_))>1:out['metrics']['silhouette_cosine']=float(silhouette_score(matrix,km.labels_,metric='cosine'))
        coords=PCA(n_components=2).fit_transform(matrix.toarray());out['tables']['pca_coordinates']=records(pd.DataFrame({'PC1':coords[:,0],'PC2':coords[:,1],'cluster':km.labels_,'text':df.text}));out['plots']=[figure_plot('Synthetic text cluster-size distribution',np.arange(k),{'Texts':np.bincount(km.labels_,minlength=k)},'bar',xlabel='Cluster',ylabel='Texts'),figure_plot('PCA projection of synthetic text clusters',coords[:,0],{'PC2':coords[:,1]},'scatter',xlabel='PC1',ylabel='PC2')];out['interpretation'].append('Unsupervised TF-IDF + K-Means clustering of synthetic teaching reports; clusters do not represent diagnoses.')
    return out

def _run_text_analysis(tutorial,data=None,*,text='',query='glioma surrounding edema',clusters=3,operation='classification'):
    synthetic=data is None;df=example_data(tutorial) if synthetic else pd.DataFrame(data)
    if df.empty or 'text' not in df:raise ValueError('Text records must include a nonempty text column')
    if df.text.isna().any() or not df.text.map(lambda x:isinstance(x,str) and bool(x.strip())).all():raise ValueError('Text records must be nonempty strings')
    out=_base(tutorial,df,'Clearly labeled synthetic educational brain-tumour report templates; not real patient reports' if synthetic else 'User-supplied text; provenance not independently verified');out['tables']['missing_values']=records(df.isna().sum().rename('missing').rename_axis('column').reset_index());docs=df.text.tolist()
    if tutorial==10:
        return _t10(df,out,operation,clusters)
    if tutorial==10:
        if 'label' not in df:raise ValueError('T10 text classification requires a label column')
        counts=df.label.value_counts()
        if len(df)<12 or len(counts)<2 or counts.min()<3:raise ValueError('Need 12 texts, two classes and at least three texts per class')
        tr,te=train_test_split(np.arange(len(df)),test_size=max(len(counts),int(np.ceil(.25*len(df)))),stratify=df.label,random_state=42);vectorizer=TfidfVectorizer(stop_words='english',max_features=1000);xtr=vectorizer.fit_transform(df.text.iloc[tr]);xte=vectorizer.transform(df.text.iloc[te]);model=LogisticRegression(max_iter=500,random_state=42).fit(xtr,df.label.iloc[tr]);pred=model.predict(xte);labels=model.classes_;cm=confusion_matrix(df.label.iloc[te],pred,labels=labels);p,r,f,_=precision_recall_fscore_support(df.label.iloc[te],pred,average='weighted',zero_division=0);out['metrics'].update(accuracy=float(accuracy_score(df.label.iloc[te],pred)),precision=float(p),recall=float(r),F1=float(f),train_texts=len(tr),heldout_texts=len(te));out['tables']['confusion_matrix']=records(pd.DataFrame(cm,columns=labels).assign(actual=labels));out['tables']['holdout_predictions']=records(pd.DataFrame({'actual':df.label.iloc[te].to_numpy(),'predicted':pred}));km=KMeans(n_clusters=min(clusters,len(tr)-1),n_init=10,random_state=42).fit(xtr);terms=vectorizer.get_feature_names_out();out['tables']['clusters']=[{'cluster':i,'size':int((km.labels_==i).sum()),'terms':', '.join(terms[np.argsort(center)[-5:][::-1]])} for i,center in enumerate(km.cluster_centers_)];
        if len(set(km.labels_))>1:out['metrics']['silhouette_cosine']=float(silhouette_score(xtr,km.labels_,metric='cosine'))
        out['plots']=[_heatmap_plot('Synthetic brain-tumour text holdout confusion matrix',cm,labels,'Rows=actual, columns=predicted; measured only on held-out synthetic templates.'),figure_plot('Training text clusters',np.arange(len(out['tables']['clusters'])),{'Texts':np.bincount(km.labels_)},'bar',xlabel='Cluster',ylabel='Training texts')];out['interpretation'].append('Classifier and clusters are computed from a deliberately synthetic teaching corpus. Its metrics do not establish clinical-report performance.')
    elif tutorial==11:
        context=text.strip() or docs[0];groups={'tumour type':['glioma','meningioma','tumour','lesion'],'anatomy':['frontal lobe','temporal lobe','brain'],'finding':['edema','mass effect','midline shift','necrosis','enhancement']};found=[]
        for category,terms in groups.items():
            for term in terms:
                for match in re.finditer(r'\b'+re.escape(term)+r'\b',context,re.I):found.append({'entity':match.group(),'category':category,'start':match.start(),'end':match.end(),'method':'transparent lexicon'})
        for match in re.finditer(r'\b\d+(?:\.\d+)?\s*(?:mm|cm)\b',context,re.I):found.append({'entity':match.group(),'category':'measurement','start':match.start(),'end':match.end(),'method':'measurement regex'})
        found.sort(key=lambda r:r['start']);out['tables']['entities']=found;out['annotations']={'text':context,'spans':[{'start':e['start'],'end':e['end'],'category':e['category']} for e in found]};out['metrics']['mentions']=len(found)
        if found:
            counts=pd.Series([e['category'] for e in found]).value_counts();out['plots'].append(figure_plot('Extracted brain-tumour report entities',counts.index,{'Mentions':counts.values},'bar',xlabel='Entity category',ylabel='Mentions'))
        out['interpretation'].append('Extraction is rule-based and transparent, not clinical-grade NER. No annotated evaluation corpus is available, so accuracy is not calculated.')
    else:
        if not query.strip():raise ValueError('Enter a nonempty ranking query')
        vectorizer=TfidfVectorizer(stop_words='english');matrix=vectorizer.fit_transform(docs);scores=cosine_similarity(vectorizer.transform([query]),matrix).ravel();order=np.argsort(-scores)[:10];out['tables']['ranking']=[{'rank':i+1,'document':docs[j],'score':float(scores[j])} for i,j in enumerate(order)];context=text.strip() or 'Glioma with surrounding edema shows contrast enhancement.';senses={'mass':{'lesion or tumour':['glioma','tumour','lesion'],'mass effect':['effect','shift','compression']},'enhancement':{'contrast enhancement':['contrast','enhancing'],'noncontrast wording':['without contrast','noncontrast']}};evidence=[];resolved=[]
        for term,options in senses.items():
            if re.search(r'\b'+term+r'\b',context,re.I):
                scored={sense:sum(cue in context.lower() for cue in cues) for sense,cues in options.items()};best=max(scored,key=scored.get);resolved.append({'term':term,'context':context,'interpretation':best if scored[best] and list(scored.values()).count(scored[best])==1 else 'ambiguous','method':'explicit context cues'});evidence += [{'term':term,'sense':k,'matched_cues':v} for k,v in scored.items()]
        out['tables']['disambiguation']=resolved or [{'context':context,'interpretation':'No supported ambiguous term found','method':'explicit context cues'}];out['tables']['sense_evidence']=evidence;out['metrics']['top_relevance']=float(scores.max());out['plots'].append(figure_plot('Ranked report relevance (TF-IDF cosine similarity)',[f'Report {i+1}' for i in range(len(order))],{'Similarity':scores[order]},'bar',xlabel='Ranked synthetic report',ylabel='Cosine similarity'))
        if evidence:out['plots'].append(figure_plot('Word-sense context evidence',[f"{e['term']}: {e['sense']}" for e in evidence],{'Matched cues':[e['matched_cues'] for e in evidence]},'bar',xlabel='Term interpretation',ylabel='Cue count'))
        out['interpretation'].append('Lexical relevance and context-cue counts are not diagnosis or clinical recommendation. Synthetic templates are marked as such.')
    return out

def run_analysis(tutorial,data=None,*,text='',query='glioma surrounding edema',clusters=3,operation='classification',workload=8):
    if tutorial not in range(7,16):raise ValueError('Tutorial must be T7–T15')
    if not 2<=clusters<=10:raise ValueError('Cluster count must be 2–10')
    if tutorial in (10,11,12):return _run_text_analysis(tutorial,data,text=text,query=query,clusters=clusters,operation=operation)
    if data is not None:raise ValueError(f'T{tutorial} uses the supplied local MRI dataset; CSV upload is not applicable')
    return _run_mri_analysis(tutorial)
