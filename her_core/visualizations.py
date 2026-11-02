"""Small PNG bridges for already computed tutorial outputs; no model fitting."""
from io import BytesIO
import base64
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def chart(title, draw, *, xlabel='', ylabel='', caption='', size=(8, 4)):
    # Import lazily to share the existing renderer's lock without import cycles.
    from .analytics import PLOT_LOCK
    with PLOT_LOCK:
        fig=Figure(figsize=size,layout='constrained')
        ax=fig.subplots()
        draw(ax)
        if ax in fig.axes:ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
        buf=BytesIO();fig.savefig(buf,format='png',dpi=100)
        return {'title':title,'png':base64.b64encode(buf.getvalue()).decode('ascii'),
                'caption':caption,'x_label':xlabel,'y_label':ylabel}


def bars(title, labels, values, *, ylabel, caption=''):
    labels=list(map(str,labels)); values=np.asarray(values,dtype=float)
    def draw(ax):
        ax.bar(np.arange(len(labels)),values,color='#277a69')
        ax.set_xticks(np.arange(len(labels)),labels,rotation=30,ha='right')
    return chart(title,draw,ylabel=ylabel,caption=caption)


def heatmap(title, matrix, rows, columns, *, caption='', limits=None):
    matrix=np.asarray(matrix,dtype=float)
    def draw(ax):
        bounds={} if limits is None else {'vmin':limits[0],'vmax':limits[1]}
        image=ax.imshow(np.ma.masked_invalid(matrix),cmap='coolwarm' if limits==(-1,1) else 'Blues',aspect='auto',**bounds)
        ax.figure.colorbar(image,ax=ax,shrink=.8)
        ax.set_xticks(range(len(columns)),list(map(str,columns)),rotation=35,ha='right')
        ax.set_yticks(range(len(rows)),list(map(str,rows)))
        if matrix.size<=200:
            for (i,j),value in np.ndenumerate(matrix):
                ax.text(j,i,'—' if not np.isfinite(value) else f'{value:.2g}',ha='center',va='center',fontsize=9)
    return chart(title,draw,caption=caption,size=(8,max(4,min(9,.35*len(rows)+2))))


def behavioral_distributions(df, features):
    units={'age':'Age (years)','activity_hours_week':'Activity (hours/week)',
           'sedentary_hours_day':'Sedentary (hours/day)','sleep_hours':'Sleep (hours/day)'}
    def draw(ax):
        fig=ax.figure;ax.remove()
        for panel,feature in enumerate(features,1):
            sub=fig.add_subplot(2,2,panel)
            sub.hist(df[feature].dropna(),bins=15,color='#277a69',alpha=.85)
            sub.set(xlabel=units[feature],ylabel='Adult observations',title=units[feature])
        fig.suptitle('Adult behavioral feature distributions')
    return chart('Adult behavioral feature distributions',draw,size=(10,7),
                 caption='Observed adult values only; missing values are not imputed for these distributions.')


def enhancement_comparison(image):
    # Same skimage methods and parameters as notebook cells 19 and 21.
    from skimage import exposure, restoration
    from .preprocess import minmax_normalize
    normalized=minmax_normalize(image)
    return exposure.equalize_adapthist(normalized,clip_limit=.03), restoration.denoise_bilateral(
        normalized,sigma_color=.05,sigma_spatial=3,channel_axis=None)


def segmentation_overlay(case, volume=None):
    from .io import load_case_modality
    if volume is None: volume=load_case_modality(case,'flair')[0]
    mask=load_case_modality(case,'seg')[0]
    if volume.shape!=mask.shape: raise ValueError('FLAIR and reference mask dimensions differ')
    z=volume.shape[2]//2
    def draw(ax):
        ax.imshow(volume[:,:,z].T,cmap='gray',origin='lower')
        ax.imshow(np.ma.masked_where(mask[:,:,z].T==0,mask[:,:,z].T),cmap='jet',alpha=.45,origin='lower')
        ax.axis('off')
    return chart('MRI + provided segmentation overlay',draw,
        caption=f"Same patient {case['case_id']}, axial slice {z}. Provided mask labels; not a predicted segmentation.")


def volume_projection(volume):
    values=volume[volume>0]
    if not values.size: return None
    threshold=float(np.percentile(values,90))
    stride=3
    coords=np.argwhere(volume[::stride,::stride,::stride]>threshold)*stride
    if not len(coords): return None
    # Uniform cap prevents slow rendering and huge response images.
    coords=coords[::max(1,int(np.ceil(len(coords)/12000)))][:12000]
    def draw(ax):
        fig=ax.figure;ax.remove();sub=fig.add_subplot(111,projection='3d')
        sub.scatter(coords[:,0],coords[:,1],coords[:,2],s=1,alpha=.3,color='#277a69')
        sub.set(xlabel='X (voxel)',ylabel='Y (voxel)',zlabel='Z (voxel)',title='3D MRI intensity threshold')
    plot=chart('3D MRI intensity threshold',draw,size=(7,5),
        caption=f'Above the 90th percentile of positive intensities ({threshold:.3g}); spatial stride 3, at most 12,000 displayed points. Voxel coordinates, not millimeters or tumor predictions.')
    plot['points_displayed']=len(coords)
    return plot


@lru_cache(maxsize=512)
def _file_digest(path, size, modified_ns):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path):
    path=Path(path);stat=path.stat()
    return _file_digest(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
