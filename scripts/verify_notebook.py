"""Execute every notebook cell with a one-epoch CNN verification budget."""
from pathlib import Path
import json
import os
import sys
import time
os.environ['HER_CNN_EPOCHS']='1'
os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['LOKY_MAX_CPU_COUNT']='4'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
import nbformat
from nbclient import NotebookClient
nb=nbformat.read(ROOT/'HER_Biomedical_Image_Tutorials_T1_T6.ipynb',as_version=4)
started=time.time()
outdir=ROOT/'artifacts';outdir.mkdir(exist_ok=True)
try:
    NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},allow_errors=False).execute()
    result={'status':'passed','cells':len(nb.cells),'executed_code_cells':sum(c.cell_type=='code' for c in nb.cells),'cnn_epochs':1,'elapsed_seconds':round(time.time()-started,1)}
except Exception as exc:
    result={'status':'failed','error':str(exc),'elapsed_seconds':round(time.time()-started,1)}
finally:
    nbformat.write(nb,outdir/'executed_notebook.ipynb')
    (outdir/'notebook_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if result['status']!='passed':sys.exit(1)
