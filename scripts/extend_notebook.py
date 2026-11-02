"""Idempotently extend the original notebook; retain its name and original tutorials."""
import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from her_core.analytics import TITLES

path=ROOT/'HER_Biomedical_Image_Tutorials_T1_T6.ipynb'
nb=json.loads(path.read_text(encoding='utf-8'))
cells=nb['cells']
def set_source(index,text): cells[index]['source']=text.splitlines(keepends=True)
def source(index): return ''.join(cells[index]['source'])

set_source(0,source(0).replace('T1–T6 Integrated Practical','T1–T15 Integrated Practical')+'\n\nT7–T15 extend this original MRI notebook using shared offline analytics in `her_core.analytics`. The filename is retained for compatibility.\n' if 'T7–T15 extend' not in source(0) else source(0))
set_source(2,source(2).replace('    _Path(r"C:/Users/Anushka/Downloads/HER_Biomedical_Image_Tutorials"),','    *_Path.cwd().parents,'))
set_source(4,'''from her_core.io import default_data_root
DATA_ROOT = default_data_root()
print("Dataset root:", DATA_ROOT)
print("Present:", DATA_ROOT.exists())
''')
set_source(14,source(14).replace('MODALITIES = ["flair", "t1", "t1ce", "t2", "seg"]','MODALITIES = ["flair", "t1ce", "t1", "t2", "seg"]'))
set_source(15,source(15).replace('    available_required = [m for m in required if m in modality_table.columns]','    modality_table = modality_table.reindex(columns=required, fill_value=0)\n    available_required = required'))
# Never overlay segmentation from a different patient on the first MRI.
set_source(23,'''from her_core.io import discover_cases, load_case_modality
labeled_example = next((c for c in discover_cases(DATA_ROOT) if {"flair", "seg"} <= set(c["modalities"])), None)
if labeled_example:
    overlay_mri = load_case_modality(labeled_example, "flair")[0]
    overlay_seg = load_case_modality(labeled_example, "seg")[0]
    if overlay_mri.shape != overlay_seg.shape:
        raise ValueError("Same-patient FLAIR and SEG dimensions differ")
    z = overlay_mri.shape[2] // 2
    plt.figure(figsize=(7, 7))
    plt.imshow(overlay_mri[:, :, z].T, cmap="gray", origin="lower")
    plt.imshow(np.ma.masked_where(overlay_seg[:, :, z].T == 0, overlay_seg[:, :, z].T), cmap="jet", alpha=.45, origin="lower")
    plt.title(f"Same-patient MRI + mask: {labeled_example['case_id']}")
    plt.axis("off")
    plt.show()
else:
    print("No paired FLAIR/segmentation case available")
''')
text=source(51)
marker='# Automatically locate the first example of each modality.'
if marker in text:
    text=text.split(marker)[0]+'''# Compare modalities from the same patient only.
from her_core.io import discover_cases
cases_for_display = discover_cases(DATA_ROOT)
if cases_for_display:
    same_case = max(cases_for_display, key=lambda c: len(c["modalities"]))
    plot_modalities({m: p for m, p in same_case["modalities"].items() if m != "seg"})
'''
    set_source(51,text)
set_source(40,source(40).replace('epochs=8,','epochs=int(os.environ.get("HER_CNN_EPOCHS", "8")),') )
if 'tf.keras.utils.set_random_seed(42)' not in source(33):
    set_source(33,source(33).replace('    import tensorflow as tf','    import tensorflow as tf\n    tf.keras.utils.set_random_seed(42)'))
set_source(56,source(56).replace('# Final Tutorial Summary','# T1–T6 Baseline Summary'))
# Avoid duplicate generated tutorial cells on repeated runs.
start=next((i for i,c in enumerate(cells) if c.get('metadata',{}).get('her_generated')),len(cells))
nb['cells']=cells[:start]
def add(kind,text):
    cell={'cell_type':kind,'metadata':{'her_generated':True},'source':text.splitlines(keepends=True)}
    if kind=='code':cell.update(execution_count=None,outputs=[])
    nb['cells'].append(cell)

details={
7:('Daily synthetic hospital operations','date, admissions, department, occupied_beds, capacity, cost','admissions','Validate complete consecutive daily records; aggregate by date. Fit scaled Ridge using trend and weekly calendar features, first 80% train and last 20% test. Report MAE/RMSE and training-mean baseline. Compare actual/predicted values and ceil(demand / 8 admissions per staff).'),
8:('Daily synthetic hospital operations','date, admissions, department, occupied_beds, capacity, cost','Descriptive operational KPIs','Reject impossible capacity/occupancy and malformed dates. Aggregate by date and department. Compute occupancy as total occupied beds / total capacity and cost per admission; interpret resource demand using measured KPIs.'),
9:('Daily synthetic hospital operations','date, admissions, department, occupied_beds, capacity, cost','Department comparison and calendar trend','Extend T8 with daily variance, linear trend slope, a Kruskal comparison and occupancy-ranked priorities. Kruskal requires independent observations; time dependence makes its p-value exploratory.'),
10:('Synthetic educational clinical-style text corpus','text, label','label','Normalize text, deduplicate identical documents and reject conflicting labels. Stratify a held-out split. Fit TF-IDF and logistic regression only on training texts. Report weighted precision/recall/F1, accuracy and confusion matrix. Fit K-means on training vectors and report cosine silhouette, sizes and terms.'),
11:('Synthetic educational clinical-style text corpus','text','Entity mentions, not diagnoses','Use a documented lexicon for conditions, medications, investigations plus measurement regular expressions. Return source, offsets and extraction method. A short negation-cue window is illustrative. Without annotated labels no honest precision/recall estimate is available.'),
12:('Synthetic educational clinical-style text corpus','text','Lexical relevance and supported abbreviation senses','Fit TF-IDF on documents, transform the query, rank cosine similarity. Explicit MS senses: multiple sclerosis / mitral stenosis; RA: rheumatoid arthritis / right atrium. Context cue counts resolve only unique positive evidence; otherwise return ambiguous or unsupported. No annotated relevance/sense benchmark is available.'),
13:('Synthetic adult behavioral measurements','age, height_m, weight_kg, activity_hours_week, sedentary_hours_day, sleep_hours','Exploratory numerical projections','Select adults with known age ≥18, validate units/ranges, median-impute and standardize numeric measurements. Report PCA explained/cumulative variance and 2D coordinates. Compare Isomap neighborhood embedding (at most 500 rows). Exploratory fit uses all observations and does not estimate predictive performance.'),
14:('Synthetic adult behavioral measurements','age, height_m, weight_kg, activity_hours_week, sedentary_hours_day, sleep_hours','Adult BMI and descriptive associations','Exclude unknown/minor ages. Height is meters, weight kilograms, activity hours/week and sedentary/sleep hours/day. Validate plausible ranges, retain missing values in descriptive counts and compute BMI only for observed height/weight. Inspect IQR flags, distributions and pairwise correlations.'),
15:('Synthetic adult behavioral measurements','age, height_m, weight_kg, activity_hours_week, sedentary_hours_day, sleep_hours','BMI','Use the T14 validation. Random 75/25 holdout; fit median imputation, scaling and regression only on training data. Predict BMI from age/activity/sedentary/sleep (exclude height and weight to avoid target leakage). Report MAE, RMSE, R² and coefficients per training SD. Pearson activity/BMI correlation uses independent pairs and Fisher 95% CI; associations do not establish causation.')}
add('markdown','''# T7–T15: Healthcare analytics extension

All examples run offline and are **deterministic synthetic demonstrations, never actual patient records**. No suitable non-MRI data or separate syllabus file was present. Topic names follow the supplied mission syllabus. Default data are generated locally with seed 42. Replace the demo with your own compatible, de-identified CSV using the paths below; missing external files are explicitly detected. The dashboard supports the same schemas and computations at http://127.0.0.1:8000.
''')
add('code','''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
from her_core.analytics import run_analysis, example_data

def tutorial_dataset(tutorial, csv_path=None):
    if csv_path is None:
        print("SOURCE: synthetic demonstration, seed 42, not real patient data")
        return None
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset absent: {path}; omit csv_path for labeled synthetic demo")
    return pd.read_csv(path)

def show_analysis(result):
    print(result["dataset"])
    display(pd.DataFrame([result["metrics"]]))
    for name, rows in result["tables"].items():
        print(name)
        display(pd.DataFrame(rows).head(12))
    for plot in result["plots"]:
        print(plot["title"])
        import base64
        display(Image(data=base64.b64decode(plot["png"])))
    for sentence in result["interpretation"] + result["warnings"]:
        print(sentence)
''')
for t,(dataset,columns,target,method) in details.items():
    add('markdown',f'''# T{t} — {TITLES[t-1]}

**Objective:** {TITLES[t-1]}.

**Dataset / source:** {dataset}; locally generated `example_data({t})`, seed 42. This is educational synthetic data with known generator relationships, not observed healthcare evidence. No download is required. For real analyses supply a compatible CSV from your approved public or de-identified source; the application does not independently verify provenance.

**Expected columns:** `{columns}`. {'Only date and admissions are required; department, paired occupied_beds/capacity and cost are optional, and their KPIs are omitted when absent.' if t in (7,8,9) else 'For T11/T12 only text is required; label is ignored.' if t in (11,12) else ''} **Target/output:** {target}.

**Preprocessing and method:** {method}

**Validation:** Empty inputs, missing required columns and malformed types produce readable errors. View missingness and actual output tables below. T11/T12 descriptive outputs lack labeled benchmarks; metrics are reported only when meaningful.

**Frontend demonstration:** Select T{t}, optionally download its labeled example CSV, set inputs and click Run analysis. Use the tables, plots and JSON export to inspect computed results.

**Reproduction:** The next cell imports the same computation as the API. Set `csv_path` in `tutorial_dataset` to use available external data. Interpretation is generated from actual results below.
''')
    options=', text="Diabetes on metformin 500 mg. MS after neurological assessment.", query="diabetes glucose"' if t in (10,11,12) else ''
    add('code',f'data_t{t} = tutorial_dataset({t})\nresult_t{t} = run_analysis({t}, data_t{t}{options})\nshow_analysis(result_t{t})\n')
add('markdown','# Complete T1–T15 summary\n\n| Tutorial | Practical |\n|---|---|\n'+''.join(f'| T{i} | {title} |\n' for i,title in enumerate(TITLES,1))+'\nMRI/CNN and all statistical examples are educational. Refer to SUBMISSION_REPORT.md for actual test status and limitations.\n')
# Clear stale outputs only in corrected original cells, keeping all other original outputs.
for i in (2,4,14,15,23,33,40,51):
    cells[i]['outputs']=[];cells[i]['execution_count']=None
for i,c in enumerate(nb['cells']):
    c.setdefault('id',f'her-cell-{i:03d}')
    if c['cell_type']=='code':ast.parse(''.join(c['source']),filename=f'cell_{i}')
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('Updated',path.name,'cells:',len(nb['cells']))
