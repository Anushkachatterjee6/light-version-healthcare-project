# Tutorial output audit — 2 November 2026

This audit is evidence-based on the current FastAPI computations and the DOM integration regression. Browser screenshots could not be captured because browser automation is unavailable in this environment.

| Tutorial | Objective / actual method | Data and returned outputs | Status / limitation |
|---|---|---|---|
| T1 | NIfTI inventory | Local MRI case/modality table and plots | PASS — rendered by regression harness |
| T2 | Cleaning/normalization | Local MRI slice comparisons | PASS |
| T3 | Missing/duplicate review | Local file inventory/fingerprints | PASS |
| T4 | Image processing | Enhancement/restoration outputs | PASS |
| T5 | SIFT/RANSAC and CNN | Local slice matches; compact CNN metrics/plots | PASS — CNN is classification, not segmentation |
| T6 | Visualization/reference segmentation | Supplied modalities/masks only | PASS |
| T7 | Predictive-model topic | Reference-availability review queue | PARTIAL — no scientifically supported outcome target or negative cohort for a predictive model |
| T8 | Strategic planning I | Inventory/modality/annotation summaries | PASS — descriptive only |
| T9 | Strategic planning II | Feature correlation and exploratory test | PASS — tiny sample limits inference |
| T10 | Text classification and clustering | Separate TF-IDF Logistic Regression or TF-IDF K-Means paths | PASS — synthetic teaching corpus only |
| T11 | Entity extraction | Lexicon/regex entities and offsets | PASS — rule based, not validated NER |
| T12 | Ranking/disambiguation | TF-IDF cosine ranking and explicit cue logic | PASS — relevance is not clinical confidence |
| T13 | Dimensionality reduction | Standardized MRI features, PCA/t-SNE | PASS — exploratory small-sample display |
| T14 | Factor complexity adaptation | MRI-feature/reference-burden associations | PARTIAL — adult-weight syllabus topic is explicitly adapted; no clinical variables are available |
| T15 | Multivariable factor adaptation | Held-out Ridge prediction of mask-derived burden | PARTIAL — very small labeled-case split; not diagnosis prediction |

## T10 independent validation

- Classification: TF-IDF + Logistic Regression, stratified held-out synthetic texts; returns predictions, accuracy, precision, recall, F1, confusion matrix, classification report and a confusion-matrix visualization.
- Clustering: TF-IDF + K-Means; returns assignments, representative terms, cluster-size plot, PCA coordinates/plot and silhouette score where valid.
- The two paths are opened from separate visible **TEXT CLASSIFICATION** and **TEXT CLUSTERING** buttons and do not call each other.

## Corrections made

- T10 no longer executes K-Means as part of a classification request.
- T10 frontend exposes two independent workflow buttons, dedicated run labels, and a Back-to-T10-workflows action.
- All reported text results are labeled synthetic and non-clinical.
