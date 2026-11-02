# Academic correctness audit — T1–T15

This review compares the current code and returned data to the stated tutorial topics. It does not treat HTTP success as academic validation. Browser screenshots are unavailable because browser automation is not available.

| Tutorial | Academic assessment | Status |
|---|---|---|
| T1–T3 | Local NIfTI inventory, cleaning and missing-data checks match introductory data-management objectives; outputs are derived from files. | PASS |
| T4 | Enhancement/restoration visual comparisons are appropriate educational image-processing outputs. | PASS |
| T5 | SIFT/RANSAC metrics and compact CNN outputs are appropriate, provided CNN results remain clearly bounded to the local split. | PASS |
| T6 | Supplied-mask visualization is correct; no unsupported predicted segmentation is claimed. | PASS |
| T7 | A reference-availability queue is not a predictive model. A verified outcome label and patient-level train/test data are required. | PARTIAL |
| T8 | Inventory and availability summaries support descriptive planning. | PASS |
| T9 | Correlations and exploratory Mann–Whitney output extend T8, but tiny-sample inference is explicitly limited. | PASS |
| T10 | Separate TF-IDF/Logistic Regression classification and TF-IDF/K-Means clustering now have independent computation and outputs. Synthetic corpus results are not clinical validation. | PASS |
| T11 | Returned offsets are generated from actual regex matches; rule-based extraction is not NER evaluation. | PARTIAL |
| T12 | TF-IDF cosine scores are sorted descending and are relevance, not confidence; cue disambiguation is heuristic. | PASS |
| T13 | Standardization precedes PCA/t-SNE; variance and coordinates originate from the transformed matrix. Small-sample t-SNE is exploratory. | PASS |
| T14 | MRI-burden correlation does not satisfy an adult-weight-factors syllabus without a documented adaptation and relevant variables. | PARTIAL |
| T15 | Held-out Ridge regression is technically a burden-prediction demonstration, but not adult-weight complexity analysis; sample is too small for substantive evaluation. | PARTIAL |

## Required data for incomplete objectives

- T7: a scientifically defined, verified target and sufficient independent labeled cases.
- T11: annotated clinical-text corpus for precision/recall/F1 NER evaluation.
- T14–T15: the original adult-weight factor dataset, or a formally approved MRI-specific adaptation with matching objectives.
