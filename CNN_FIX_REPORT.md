# CNN output investigation and fix

The only production code changed in this follow-up is `cnn_demo()` in `backend/tutorials.py`. Existing uncommitted project work was preserved. No notebook, preprocessing helper, model architecture, MRI/SIFT implementation, other tutorial computation, dataset or weight file was modified. No Git commit or push was made.

## Reproduction and root cause

Before the fix, `POST /api/cnn/demo` completed successfully: HTTP 200, **1,866 response bytes**, matching `Content-Length: 1866`, 12 validation predictions, one patient-split table and four scalar metrics. Its response explicitly contained `plots: []`. The complete response is recorded in ignored `artifacts/cnn_before.json`.

The traced workflow was:

1. T5's **Run compact CNN training & inference** button calls `analyze(true)` in `frontend/lab.js`. The ordinary **Run analysis** button separately calls the T5 SIFT/RANSAC endpoint.
2. `analyze(true)` posts `{}` to `/api/cnn/demo`, parses JSON and calls the existing generic `render(result)`.
3. The endpoint discovers labeled FLAIR/segmentation cases, uses the existing cleaning, center crop, min-max scaling and resizing, and separates patients before building train/validation arrays.
4. It builds a fresh model using `make_tumor_cnn((32,32,1))`; it does not load saved weights. Previously it ran a single `train_on_batch`, discarded the returned training measurements and performed validation inference.
5. Its response builder returned only numeric predictions, patient split and headline metrics. It never constructed input images, a training history, curves, a confusion matrix or a classification report, despite importing the confusion-matrix helper.
6. The frontend rendered everything it received. Its plot loop had nothing to render; predictions and split were in expandable tables. No frontend truncation, missing image conversion or post-prediction exception explained the reproduced successful response.

This was an incomplete backend response, not a transport truncation or a broken model prediction. An already-running server was found on port 8000 during verification; it was left alone. The fixed code was tested on an isolated port 8001. Restart an existing server to load the fix.

## Outputs now returned

The bounded web experiment now trains **three actual epochs**, one full training batch per epoch, using the same architecture, 32×32 input size and at most eight slices per labeled case. It measures each complete train/validation split after the epoch, with metrics reset between splits and dropout disabled during measurement. Validation records never enter weight updates.

It returns the existing response keys plus:

- `workflow: training_then_validation_inference` and training size/epoch/input-size metrics.
- Four PNG plots in the existing `plots[].png` format: train/validation accuracy, train/validation loss, validation confusion matrix, and up to four evenly spaced input/mask/prediction previews. Preview selection does not depend on prediction success.
- All validation predictions, preserving the original numeric `actual`, `predicted` and `probability` fields. Added fields name both classes, provide the complementary no-tumor score, predicted-class score and correctness.
- Three-row `training_history`, two-class `confusion_matrix` and per-class/macro/weighted `classification_report` tables, plus the existing patient split.
- Visible explanations of training versus inference, score meaning, evaluation scope and limitations. Metrics require ground-truth labels; missing labeled data produce HTTP 503 instead of fabricated results. A post-prediction plotting failure produces HTTP 500 rather than a partial successful response. The CNN lock is released even if model cleanup fails.

The existing frontend can display every added output without changing production HTML/JavaScript. Expand the tables below the plots to view all predictions and the classification report. JSON export retains the complete response.

## Comparison with the notebook

The unchanged T1–T6 notebook's CNN cells 38–49 build the same architecture at 128×128, use patient-separated data, train up to eight epochs, plot accuracy/loss, evaluate a classification report/confusion matrix and compare FLAIR inputs, masks and predictions. The web endpoint now exposes those output categories using a deliberately smaller experiment. It does not claim the notebook's training budget, sample count or results.

Scores are actual sigmoid model outputs. `probability` means P(Tumor Present); the no-tumor score is its complement, and the predicted-class score selects the appropriate one. These are not calibrated clinical confidence estimates. Labels describe tumor-pixel presence in the prepared segmentation mask, not a clinical diagnosis. Ground-truth masks are used for labels/display, never as CNN inputs. Validation metrics summarize slices from held-out patients; such a small sample does not establish clinical performance.

## Verification performed

```powershell
py -3 -m unittest discover -s tests -p test_cnn_output.py -v
py -3 -m unittest discover -s tests -p test_analytics.py -v
node --check frontend/lab.js
node --check tests/frontend_integration.cjs
py -3 -m py_compile backend/tutorials.py tests/test_cnn_output.py
```

- **8 CNN regression tests passed** in 10.865 seconds. A real local labeled-data run used 25 training and 12 validation slices from disjoint patient groups, 32×32 inputs and three epochs. Tests checked complete JSON length/serialization, all required fields, PNG decoding/dimensions, actual-versus-predicted labels, both class scores, score-to-class consistency, per-epoch finite metrics, independently recomputed validation cross-entropy, classification-report/confusion-matrix agreement with every validation label, missing-label errors, concurrency and forced failure after inference.
- **13 existing analysis/API tests passed** in 6.461 seconds, covering T7–T15 computations/uploads, metadata/health, malformed inputs, MRI preprocessing and patient splitting. Those tutorial implementations were not changed.
- Syntax/compilation checks and `git diff --check` passed.
- A real uvicorn server ran on 127.0.0.1:8001. With it running, the actual frontend JavaScript was exercised using:

```powershell
$env:HER_TEST_BASE_URL='http://127.0.0.1:8001'
node tests/frontend_integration.cjs
```

The test exercises all 15 existing tutorial flows, the distinct CNN button and its four PNG plots/five tables, CSV upload, invalid-file rejection and backend-disconnect handling. It uses a minimal DOM adapter and actual HTTP, not browser automation. The generated CNN input/mask/prediction PNG was also visually inspected directly. Browser layout and native file-picker/download behavior were not tested; production frontend code was unchanged. The full notebook was inspected for comparison but was not modified or rerun during this focused fix.

The successful fixed response is recorded in ignored `artifacts/cnn_after.json`. A direct live HTTP request also returned a complete **200,400-byte** JSON body matching `Content-Length: 200400`, with 12 validation predictions, four plots, three history rows and four classification-report rows. Its measured validation confusion matrix was `[[4, 2], [0, 6]]`; scores and metrics may vary with numerical execution. No assertions depend on achieving a particular accuracy. A live request to the original SIFT/RANSAC endpoint also passed, returning 44 good matches and 35 inliers.

## Changed files and reproduction

- `backend/tutorials.py`: CNN training measurement and response construction only.
- `tests/test_cnn_output.py`: focused CNN regression checks.
- `tests/frontend_integration.cjs`: explicit CNN-button/result coverage and optional test-server URL.
- `README.md`, `SUBMISSION_REPORT.md`: update the compact CNN description.
- `CNN_FIX_REPORT.md`: this investigation record.

To see the complete output, stop and restart your existing project server using `.\run_server.bat`, then open **http://127.0.0.1:8000/#T5**. Click **Run compact CNN training & inference**, wait for completion and scroll through the four plots. Expand **predictions**, **training history**, **confusion matrix**, **classification report** and **split**. Use **Export result JSON** to inspect every returned field. The ordinary T5 Run analysis button continues to demonstrate SIFT/RANSAC.
