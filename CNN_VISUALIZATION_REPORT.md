# CNN website visualization integration

## Cause reproduced

The live website on port 8000 was running a stale Python/uvicorn process. Its real `/api/cnn/demo` response was HTTP 200 with 1,866 bytes and `plots: []`, although the current source already generated four plots. Restarting that confirmed project process was necessary to load the existing CNN plot code. The updated process was then tested on the normal port 8000.

The current frontend also treated CNN images as generic plots: it put a tall four-row montage after the curves, metrics and explanatory text. It had no prominent per-image comparison gallery or image-load failure handling. In contrast, the SIFT studio has explicit labeled image cards. This fix adds CNN-specific cards and keeps the existing generic plot renderer for its measured training/evaluation outputs.

The original notebook was inspected, particularly cells 42, 46 and 48: accuracy/loss, confusion matrix, and input/mask/classification comparisons. It was not modified. The network is a binary classifier, not a segmentation model. Its prediction image is the actual model input displayed with the computed class and score; no predicted mask is invented.

## Changes

- `backend/tutorials.py`: CNN-only additive response fields `model_task: classification` and `previews`. Each preview has the actual case/slice, existing prediction/score, compact PNG data URLs for the prepared model input and optional ground-truth mask. PNG orientation matches the notebook's transpose/lower-origin display. Existing plots, predictions, training history, training procedure and metrics are preserved.
- `frontend/lab.js`: CNN-only results section above metrics, showing Input Image, Ground Truth and CNN Prediction side by side. Scores come from the response. Empty/stale API responses explain that the backend needs restarting; missing image/mask and image-decode errors receive readable messages. An existing composite preview is a fallback for responses without individual images. The gallery is created only after the CNN request succeeds, and cleared on a new run or API failure.
- `frontend/lab.css`: scoped CNN card styles; three columns on desktop, stacked cards below 850px; images use their natural aspect ratio with `width:100%; height:auto; object-fit:contain`.
- `tests/test_cnn_output.py`: verifies four individual previews, PNG decoding, 32×32 dimensions and exact pixel equality with real prepared model inputs/reference masks; no predicted-mask field.
- `tests/frontend_integration.cjs`: verifies prominent gallery placement, all comparison images/scores, four existing plots/five tables, missing mask/image decode failure, stale/empty responses and API errors in addition to existing tutorial flows.
- `CNN_VISUALIZATION_REPORT.md`: this report.

No model architecture, weights, datasets, original notebook, SIFT frontend/backend implementation or other tutorial computations were changed in this follow-up. No Git push or commit was made.

## Verification

`py -3 -m unittest discover -s tests -v`: **22 tests passed** in 21.069 seconds, including 9 CNN checks and 13 existing analysis/API regression tests. Tests ran actual training/inference on a small labeled sample and checked metrics, patient splits, serialized PNG plots and exact preview pixel values. The model still uses its existing compact 32×32, three-epoch web experiment, not the notebook's full-size experiment.

`node --check frontend/lab.js` and `node tests/frontend_integration.cjs` passed. The latter exercised actual website JavaScript against the live backend using a minimal DOM adapter: all 15 tutorial flows plus CNN gallery, training/evaluation plots, tables and failure states. This is not browser layout verification. A direct live CNN request returned HTTP 200 with 205,691 bytes matching Content-Length, four previews and four plots. All preview input/mask images were compact PNG data URLs. The original live SIFT/RANSAC endpoint also returned HTTP 200, 44 good matches and 35 inliers. Python compilation, test JavaScript syntax and `git diff --check` passed.

Browser automation is not available in this session. Native browser layout, responsive rendering and actual image `load` events therefore require the manual check below; DOM tests verify generated image elements/data URLs and simulate the error handler. The notebook was inspected, not re-executed or edited during this fix.

## Manual browser verification

1. If the project server is not running, run `.\run_server.bat`. If an older server is still running, stop it with Ctrl+C first and restart it. This matters because the startup script does not auto-reload backend changes.
2. Open **http://127.0.0.1:8000/#T5**, then press **Ctrl+F5** to reload the updated JavaScript/CSS.
3. Click **Run compact CNN training & inference**, not the ordinary Run analysis button (which runs SIFT/RANSAC).
4. After completion, verify **CNN Image Results** appears above the metrics. It shows four cases/slices sampled from validation, each with **Input Image (preprocessed model input)**, **Ground Truth**, and **CNN Prediction** with predicted class and model score. The third card reuses the input because this classifier does not produce a mask.
5. Scroll to **Training History & Evaluation** for the measured accuracy/loss curves, confusion matrix and existing composite comparison. Expand the prediction/report/history tables to inspect actual outputs.
6. Narrow the window below 850px (or use a mobile viewport). Each sample's three cards should stack, preserving image proportions. Restore the desktop width to see side-by-side cards.
7. In browser DevTools Network, inspect `POST /api/cnn/demo`: expect four `previews` with `data:image/png;base64,...` images, four `plots`, and `model_task: classification`. If `plots` is empty or `previews` absent, an older backend is still serving that port; restart the server.
8. Open `/image-analysis` and run nearby-slice SIFT/RANSAC to confirm its existing visuals remain available.
