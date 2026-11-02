# CNN integration in Image Analysis Studio

The studio now contains two clearly labeled sections in its existing dark theme. Its SIFT + RANSAC controls, source script and six-image results remain in place. A new **CNN Analysis** section is accessible through the studio's workflow links or directly at **http://127.0.0.1:8000/image-analysis#cnn-analysis**.

## Implementation and scope

- The studio controller sends **one `POST /api/cnn/demo` with `{}`** per completed run. It reuses the current backend without adding any endpoint, dataset, model or preprocessing pipeline.
- The existing endpoint automatically selects local cases with FLAIR and segmentation labels, prepares 32×32 inputs, trains the existing compact model for three epochs and evaluates separate validation patients. These are fixed settings of that endpoint. It does not load saved weights or accept a selected/uploaded single image. The studio explains this explicitly, and the SIFT selectors are labeled as applying only to SIFT.
- The dashboard's existing result components were extracted without changing their display logic into `frontend/results-renderer.js`. Both pages now call the same `HERResults.render` function. The component only renders data; it performs no requests or model training.
- The studio renders the actual input/reference-mask/classification comparisons and scores, measured accuracy/loss curves, confusion matrix, classification report, all validation predictions, history and patient split. It does not invent segmentation predictions for this classification network.
- A scoped CSS file applies the existing dark palette, typography, borders and cards to the shared components. Comparisons use three columns and charts two columns; both stack below 800px. Images preserve aspect ratio.
- Results remain hidden before a successful run. While running, the button is disabled, duplicate clicks are ignored and an accessible processing message is shown. API errors, missing labeled data, an already-running CNN, malformed/empty responses and backend disconnection show readable messages and clear stale results. SIFT and CNN have independent result containers.

No backend, original notebook, datasets, weights, SIFT source script, existing SIFT stylesheet or dashboard layout/style was changed. No Git commit or push was made. Running the CNN button explicitly trains the same small demo as before; merely opening the studio or following its anchor links does not train anything.

## Changed files

| File | Change |
|---|---|
| `frontend/image-analysis.html` | Add section links, CNN explanation/button/status/result container and shared/controller/style references; preserve SIFT sections and IDs. |
| `frontend/studio-cnn.js` — new | Small studio controller using the existing endpoint and shared renderer; loading, validation and error states. |
| `frontend/studio-cnn.css` — new | Scoped dark styling for the shared CNN results and responsive sections. |
| `frontend/results-renderer.js` — new | Extract the existing result/image/plot/table components so the studio and dashboard use one implementation. |
| `frontend/index.html` | Load the shared renderer before the existing dashboard script; no layout redesign. |
| `frontend/lab.js` | Replace the moved result-rendering functions with a call to the shared renderer; existing controls/export behavior retained. |
| `tests/frontend_integration.cjs` | Load the shared renderer before testing the existing dashboard. |
| `tests/studio_integration.cjs` — new | Exercise real studio scripts and HTTP for SIFT→CNN→SIFT, endpoint/schema, rendered outputs and failures. |
| `STUDIO_CNN_REPORT.md` — new | This scope, verification and browser checklist. |

## Tests and observed results

With the local project server running:

```powershell
node tests/studio_integration.cjs
node tests/frontend_integration.cjs
$env:LOKY_MAX_CPU_COUNT='4'
py -3 -m unittest discover -s tests -v
node --check frontend/results-renderer.js
node --check frontend/lab.js
node --check frontend/studio-cnn.js
node --check tests/studio_integration.cjs
git diff --check
```

- **Studio integration passed:** actual studio scripts called the existing CNN endpoint once with `{}` and displayed all returned preview images, class labels/scores, four PNG plots and five tables. No training call occurred on page load. Repeated clicks during processing did not issue another request. Live SIFT ran before and after CNN, each displaying all six images; neither workflow cleared the other's results. Loading states, API 409/503 responses, empty predictions, invalid JSON and disconnection were checked.
- **Dashboard integration passed:** all 15 tutorial flows, all 46 plot data URLs/captions, annotations and cluster cards, plus the previous CNN gallery/plots/tables and error/upload checks remained functional with the shared renderer.
- **31 Python tests passed in 44.476 seconds**, including existing CNN data/prediction/PNG/history/evaluation checks and all analysis/API/visualization regressions. These tests use the existing small CNN experiment; no full-size notebook training was run.
- JavaScript syntax checks and `git diff --check` passed. The studio HTML and all three new shared/studio static assets were served successfully over HTTP with the appropriate content types.

The JavaScript integration tests use actual frontend scripts, real local HTTP and a minimal DOM adapter. They are **not browser automation**. Browser automation tools are unavailable in this session; pixel layout, native anchor scrolling, responsive rendering and actual image-load events require the manual checks below. CSS breakpoints/aspect-ratio rules were checked in source, not claimed as visually browser-tested.

## Exact browser verification steps

1. If necessary, run `.\run_server.bat` from the project directory. Open **http://127.0.0.1:8000/image-analysis** and press **Ctrl+F5** to reload changed assets.
2. In **SIFT + RANSAC**, leave a valid case/modality selected and use two different nearby central slices. Click **Run SIFT + RANSAC Analysis**. Verify both original images, both keypoint images, feature matches, RANSAC visualization and statistics appear as before.
3. Click the **CNN Analysis** section link, or open **/image-analysis#cnn-analysis**. Confirm the page states that CNN uses automatically selected local labeled cases and fixed demo training settings. Changing the SIFT selectors must not imply changing the CNN input.
4. Click **Train & evaluate CNN** once. Verify the processing message and disabled button. Wait for completion; no upload or synthetic-image substitution is involved.
5. Inspect the **CNN Image Results** cards: actual preprocessed input, reference mask when available, and CNN classification on the input with the returned class score. This classifier must not show a fabricated predicted segmentation mask.
6. Inspect training/validation accuracy and loss, confusion matrix, and the existing composite comparison. Expand **predictions**, **training history**, **classification report**, **confusion matrix** and **split** to check their actual values.
7. Return to the SIFT section and run it again. Confirm its original controls/results still work and the CNN output remains present. A fresh CNN click intentionally trains another compact demo; browsing sections does not.
8. Narrow the window below 800px or use a phone viewport. Confirm CNN comparison cards and charts stack, tables scroll horizontally if necessary, and images retain their proportions. This is a manual responsive check, not an automated pass.
9. For an error check, stop the server and click CNN again. Confirm a connection error replaces the result area and the button becomes available for retry. Restart the server afterward.
10. Open **http://127.0.0.1:8000/**, hard-refresh and run a tutorial plus its existing CNN button to confirm the main dashboard appearance and interactions remain unchanged.
