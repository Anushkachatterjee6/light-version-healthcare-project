# Main application architecture audit

Audit date: 2 November 2026. This is a scope and dependency record, not a completion claim.

## Entry points and separation

| Area | Frontend | Application API/service | Tutorial dependency |
|---|---|---|---|
| Main dashboard | `app.html`, `app.js` | `/api/application-summary`, local MRI inventory | No |
| Dataset Intelligence | `module.html`, `module.js` | `/api/app/dataset/*`, `her_core.platform.dataset_profile` | No |
| MRI Research Studio | `module.html`, `module.js` | `/api/app/mri/*`, local NIfTI image service | No |
| Preprocessing Laboratory | `module.html`, `module.js` | `/api/app/preprocessing/*`, `her_core.preprocess` | No |
| Radiomics Lab | `module.html`, `module.js` | `/api/app/radiomics/*`, `her_core.platform.radiomics` | No |
| Region Quantification | `module.html`, `module.js` | `/api/app/quantification/*`, supplied segmentation only | No |
| Image Processing Lab | `module.html`, `module.js` | `/api/app/image-processing/*`, `her_core.sift_ransac` | No |
| Segmentation review | `module.html`, `module.js` | MRI slice + `/api/app/quantification/*` | No |
| Clinical NLP | `module.html`, `module.js` | `/api/app/clinical-nlp/*`, separate synthetic-text workflows | No |
| Research Discovery | `module.html`, `module.js` | `/api/app/research/discovery`, local FLAIR summaries + PCA | No |
| Experiment Manager | `module.html`, `module.js` | `/api/app/experiments`, `app_data/experiments.json` | No |
| Research Reports | `module.html`, `module.js` | `/api/app/reports/*`, saved experiment HTML export | No |
| CNN experiment/evaluation | `module.html`, `module.js` | `/api/cnn/demo`, reusable `her_core.cnn` | No `/api/tutorials/*` route dependency |
| Tutorial Lab | `index.html`, `lab.js` | `/api/tutorials/*` | Yes, by design |

The main application JavaScript contains no `/api/tutorials/*` calls. Tutorial Lab remains the only frontend that uses its tutorial APIs.

## Actual local data and reusable implementation

- Backend entry point: `backend.main:app`.
- Local data discovery and NIfTI loading: `her_core.io`.
- Reusable image processing: `her_core.preprocess` and `her_core.sift_ransac`.
- CNN data/model helpers: `her_core.cnn`.
- Application-level independent research functions: `her_core.platform`.
- Dataset found during verification: 8 local cases; supplied reference masks are only used where present. No negative cohort is invented.

## Startup finding and fix

The earlier apparent route failure was an old Uvicorn process serving stale imports on port 8000. The current `run_server.bat` checks the Python launcher, imports required dependencies, detects an occupied port, prints the URLs, and starts `backend.main:app` from the repository directory. After stopping only the identified stale listener and launching from this workspace, `/api/app/dataset/cases` returned HTTP 200 and the application APIs were available.

## Remaining product scope

PDF export, threshold-sensitivity analysis, registration/alignment output beyond the SIFT/RANSAC transformation evidence, experiment comparison UI, and a richer saved-result schema for CNN experiments remain incomplete. HTML reports and JSON-file persistence are implemented; synthetic clinical text remains explicitly non-clinical.
