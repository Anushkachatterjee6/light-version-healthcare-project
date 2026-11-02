# Data sources and limitations

## Local MRI source of truth

T1–T9 and T13–T15 read the existing `BraTS2020/` folders. The supplied set has eight MRI cases: all have FLAIR and T2; six additionally have T1, T1ce and a segmentation file. The application derives only file inventory, modality availability, FLAIR intensity features, volume depth, and nonzero segmentation burden from those assets.

A supplied segmentation containing positive voxels supports the label **Tumour present** for that case. A case without a segmentation is **Unknown — no segmentation**, not “no tumour.” The project has no verified tumour-negative cohort, age, grade, anatomical location, clinical report, or clinical outcome metadata. It does not manufacture these values. All case splits in the CNN and T15 evaluation are case/patient-level.

## Text tutorials

T10–T12 start with compact **synthetic educational brain-tumour report templates**. They are visibly labeled synthetic in API responses and the UI; they are not patient reports. T10 requires `text,label`; T11 and T12 require `text`. UTF-8 CSV uploads are restricted to T10–T12 (2 MB, 5,000 rows, 50 columns), parsed only in memory, and marked user-supplied/provenance unverified.

No operations, admissions, occupancy, adult-weight, BMI, behavioral, or unrelated clinical datasets are used by the 15 tutorial workflow.
