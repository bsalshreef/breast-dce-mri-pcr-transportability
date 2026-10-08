# Reproducibility and Provenance



## Scope



This document describes the reproducibility status of the analysis code supporting the study:



\*\*Cross-Cohort Transportability, Calibration, and Incremental Value of Breast DCE-MRI Radiomics for Predicting Pathologic Complete Response\*\*



The repository was prepared from the locked analysis workflow after completion of the primary statistical analyses. Its purpose is to provide an auditable code release while clearly distinguishing verified components from historical intermediate artifacts for which the original generation scripts were not recovered.



## Verified analysis components



The repository contains 26 audited Python scripts organized into five functional groups:



\- `code/preprocessing/`

\- `code/radiomics/`

\- `code/modeling/`

\- `code/validation/`

\- `code/figures/`



Before repository reorganization, SHA-256 hashes were recorded for all 26 scripts.



After the scripts were reorganized into the repository structure, all files were rehashed and compared with the baseline.



The integrity audit returned:



```text

Baseline files: 26

Current files: 26

Missing: 0

Extra: 0

Changed: 0

INTEGRITY STATUS: PASS

Thus, repository reorganization did not alter the audited script contents.

## Locked development cohort

The primary I-SPY2 development and internal-validation cohort contained 982 participants, including 316 with pCR (32.2%).

Five-fold stratified cross-validation was locked before the final comparative analyses.

## Complete-case sensitivity analysis

Three I-SPY2 participants had missing age in the source metadata.

The primary analysis used mean imputation based on the 979 participants with observed age.

For the complete-case analysis, the three cases were identified programmatically from missing age in the source metadata rather than from hard-coded patient identifiers.

An equivalence audit confirmed that this criterion reproduced the historical three-case exclusion exactly.

The resulting complete-case cohort contained N = 979 while preserving the original fold assignments.

## M4 model verification

The combined M4 model retained four clinical predictors:

- age;
- HR status;
- HER2 status;
- log-transformed tumor volume.

Radiomic feature selection was applied only to radiomic predictors.

The nested selection grid used SelectKBest with k = 10, 20, or 40 and logistic-regression C = 0.01, 0.1, 1, or 10, with inner five-fold stratified cross-validation and AUROC as the selection metric.

The selected outer-fold k values were:

20, 10, 20, 40, 20

The selected outer-fold C values were:

1.0, 0.1, 10, 10, 0.1

The modal k was 20. C = 0.1 and C = 10 were tied in frequency. The deterministic tie-breaking behavior implemented in the locked pipeline selected C = 0.1 for the final development refit.

The final M4 model therefore contained four clinical predictors and 20 selected radiomic predictors.

## External validation

### I-SPY1

The locked external-validation cohort contained 167 participants, including 47 with pCR (28.1%).

The frozen M4 model was applied without external refitting, feature reselection, preprocessing estimation, hyperparameter tuning, or recalibration.

The corrected external M4 predictions produced:

- AUROC = 0.758333
- AUPRC = 0.540461
- Brier score = 0.172306

The corresponding transportability analysis reproduced the locked manuscript results.

### Duke

The Duke clinical-model external-validation cohort contained 297 participants, including 62 with pCR (20.9%).

M4 was intentionally not applied to Duke because the available segmentation semantics were not considered compatible with the tumor masks used for the locked radiomics workflow.

The M2 clinical model was therefore the externally evaluated model for Duke.

## Historical provenance gaps

The following historical generation scripts were not recovered during repository preparation:

1. the script originally used to construct `ISPY1_EXTERNAL_READY.csv`;
2. the script originally used to generate the I-SPY1 M2 external prediction file;
3. the script originally used to generate the Duke M2 external prediction file.

These missing scripts are treated as provenance limitations.

No replacement historical provenance has been fabricated or inferred.

The repository therefore should not currently be described as a fully self-contained, raw-data-to-final-results reproduction package.

The available downstream validation scripts and locked aggregate results remain auditable, but reproduction of these three historical intermediate artifacts requires either recovery of the original generation code or a separately documented reconstruction.

## Patient-level artifacts

The public repository is intentionally designed not to contain:

- patient-level clinical metadata;
- patient-level model predictions;
- radiomic feature matrices;
- MRI data;
- segmentation masks;
- serialized fitted models;
- patient identifiers.

Such files are excluded from version control.

## Software environment

The repository-preparation environment was verified as:

- Python 3.11.9
- NumPy 2.4.6
- pandas 3.0.6
- scikit-learn 1.9.1
- Matplotlib 3.11.2
- joblib 1.6.0
- PyRadiomics 3.0.1
- SimpleITK 2.5.6

`pip check` returned `No broken requirements found.`

These versions represent the audited repository-preparation environment. Except where independently documented, they should not be interpreted as proof that every historical analysis artifact was originally generated under exactly these package versions.

## Reproducibility classification

Current repository status:

**Audited analysis-code release with documented provenance limitations.**

The repository supports inspection of the locked modeling, validation, bootstrap, transportability, sensitivity-analysis, and figure-generation logic available from the study workflow.

It does not currently claim complete end-to-end reproduction from raw imaging data to every final manuscript result.

## Future provenance completion

If the missing historical generation scripts are recovered, they can be independently audited and added in a future version.

Alternatively, reconstructed scripts may be developed, but any such scripts should be explicitly labeled as reconstructed implementations rather than represented as the original historical analysis code.
