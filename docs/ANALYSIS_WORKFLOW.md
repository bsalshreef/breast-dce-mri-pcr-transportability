# Analysis Workflow



## Purpose



This document summarizes the logical analysis workflow supporting the study:



\*\*Cross-Cohort Transportability, Calibration, and Incremental Value of Breast DCE-MRI Radiomics for Predicting Pathologic Complete Response\*\*



The repository should be executed from the repository root because the audited scripts use relative file paths.



Example:

`python code/modeling/run_M2_lockedCV.py`

`python code/modeling/run\_M2\_lockedCV.py`



The workflow is organized into preprocessing, radiomics, modeling, external validation, statistical comparison, transportability analysis, and figure generation.



## Repository workflow



The audited Python scripts are organized into:



\- `code/preprocessing/` — phase mapping, manifest construction, and locked cross-validation folds.

\- `code/radiomics/` — radiomic feature extraction for the development and I-SPY1 external cohorts.

\- `code/modeling/` — M2, M3, M4, and complete-case sensitivity analyses.

\- `code/validation/` — external validation, bootstrap comparisons, and transportability analyses.

\- `code/figures/` — manuscript tables and figures.



## Development workflow



The primary development cohort is I-SPY2 (N = 982).



The principal sequence is:



1\. establish the imaging phase map and locked manifest;

2\. construct the locked cross-validation folds;

3\. extract radiomic features;

4\. fit and evaluate the M2 clinical model;

5\. fit and evaluate the M3 radiomics model;

6\. fit and evaluate the M4 combined clinical-radiomics model;

7\. perform paired bootstrap model comparisons;

8\. perform complete-case sensitivity analyses.


## External validation workflow

### I-SPY1

I-SPY1 was used for external evaluation of both the clinical M2 model and the frozen combined M4 model.

For M4, the external-validation workflow used the locked development model without external refitting, feature reselection, preprocessing estimation, hyperparameter tuning, or recalibration.

The corrected M4 external-validation and transportability scripts are included under `code/validation/`.

### Duke

Duke was used for external evaluation of the M2 clinical model.

M4 was intentionally not applied to Duke because the available segmentation semantics were not considered compatible with the tumor masks used in the locked radiomics workflow.

## Statistical comparison and transportability

The validation scripts include bootstrap-based model comparisons and cross-cohort transportability analyses.

Paired bootstrap resampling is used when models are evaluated on the same participants.

External-minus-internal transportability comparisons use independently resampled cohorts.

The bootstrap analyses use 2,000 replicates in the locked study workflow.

## Provenance limitations

Three historical generation steps are not represented by their original scripts:

1. construction of `ISPY1_EXTERNAL_READY.csv`;
2. generation of the I-SPY1 M2 external prediction file;
3. generation of the Duke M2 external prediction file.

These gaps do not change the audited downstream analysis code, but they prevent the current repository from being described as a complete raw-data-to-final-results reproduction package.

Any future reconstructed implementation of these steps should be explicitly identified as reconstructed rather than represented as the original historical code.

## Figures and tables

The figure-generation scripts use the locked internal and external analysis outputs to reproduce the manuscript visualizations and cohort summary table.

Figure and table scripts are located under `code/figures/`.

Generated figures and patient-level intermediate files are not intended to be committed automatically to the public repository.

## Execution notes

Run scripts from the repository root so that the relative input and output paths used by the audited code resolve as intended.

The repository does not automatically download the source datasets.

Required source data should be obtained from the official repositories described in `docs/DATA_AVAILABILITY.md`.

The pinned package requirements are provided in `requirements.txt`.

Additional reproducibility and provenance details are provided in `docs/REPRODUCIBILITY.md`.

This workflow describes the audited analysis-code release and should not be interpreted as a claim of complete end-to-end reproducibility where historical provenance gaps remain.
