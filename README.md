# Cross-Cohort Transportability, Calibration, and Incremental Value of Breast DCE-MRI Radiomics for Predicting Pathologic Complete Response



## Overview



This repository contains the analysis code supporting the study:



**Cross-Cohort Transportability, Calibration, and Incremental Value of Breast DCE-MRI Radiomics for Predicting Pathologic Complete Response**



The study evaluates whether DCE-MRI radiomic features provide incremental predictive value beyond routinely available clinical variables for predicting pathologic complete response (pCR), and examines model transportability and calibration across independent breast cancer cohorts.



The analysis uses publicly available, deidentified data derived from the BreastDCEDL resource and its source cohorts, including I-SPY2, I-SPY1, and the Duke Breast Cancer MRI cohort.



## Study cohorts



The locked analysis includes:



| Cohort | Role | N | pCR |

|---|---|---:|---:|

| I-SPY2 | Model development and internal validation | 982 | 316 (32.2%) |

| I-SPY1 | External validation | 167 | 47 (28.1%) |

| Duke | External clinical-model validation | 297 | 62 (20.9%) |



M4 was intentionally not evaluated in the Duke cohort because the available Duke segmentation masks were not considered semantically compatible with the tumor masks used to derive the locked M4 radiomic features.



## Models



Three prespecified model configurations were evaluated:



- **M2 — Clinical model:** age, hormone receptor (HR) status, HER2 status, and log-transformed tumor volume.

- **M3 — Radiomics model:** radiomic features selected within nested cross-validation using SelectKBest and L2-penalized logistic regression.

- **M4 — Combined model:** all M2 clinical predictors plus selected radiomic features.



Clinical predictors were retained in M4 while radiomic feature selection was performed within the training data.



## Validation framework



Internal performance in I-SPY2 was estimated using locked five-fold stratified cross-validation.



Hyperparameter and feature-selection decisions for M3 and M4 were performed within the training portion of each outer fold.



The final M4 model used:



- 4 clinical predictors

- 20 selected radiomic predictors

- L2-penalized logistic regression

- final `C = 0.1`



The final value of `C` followed the deterministic tie-breaking behavior implemented in the locked analysis pipeline.



External validation in I-SPY1 used the frozen model without external refitting, feature reselection, preprocessing estimation, hyperparameter tuning, or recalibration.



## Primary results



### I-SPY2 internal validation



| Model | AUROC | AUPRC | Brier score |

|---|---:|---:|---:|

| M2 | 0.698 | 0.524 | 0.194 |

| M3 | 0.625 | 0.426 | 0.210 |

| M4 | 0.707 | 0.526 | 0.193 |



The incremental performance of M4 over M2 was small.



### I-SPY1 external validation



| Model | AUROC | AUPRC | Brier score |

|---|---:|---:|---:|

| M2 | 0.773 | 0.572 | 0.167 |

| M4 | 0.758 | 0.540 | 0.172 |



### Duke external validation



The clinical M2 model achieved:



- AUROC: **0.681**

- AUPRC: **0.356**

- Brier score: **0.181**



## Repository structure




```text

code/

├── preprocessing/

├── radiomics/

├── modeling/

├── validation/

└── figures/



docs/

requirements.txt

## Radiomics

Radiomic features were extracted from three DCE-MRI acquisitions and one tumor mask per I-SPY2 participant using PyRadiomics 3.0.1.

The default PyRadiomics extractor was used with original-image features only (`original_`). No explicit image resampling, intensity normalization, study-specific bin width, or additional image filters were applied.

A total of 107 radiomic features were extracted per acquisition, yielding 321 candidate radiomic features across the three acquisitions.

Radiomic feature selection was performed within the training data to reduce information leakage. No formal claim of full IBSI compliance is made.


## Complete-case sensitivity analysis

Three I-SPY2 participants had missing age in the source metadata. The primary analysis used mean imputation based on the 979 participants with observed age.

A complete-case sensitivity analysis excluded these three cases using the source-metadata missing-age criterion rather than hard-coded patient identifiers. An equivalence audit confirmed that this criterion reproduced the historical exclusion exactly.

The complete-case cohort contained N = 979 while preserving the locked fold assignments.

## Software environment

The audited repository-preparation environment used:

- Python 3.11.9
- NumPy 2.4.6
- pandas 3.0.6
- scikit-learn 1.9.1
- Matplotlib 3.11.2
- joblib 1.6.0
- PyRadiomics 3.0.1
- SimpleITK 2.5.6

Pinned package versions are provided in `requirements.txt`.

`pip check` returned `No broken requirements found.`

These versions describe the audited repository-preparation environment and should not be interpreted as proof that every historical intermediate artifact was originally generated under exactly the same software environment.

## Data availability

This repository does not redistribute patient-level source or derived data.

The study uses publicly available, deidentified data from the BreastDCEDL resource and its contributing I-SPY1, I-SPY2, and Duke breast DCE-MRI cohorts.

Users should obtain the required datasets from their official repositories and comply with the applicable licenses, access conditions, and terms of use.

Dataset identifiers and local reproduction requirements are documented in `docs/DATA_AVAILABILITY.md`.

## Reproducibility status

This repository is an audited analysis-code release with documented provenance limitations.

The released analysis code includes 26 audited Python scripts covering preprocessing, radiomics, modeling, validation, and figure/table generation.

Three historical generation steps are not currently represented by preserved scripts:

1. Construction of `ISPY1_EXTERNAL_READY.csv`.
2. Generation of the I-SPY1 M2 external prediction file.
3. Generation of the Duke M2 external prediction file.

These gaps do not alter the reported audit of the preserved analysis code, but they prevent a claim of complete raw-data-to-final-results end-to-end reproducibility.

See `docs/REPRODUCIBILITY.md` and `docs/ANALYSIS_WORKFLOW.md` for details.

## Privacy and repository policy

Patient-level data and derived patient-level artifacts are intentionally excluded from the public repository.

This includes clinical metadata, patient-level predictions, radiomics matrices, MRI data, segmentation masks, serialized model objects, and files containing patient identifiers.

The repository is intended to provide analysis code and reproducibility documentation without redistributing protected or license-restricted study data.

## Installation

Create or activate a Python environment and install the pinned dependencies from the repository root:

    pip install -r requirements.txt

The audited repository-preparation environment used Python 3.11.9.

## Reproduction notes

Run analysis scripts from the repository root so that relative file paths resolve as intended.

For example:

    python code/modeling/run_M2_lockedCV.py

The repository does not automatically download the source datasets. Obtain the required data from the official sources described in `docs/DATA_AVAILABILITY.md`.

The overall analysis sequence and known provenance limitations are documented in `docs/ANALYSIS_WORKFLOW.md` and `docs/REPRODUCIBILITY.md`.

## Citation

Citation information for this code release will be finalized when the public repository and versioned release are created.

The associated manuscript is:

Cross-Cohort Transportability, Calibration, and Incremental Value of Breast DCE-MRI Radiomics for Predicting Pathologic Complete Response.

## License

A code license has not yet been assigned.

The licenses and terms governing the source datasets are separate from the license that may later be selected for this repository.

## Disclaimer

This repository is provided for research and reproducibility purposes.

It is not intended for clinical diagnosis, treatment decisions, or direct clinical deployment.

The repository does not redistribute patient-level study data, and users are responsible for complying with the licenses and terms of the original data sources.
