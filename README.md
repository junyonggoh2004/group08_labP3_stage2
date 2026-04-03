# INF2008 Group Project - Hotel Booking Cancellation Risk Prediction

## Overview
This repository contains Group 08's INF2008 Lab P3 project on hotel booking cancellation risk prediction. The project follows a CRISP-DM-style workflow and focuses on building a leak-free machine learning pipeline that predicts whether a booking is likely to be cancelled at the point of reservation.

The final notebook covers:
- business understanding and success targets
- exploratory data analysis with formal statistical testing
- deterministic cleaning before split
- pipeline-based preprocessing and imbalance handling
- comparison of three model families
- hyperparameter tuning and champion selection
- controlled ablation experiments
- row-level failure analysis
- business decision translation through risk bands

## Problem Statement
Hotels face two competing risks:
- underestimating cancellations, which leads to unsold room-nights and revenue loss
- overestimating cancellations, which can trigger unnecessary intervention and operational cost

This project produces a booking-level cancellation risk score to support:
- pre-arrival outreach for riskier bookings
- safer inventory protection and overbooking decisions
- better operational planning for staffing and room readiness

## Dataset
- **Source:** [Kaggle - Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Records:** 119,390 hotel bookings
- **Period:** 1 July 2015 to 31 August 2017
- **Hotels:** Resort Hotel and City Hotel in Portugal
- **Target:** `is_cancelled` where `1 = cancelled`, `0 = not cancelled`

The raw dataset is included as:
- [`hotel_bookings.csv`](./hotel_bookings.csv)

## Repository Structure
```text
group08_labP3_stage2/
├── CRISPDM_INF2008_assignment02.ipynb   # Main project notebook
├── hotel_bookings.csv                   # Dataset used in the notebook
└── README.md                            # Project documentation
```

## Main Deliverable
The main deliverable is the notebook:
- [`CRISPDM_INF2008_assignment02.ipynb`](./CRISPDM_INF2008_assignment02.ipynb)

Its top-level flow is:
1. Business Understanding
2. About the Dataset
3. Importing Relevant Libraries
4. Loading the Dataset
5. Exploratory Data Analysis
6. Data Preprocessing
7. Post-Data Preprocessing EDA
8. Model Justification via Correlation and PCA
9. Model Evaluation Strategy
10. Train / Validation / Test Split
11. Model Development
12. Baseline Validation Snapshot Metrics
13. Baseline Ordinary-Split Snapshot
14. Leak-Free Imbalance-Handling Workflow
15. Transition to Advanced Modelling
16. Transition Summary
17. Advanced Modelling, Evaluation, and Business Translation
18. Final Candidate Lock and Refit on Train + Validation
19. Final Test Evaluation
20. High-Confidence Error Analysis
21. Business Decision Translation
22. Final Integrated Summary

## Modelling Objective and Evaluation Hierarchy
The project uses a Recall-first evaluation hierarchy because missing true cancellations is more damaging to hotel revenue planning than raising some extra alerts.

Model comparison follows this order of importance:
1. **Recall** - catch enough real cancellations
2. **F2-score** - optimise a Recall-heavy trade-off
3. **Precision** - keep false alarms manageable
4. **ROC-AUC / PR diagnostics** - provide threshold-independent context

### Deployment Gates
The held-out test set is evaluated against these explicit gates:
- **Recall >= 0.75**
- **F2-score >= 0.64**
- **Precision >= 0.50**

The model is considered deployment-ready only if all three conditions are met on the untouched test split.

## Pipeline Design
The notebook was deliberately structured to prevent leakage.

### Deterministic Cleaning Before Split
The following steps are applied before any fitted transformation:
- remove duplicates and invalid rows
- standardise month and meal labels
- correct negative ADR values
- fill simple missing values that do not require learning from the data
- drop leakage-prone or unnecessary fields

### Train-Only Feature Processing
After the raw `70 / 15 / 15` split, modelling is done through formal pipelines:
- feature engineering inside the workflow:
  - `parking_required`
  - `total_stay_nights`
  - `is_family`
  - `arrival_month_num`
- country grouping learned from training data only:
  - `Top 10 countries + Other`
- `ColumnTransformer` for:
  - `log1p + StandardScaler` on `lead_time` and `adr`
  - one-hot encoding for categorical variables
  - passthrough for binary variables
- `SMOTENC` for class imbalance handling inside training folds only

This ensures the validation and test splits are never used to fit encoders, scalers, or resampling logic.

## Model Families Compared
Three distinct families were compared under the same pipeline structure:
- **Logistic Regression** - linear baseline
- **Random Forest** - bagging ensemble
- **HistGradientBoostingClassifier** - boosting ensemble

`DummyClassifier` was also used as a naive majority-class reference in the earlier baseline snapshot.

## Key Analytical Additions
Compared with a simpler baseline workflow, this project added:
- explicit business success thresholds
- formal statistical testing in EDA:
  - chi-square tests for categorical variables
  - Welch's t-tests for numeric variables
  - Benjamini-Hochberg correction for multiple testing
- leak-free preprocessing inside formal pipelines
- cross-validation-based champion selection
- tuning of the best candidates
- controlled ablation on the selected champion
- validation-based failure analysis
- business risk-band translation

## Hyperparameter Tuning Results
The tuned models and their best cross-validated F2-scores were:

- **Logistic Regression**
  - Best params: `{'C': 10.0, 'solver': 'liblinear'}`
  - Best CV F2: `0.6775`

- **Random Forest**
  - Best params: `{'n_estimators': 350, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 20}`
  - Best CV F2: `0.7082`

- **HistGradientBoostingClassifier**
  - Best params: `{'min_samples_leaf': 40, 'max_iter': 300, 'max_depth': 5, 'learning_rate': 0.1}`
  - Best CV F2: `0.7099`

## Champion Selection
Post-tuning 5-fold CV selected:
- **Champion:** `HistGradientBoosting (Tuned)`

Post-tuning CV summary:
- **CV F2 Mean:** `0.7099`
- **CV F2 Std:** `0.0061`
- **CV Recall Mean:** `0.7355`
- **CV Precision Mean:** `0.6232`
- **CV ROC-AUC Mean:** `0.8774`

This champion was carried forward into:
- validation-stage ensemble comparison
- controlled ablation
- final refit on `train + validation`
- final test evaluation

## Ensemble Comparison
The tuned champion was compared against:
- `VotingClassifier (soft)`
- `StackingClassifier`

After enforcing threshold-consistent validation scoring, the tuned `HistGradientBoostingClassifier` remained the preferred candidate. The ensemble variants did not deliver a strong enough improvement to replace it.

## Ablation Findings
Controlled experiments were run against the selected champion workflow.

### Reference
- **E1: Full selected candidate**
  - CV F2 Mean: `0.7099`

### Controlled Changes
- **E2: No SMOTENC**
  - CV F2 Mean: `0.6229`
  - Strong drop in Recall and overall F2
  - Conclusion: class imbalance handling is critical

- **E3: Drop engineered features**
  - CV F2 Mean: `0.6853`
  - Clear performance drop
  - Conclusion: engineered booking-context features add meaningful value

- **E4: Drop prior-history features**
  - CV F2 Mean: `0.7085`
  - Slight drop
  - Conclusion: prior-history features help, but only modestly

- **E5: Drop deposit type**
  - CV F2 Mean: `0.7078`
  - Slight drop
  - Conclusion: `deposit_type` contributes, but is not the dominant feature on its own

## Final Test Results
The final locked candidate was refit on `train + validation` and evaluated once on the untouched test set.

### Final Metrics
- **Accuracy:** `0.8032`
- **Precision:** `0.6188`
- **Recall:** `0.7409`
- **F2-score:** `0.7128`
- **ROC-AUC:** `0.8798`

### Confusion Matrix
- **True Negatives:** `7824`
- **False Positives:** `1638`
- **False Negatives:** `930`
- **True Positives:** `2659`

### Threshold Used
- **Decision threshold:** `0.50`
- Rule: highest validation threshold that still satisfied the Recall and Precision gates

### Final Verdict
- **Deployment verdict:** `Not ready`

Reason:
- Precision passed the required floor
- F2 passed the required floor
- Recall (`0.7409`) fell slightly below the deployment gate of `0.75`

## Error Analysis
High-confidence validation errors were reviewed at row level.

Observed patterns:
- **False Negatives**
  - often had `No Deposit`
  - often came from `Groups`, `Offline TA/TO`, `Direct`, or `Corporate`
  - tended to have shorter or moderate lead times and weaker prior-risk signals

- **False Positives**
  - often had `No Deposit`
  - frequently came from `Online TA`
  - tended to have long lead times, higher ADR, and transient booking patterns

This analysis was used to explain where the model remains overconfident and where future feature improvements or threshold refinements may help.

## Business Decision Translation
The notebook separates:
- a **formal yes/no decision threshold** for classification
- **risk bands** for operational prioritisation

### Locked Decision Threshold
- `0.50` was selected from the validation set
- it was kept fixed for final test evaluation
- it was not changed after seeing the test results

### Risk Bands
- **Low risk:** `< 0.30`
- **Medium risk:** `0.30 to < 0.65`
- **High risk:** `>= 0.65`

These bands are used to sort bookings by risk so hotel teams can decide which reservations need attention first.

## How to Run
### 1. Create and activate a virtual environment
Example on Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Example on bash:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
This project uses:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `imbalanced-learn`
- `jupyter`
- `ipykernel`

Example:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn jupyter ipykernel
```

### 3. Launch Jupyter
```bash
jupyter notebook
```

Then open:
- `CRISPDM_INF2008_assignment02.ipynb`

### 4. Run the notebook
Run the notebook from top to bottom to ensure:
- variables are defined in the right order
- thresholds are declared before business translation
- outputs and markdown findings stay aligned

## Reproducibility Notes
- The project assumes the dataset file is available as `hotel_bookings.csv` in the same directory as the notebook.
- The notebook should be run sequentially from the first cell.
- Later sections depend on earlier derived objects such as:
  - train / validation / test splits
  - tuned estimators
  - selected thresholds
  - final candidate pipeline

## Team
Group 08:
- Thaw Zin Lin (`2500651`)
- Chun Jin Xiang (`2500652`)
- Lim Xin Chian (`2500671`)
- Justin Tan Jun An (`2500966`)
- Goh Jun Yong (`2504002`)

## Summary
This project improved a simple hotel-cancellation baseline into a full leak-free machine learning workflow with:
- formal preprocessing pipelines
- Recall-first model selection
- tuned champion selection
- controlled ablation
- row-level failure analysis
- business-facing risk translation

The final tuned `HistGradientBoostingClassifier` performed strongly, but the current version remains slightly below the required Recall deployment gate. The project therefore concludes with a technically solid model and a clear explanation of what still prevents immediate operational deployment.
