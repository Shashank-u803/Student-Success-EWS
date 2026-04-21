# Technical Report
## Early Warning System for Student Attrition
### Machine Learning AAT -- Comprehensive Submission

---

## 1. Problem Formulation

### 1.1 Objective
Design a machine learning system to identify students at high risk of academic dropout **before the end of Semester 2**, enabling institutions to provide timely, personalized academic and financial intervention.

### 1.2 Task Formulation
This is a **Supervised Binary Classification** problem.

| Element | Definition |
|---|---|
| **Input X** | Student demographic, socio-economic, and academic performance data (42 features after engineering) |
| **Output Y** | 1 = At risk of Dropout, 0 = On track to Graduate |
| **Prediction Horizon** | End of Semester 1 / Mid Semester 2 (provides a 2-3 month intervention window) |
| **Algorithm Class** | Gradient Boosted Trees (XGBoost) with Logistic Regression baseline |

### 1.3 Mathematical Objective

We formulate the optimization objective as a constrained recall maximization problem:

```
Maximize:  Recall(Dropout) = TP / (TP + FN)
Subject to:
    (1) Precision(Dropout) >= 0.80    -- avoid excessive false alarms
    (2) F1(Dropout)        >= 0.85    -- balanced harmonic performance
    (3) P(data leakage)    =  0       -- scaler fit only on training set

Where:
    f: R^42 -> {0, 1}
    f(x_i) = 1  if  P(Dropout | x_i) >= tau,  tau = 0.60 (alert threshold)
```

This formulation reflects the asymmetric error costs: the cost of a False Negative (missing an at-risk student, leading to irreversible dropout) vastly exceeds the cost of a False Positive (sending an unnecessary wellness check email to an on-track student).

### 1.4 Error Cost Analysis
- **False Negative (missed dropout):** Catastrophic -- a student drops out silently without receiving available support. Estimated institutional cost: lost tuition fees + reputational damage.
- **False Positive (unnecessary alert):** Low cost -- a counselor sends an unnecessary wellness email or schedules an optional check-in (< 30 minutes of counselor time).

### 1.5 Assumptions
- Data is available after Semester 2 Week 4 (sufficient history for temporal features).
- Labels are provided by the institution (Dropout is confirmed upon official withdrawal from the program).
- Privacy compliance: all student IDs are anonymized to GUIDs before model access.
- The feature distribution in future cohorts remains approximately stationary (monitored via drift detection in Section 7).

---

## 2. Dataset Strategy

### 2.1 Data Source
**Official: [UCI Machine Learning Repository (ID: 697)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)**  
**Working Mirror: [Direct CSV Link](https://raw.githubusercontent.com/shivamsingh96/Predict-students-dropout-and-academic-success/main/dataset.csv)**  
- **4,424 real student profiles** from a Portuguese higher education institution (Polytechnic Institute of Portalegre)
- **37 columns**: 36 features + 1 target label
- **3 classes**: Dropout (32.1%, n=1421), Graduate (49.9%, n=2209), Enrolled (17.9%, n=794)
- **Citation**: Martins, M.V., Tolledo, D., Machado, J., Baptista, L., Realinho, V. (2021)

### 2.2 Dataset Features Overview

| Category | Example Features |
|---|---|
| **Demographics** | Age at enrollment, Gender, Nationality, International status, Marital status |
| **Socio-Economic** | Mother's/Father's qualification, Mother's/Father's occupation |
| **Macroeconomic** | GDP, Unemployment rate, Inflation rate |
| **Financial** | Debtor status, Tuition fees up to date, Scholarship holder |
| **Academic (Sem 1)** | Units credited, enrolled, evaluations, approved, grade (0-20), without evaluations |
| **Academic (Sem 2)** | Units credited, enrolled, evaluations, approved, grade (0-20), without evaluations |
| **Admission** | Admission grade, Previous qualification grade, Application mode, Application order, Course |

### 2.3 Consideration of Multiple Data Sources
In a production deployment, additional data sources would strengthen the model:
- **University LMS Logs**: Login frequency, assignment submission latency, forum participation (engagement proxies).
- **Financial Aid Office**: Detailed payment history, loan status, aid application timeline.
- **Student Services**: Counseling visit records, accommodation status.

For this study, the UCI dataset was selected as the sole source because it already integrates academic, demographic, financial, and macroeconomic dimensions into a single validated research dataset. The 37-feature breadth provides sufficient signal diversity without requiring cross-system data integration, which introduces privacy and consistency risks in academic settings.

### 2.4 Preprocessing Pipeline

The preprocessing pipeline is illustrated in the Data Pipeline Diagram (see Section 6 diagrams).

```
Raw CSV (4,424 rows x 37 columns)
    | Binary Filter: Remove "Enrolled" class (794 records removed)
    | Target Encoding: Dropout -> 1, Graduate -> 0
    | Feature Engineering (6 new derived features, see Section 3)
    | Stratified 80/20 Train-Test Split (preserves class ratios, prevents leakage)
    | StandardScaler (fit ONLY on training set; transform applied to both)
    v
    Ready: Train (2,904 rows) + Test (726 rows) x 42 features
```

### 2.5 Data Quality Assessment
- **Missing Values**: Zero -- the UCI dataset is complete by design. Validated programmatically with `df.isnull().sum()` confirming 0 across all columns.
- **Noise/Inconsistencies**: Grade values were validated to fall within the expected 0-20 range. Enrollment counts were checked for non-negative integers. No anomalous records were detected.
- **Class Imbalance**: The Dropout class (39.1% post-filtering) is the minority class. Addressed via `scale_pos_weight` in XGBoost and `class_weight='balanced'` in Logistic Regression.

---

## 3. Feature Engineering

Raw semester grades are poor standalone predictors. A student with a 14/20 grade may be performing excellently (improved from 10/20) or collapsing (dropped from 18/20). The **direction of change** is what matters for early warning.

### 3.1 Engineered Features

| Feature | Formula | Purpose |
|---|---|---|
| `Grade_Trajectory` | Sem2 Grade - Sem1 Grade | Detects performance collapse even in high-achieving students |
| `Approval_Trajectory` | Sem2 Approved - Sem1 Approved | Measures disengagement from coursework over time |
| `Struggle_Index` | Sem2 Approved / (Sem2 Enrolled + 1) | Immediate-semester success ratio under current load |
| `Completion_Rate_S1` | Sem1 Approved / (Sem1 Enrolled + 1) | Historical baseline for comparison |
| `Financial_Risk` | (Debtor=1) OR (Tuition Not Paid) | Compound binary financial distress indicator |
| `Support_Score` | Scholarship + (1 - Debtor) | Institutional safety net strength |

### 3.2 Feature Impact Validation (SHAP Analysis)
The SHAP summary plot (generated in Script 03) confirmed the following feature importance ranking:

- **Curricular units 2nd sem (approved)** and **grade** dominate predictions -- students failing units in the most recent semester are the strongest dropout signal.
- **Tuition fees up to date** is a top-5 predictor -- financial distress correlates strongly with attrition.
- **Grade_Trajectory** (engineered) appears in the top-10, validating that temporal features add predictive power beyond raw static grades.
- **Nationality** and **Marital status** have near-zero SHAP impact -- confirming the model does not over-rely on immutable demographic attributes (important for fairness, see Section 8).

### 3.3 Justification
These features capture **temporal momentum** -- the direction and rate of a student's academic trajectory -- which early-warning literature has shown to outperform static snapshot features in predicting student attrition.

---

## 4. Model Design Strategy

### 4.1 Dual Model Architecture

We trained two models to enable rigorous comparative analysis:

**Model A -- Baseline (Logistic Regression)**
- Linear probabilistic model with `class_weight='balanced'`
- Purpose: Establishes a minimum recall threshold the advanced model must exceed
- Justification: Simple, interpretable, and a valid production fallback in low-resource settings
- Training: `max_iter=1000` to ensure convergence on 42 features

**Model B -- Primary (XGBoost Gradient Boosted Trees)**
- Non-linear ensemble method with the following hyperparameters:
  - `max_depth=4` -- controls tree complexity to prevent overfitting
  - `learning_rate=0.05` -- slow learning rate for better generalization
  - `n_estimators=200` -- 200 sequential boosting rounds
  - `scale_pos_weight=neg/pos` -- dynamically balances class importance
  - `subsample=0.8, colsample_bytree=0.8` -- row and column subsampling for regularization
- Purpose: Captures non-linear feature interactions (e.g., financial risk AND declining grades compound dropout probability more than either factor in isolation)

### 4.2 Hyperparameter Selection Justification
Parameters were selected based on established best practices for tabular classification with moderate-sized datasets (n < 5000):
- `max_depth=4` prevents overfitting on a dataset of 2,904 training samples while allowing sufficient interaction depth.
- `learning_rate=0.05` with `n_estimators=200` follows the "many weak learners" principle -- slower learning with more trees generally outperforms aggressive learning with fewer trees.
- `subsample=0.8` introduces stochastic gradient boosting, reducing variance without significantly increasing bias.

### 4.3 Robustness Considerations
- **Outliers**: XGBoost uses tree-based splits (inequalities), making it inherently robust to outlier values in continuous features like Age and Grade.
- **Feature noise**: The `colsample_bytree=0.8` parameter ensures that no single noisy feature can dominate every tree in the ensemble.
- **Missing values at inference**: XGBoost natively handles missing values at inference time by learning default split directions during training -- a critical advantage for production deployment where LMS data may arrive incomplete.

### 4.4 Trade-offs

| Dimension | Logistic Regression | XGBoost |
|---|---|---|
| Interpretability | High (coefficients) | Medium (requires SHAP) |
| Non-linear patterns | Cannot capture | Native support |
| Robustness to outliers | Sensitive | Robust |
| Deployability | Lightweight (~KB) | Heavier (~MB) |
| Class Imbalance | class_weight | scale_pos_weight |
| Missing values | Cannot handle | Native support |

### 4.5 Scalability
The trained XGBoost model file (`xgb_model.pkl`) is approximately 1 MB. Inference for a single student takes < 1 ms. The system can score an entire university cohort of 10,000 students in under 10 seconds on standard hardware, making it suitable for both small colleges and large universities.

---

## 5. Evaluation Strategy

### 5.1 Why Not Just Accuracy?
A naive classifier that always predicts "Graduate" would achieve ~60% accuracy while identifying **zero actual dropouts**. In an early warning system, this represents a complete system failure. Accuracy alone is an insufficient and misleading metric.

### 5.2 Primary Metric: Recall

```
Recall = TP / (TP + FN)
```

Maximized to ensure we identify as many at-risk students as possible before they leave.

### 5.3 Secondary Metrics

| Metric | Purpose |
|---|---|
| **F1-Score** | Harmonic mean of precision and recall -- ensures balanced performance |
| **PR-AUC** | Area under Precision-Recall curve -- preferred over ROC-AUC for imbalanced datasets |
| **ROC-AUC** | Overall discrimination ability across all thresholds |
| **Confusion Matrix** | Visual breakdown of TP, TN, FP, FN |

### 5.4 Validation Strategy
**Stratified 80/20 Split** -- the `stratify=y` parameter ensures both training (2,904) and test (726) sets preserve the original Dropout:Graduate ratio. The StandardScaler is fit exclusively on the training set and only transformed on the test set to prevent any form of data leakage.

### 5.5 Acceptable Performance Thresholds

For the system to be considered **safe for deployment**, we define the following minimum thresholds:

| Metric | Minimum Acceptable | Achieved | Status |
|---|---|---|---|
| Dropout Recall | >= 0.85 | 0.93 | PASS |
| Dropout Precision | >= 0.80 | 0.89 | PASS |
| Dropout F1 | >= 0.85 | 0.91 | PASS |
| ROC-AUC | >= 0.90 | 0.975 | PASS |
| Max Fairness Gap (Recall) | <= 0.15 | 0.108 | PASS |

If any metric falls below its threshold during monitoring, the model must be retrained before continued use (see Section 7).

### 5.6 Results Summary (Real Dataset -- UCI ID 697, 3,630 students)

| Model | Dropout Recall | Dropout Precision | Dropout F1 | Overall Accuracy | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.94 | 0.88 | 0.91 | 93% | 0.973 | 0.974 |
| **XGBoost (Primary -- Deployed)** | **0.93** | **0.89** | **0.91** | **93%** | **0.973** | **0.975** |

**Key Findings:**
- Both models achieve 93% recall on Dropouts -- 93 out of every 100 at-risk students are correctly flagged for intervention.
- XGBoost achieves a ROC-AUC of 0.975, confirming excellent discrimination ability across all threshold settings.
- The PR-AUC of 0.973 is particularly strong given the class imbalance -- the model maintains high precision even at high recall operating points.
- XGBoost is selected as the deployment model due to its superior feature interaction modeling, native missing-value handling, and SHAP compatibility.

---

## 6. Deployment Design

### 6.1 System Architecture

*(See architecture diagram: `docs/images/architecture_diagram.png`)*

The system operates as an **asynchronous batch-inference pipeline**:

1. At the end of Week 4 of Semester 2, the university LMS and Financial databases trigger a weekly ETL (Extract, Transform, Load) dump.
2. `01_data_acquisition.py` ingests and validates the raw student data.
3. `02_preprocessing.py` executes the feature engineering pipeline, producing 42 standardized features per student.
4. The serialized XGBoost model (`xgb_model.pkl`) scores each student, producing P(Dropout) in < 1 ms per student.
5. Students with P(Dropout) >= 0.60 are flagged as HIGH RISK and surfaced in the Counselor Dashboard.
6. Students with 0.40 <= P(Dropout) < 0.60 are placed on MONITOR status.
7. All remaining students are logged for audit purposes only.

### 6.2 Data Pipeline

*(See data pipeline diagram: `docs/images/data_pipeline_diagram.png`)*

### 6.3 System Components

| Component | Technology | Role |
|---|---|---|
| **Batch ETL** | Python script (cron/scheduled task, weekly) | Extracts student data from LMS |
| **Feature Pipeline** | `02_preprocessing.py` + `scaler.pkl` | Cleans data and engineers 42 features |
| **Inference Engine** | XGBoost + Pickle | Scores each student in < 1 ms |
| **API Layer** | FastAPI (production) | Exposes `/predict` and `/batch-predict` endpoints |
| **Dashboard** | Streamlit (`05_dashboard.py`) | Counselor-facing web interface |
| **Alerting** | In-dashboard status cards + email integration | Notifies counselors of high-risk students |

### 6.4 Counselor Dashboard Features (Implemented)
The Streamlit dashboard (`05_dashboard.py`) is a fully functional, deployable web application with five views:

- **Overview**: KPI cards showing High Risk / Monitor / On Track counts, risk score distribution histogram, and status breakdown donut chart.
- **Student Profiles**: Filterable by risk status. Per-student academic record, financial status, and automated intervention recommendations.
- **What-If Analysis**: Counselors can adjust a student's grades, tuition status, or scholarship and immediately see how the predicted risk score changes. This enables data-driven intervention planning.
- **Model Insights**: Performance comparison table, confusion matrices, Precision-Recall curve, feature importance chart, and SHAP summary.
- **EDA Reports**: All exploratory data analysis charts from the dataset study.

### 6.5 Deployment Constraints and Mitigations

| Constraint | Mitigation |
|---|---|
| **Latency** | Batch inference (weekly), not real-time -- eliminates cold-start and latency issues |
| **Scalability** | Stateless model; horizontally scalable on any cloud VM or on-premise server |
| **Reliability** | Model is serialized and version-controlled; automatic fallback to previous model version if validation checks fail |
| **Offline Access** | Dashboard runs entirely locally; no external API dependencies at inference time |

---

## 7. Monitoring and Maintenance Strategy

### 7.1 Concept Drift Detection
Student behavior patterns shift over time (e.g., post-COVID shift to hybrid learning, new grading policies). We monitor:
- **Input Distribution Drift**: KL-Divergence of each feature's distribution vs. training-time baseline. Alerts trigger if KL-divergence exceeds 0.1 for any top-10 feature.
- **Performance Drift**: If quarterly Recall on the most recent semester's confirmed outcomes drops below 0.80, a mandatory retraining alert is triggered.

### 7.2 Retraining Schedule

| Event | Action |
|---|---|
| **Annual (every August)** | Full model retrain on the previous year's confirmed academic outcomes |
| **Emergency Trigger** | If Recall drops > 10% month-over-month on validated labels |
| **Policy Change** | Any change in grading scheme, semester structure, tuition policy, or new course addition |

### 7.3 Logging and Alerting
- All predictions are logged to an append-only audit table with timestamp, student GUID, predicted probability, and assigned risk tier.
- Counselor actions (student reviewed, intervention delivered, outcome recorded) are logged for future training signal and accountability.
- A monitoring dashboard (Prometheus / Grafana in production) tracks daily inference counts, average risk score distribution, and model latency.

### 7.4 Fallback Mechanism
If the ML model is unavailable or fails validation checks, the system defaults to a **rule-based fallback**: students with Semester 2 approval rate < 50% AND tuition overdue are auto-flagged. This ensures continuous operation even during model retraining windows.

---

## 8. Ethical, Social, and Risk Considerations

### 8.1 Bias Analysis
**Risk Identified:** If the historical dataset encodes systemic bias (e.g., students from lower socio-economic groups historically dropped out more frequently due to financial constraints rather than academic inability), the model may perpetuate discriminatory patterns.

### 8.2 Fairness Audit (Conducted)
A formal sub-group fairness audit was conducted using `06_fairness_audit.py`. The model's Dropout Recall was evaluated across demographic and socio-economic sub-groups:

| Sub-group | n | Recall | Precision | F1 |
|---|---|---|---|---|
| ALL STUDENTS (Overall) | 726 | 0.930 | 0.895 | 0.912 |
| Gender: Female | 438 | 0.923 | 0.902 | 0.913 |
| Gender: Male | 288 | 0.935 | 0.889 | 0.911 |
| Scholarship: Yes | 174 | 0.923 | 0.857 | 0.889 |
| Scholarship: No | 552 | 0.930 | 0.899 | 0.914 |
| Financial Risk: Yes | 147 | 0.992 | 0.952 | 0.971 |
| Financial Risk: No | 579 | 0.884 | 0.853 | 0.868 |
| Age: 17-20 (Young) | 422 | 0.903 | 0.816 | 0.857 |
| Age: 21-25 (Mid) | 130 | 0.927 | 0.895 | 0.911 |
| Age: 26+ (Mature) | 174 | 0.957 | 0.982 | 0.969 |

**Findings:**
- Gender gap in Recall: 0.012 (Female 0.923 vs Male 0.935) -- negligible, within acceptable bounds.
- Scholarship gap: 0.007 -- negligible.
- Largest gap: Financial Risk (0.108). Students with financial risk are detected at 0.992 recall vs 0.884 for non-financial-risk students. This is acceptable because financially distressed students exhibit stronger dropout signals, not because the model discriminates unfairly.
- Maximum overall Recall gap: 0.108 -- within the < 0.15 deployment threshold.

### 8.3 Model Transparency (SHAP)
- SHAP analysis confirmed that **academic performance features** (semester grades, unit approvals) dominate predictions -- not immutable attributes like nationality, gender, or parental occupation.
- The SHAP summary and feature importance charts are available in the dashboard's "Model Insights" tab for counselor review.

### 8.4 Privacy Architecture
- Student records are accessed via anonymized GUIDs; names and contact information are never fed to the model.
- The counselor dashboard displays Student ID only; name lookup requires a separate HR system authorization with its own access controls.
- Model training data is stored on-premise; no student data leaves the institutional network.

### 8.5 Risk of Over-Reliance
Counselors must be trained to treat model alerts as **decision support**, not deterministic verdicts. All flagged students require human review before any formal intervention. The system provides recommendations, not mandates.

### 8.6 Accountability
The system maintains a complete audit trail: which students were flagged, at what probability, which counselor reviewed the alert, what action was taken, and the eventual outcome. This enables accountability reviews, appeals processes, and continuous model improvement.

---

## 9. SDG Mapping

### Primary Alignment: SDG 4 -- Quality Education
> "Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all."

| SDG Target | How Our System Contributes |
|---|---|
| **4.1** -- Free, equitable education for all | By reducing dropout rates, more students complete their education regardless of socio-economic background |
| **4.3** -- Equal access to tertiary education | The What-If Simulator and financial risk flag ensure counselors prioritize economically vulnerable students for intervention |
| **4.5** -- Eliminate gender/socio-economic disparities | Fairness audits (Section 8.2) confirm the model does not discriminate against any demographic group |
| **4.b** -- Scholarships and financial aid | The system specifically surfaces students lacking tuition support, enabling proactive financial aid referral |

### Secondary SDG Alignment

| SDG | Relevance |
|---|---|
| **SDG 1 (No Poverty)** | Financial risk flags help divert students to emergency aid before dropout occurs, preventing the economic consequences of incomplete education |
| **SDG 10 (Reduced Inequalities)** | Explicit bias mitigation via fairness audits ensures marginalized groups are protected, not algorithmically punished |

### Real-World Impact Statement
The OECD estimates that each additional year of education increases individual lifetime earnings by approximately 8%. By preventing even 100 dropouts per year at a single institution, this system can contribute to measurable economic uplift and social mobility -- directly embodying the spirit of SDG 4 and the 2030 Agenda for Sustainable Development.

---

## Appendix A: Running the System

```bash
# From the project root: c:\Users\...\ML AAT\code

# Step 1 -- Download the real UCI dataset
python 01_data_acquisition.py

# Step 2 -- Preprocess and engineer features
python 02_preprocessing.py

# Step 3 -- Generate EDA charts
python 04_eda.py

# Step 4 -- Train models and generate evaluation plots + SHAP
python 03_model_training.py

# Step 5 -- Run fairness audit
python 06_fairness_audit.py

# Step 6 -- Launch the Counselor Dashboard
streamlit run 05_dashboard.py
```

## Appendix B: Generated Artifacts

| File | Description |
|---|---|
| `data/student_attrition_raw.csv` | Real UCI dataset (4,424 rows x 37 columns) |
| `data/X_train.csv`, `X_test.csv` | Preprocessed feature matrices |
| `data/y_train.csv`, `y_test.csv` | Binary target labels |
| `data/xgb_model.pkl` | Trained XGBoost model (serialized) |
| `data/scaler.pkl` | StandardScaler (for dashboard inference) |
| `data/fairness_audit.csv` | Sub-group fairness results |
| `data/eda_plots/01-07_*.png` | EDA analysis charts |
| `data/eda_plots/08-11_*.png` | Model evaluation charts |
| `data/eda_plots/12_fairness_audit.png` | Fairness audit chart |
| `data/eda_plots/13_architecture_diagram.png` | System architecture diagram |
| `data/eda_plots/14_data_pipeline_diagram.png` | Data pipeline diagram |
