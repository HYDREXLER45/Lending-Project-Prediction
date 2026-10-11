# Loan Default Prediction (LendingClub Data)

## 1. Overview

This project predicts whether a loan will be **repaid or charged off** using only information known when the loan was issued (borrower profile, loan terms, credit history). The model is an XGBoost classifier, tuned with randomized search and evaluated on a held-out test set.

- **Task:** binary classification
- **Target:** `default` (1 = loan charged off / defaulted, 0 = fully paid)
- **Model:** XGBoost inside a scikit-learn pipeline
- **Primary metric:** ROC AUC (accuracy is reported but is misleading on imbalanced data)

---

## 2. Dataset

LendingClub loan data. Columns used:

| Group | Columns |
|---|---|
| Loan terms | `loan_amnt`, `term`, `int_rate`, `installment`, `grade`, `sub_grade`, `purpose` |
| Borrower | `emp_length`, `home_ownership`, `annual_inc`, `verification_status` |
| Credit history | `dti`, `delinq_2yrs`, `fico_range_low`, `fico_range_high`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc` |
| Bookkeeping | `issue_d` (issue date), `loan_status` (outcome, used only to build the label) |

All feature columns are known at application time. No post-issuance columns (payments received, recoveries, last payment date) are used, to avoid data leakage.

---

## 3. Methodology

### 3.1 Target construction

The raw `loan_status` column contains several values. Only loans with a **final outcome** are usable, so statuses are mapped explicitly (not label-encoded, since the statuses have no meaningful order):

| loan_status | Count | Label |
|---|---|---|
| Fully Paid | 1,076,751 | 0 |
| Does not meet the credit policy. Status:Fully Paid | 1,962 | 0 |
| Charged Off | 268,559 | 1 |
| Does not meet the credit policy. Status:Charged Off | 758 | 1 |
| Default | n/a | 1 |
| Current | 878,317 | dropped |
| Late (31-120 days) | 21,467 | dropped |
| In Grace Period | 8,436 | dropped |
| Late (16-30 days) | 4,349 | dropped |

Unfinished loans (Current, Late, Grace Period) are dropped because their outcome is unknown. Labeling them as "not default" would teach the model wrong answers, since some will default later. The 33 rows with a missing `loan_status` are removed in the same step, because a missing target is never imputed.

Result: roughly 80% paid and 20% defaulted.

### 3.2 Missing values

| Column | Missing | Treatment |
|---|---|---|
| `loan_status` | 33 | Dropped (target) |
| `emp_length` | 146,907 | Column dropped (weak predictor, large share missing) |
| `annual_inc` | 4 | Rows dropped |
| `delinq_2yrs`, `open_acc`, `pub_rec`, `total_acc` | 29 each | Rows dropped (same borrowers with no credit history on file) |
| `dti` | 1,711 | Filled with the median |
| `revol_util` | 1,802 | Filled with the median |

The median is used because both `dti` and `revol_util` contain large outliers (e.g. `dti` up to 999) that would distort a mean.

### 3.3 Feature engineering and encoding

- **FICO score:** `fico_range_low` and `fico_range_high` are nearly identical, so they are averaged into one `fico` column and the originals are dropped.
- **`term`:** the number is extracted from the text (`" 36 months"` becomes 36).
- **`grade`, `sub_grade`:** ordered categories, converted to integer codes with `.astype("category").cat.codes`. Alphabetical order matches risk order (A is best, G is worst).
- **`home_ownership`, `verification_status`, `purpose`:** nominal categories, one-hot encoded with `pd.get_dummies`.

### 3.4 Leakage controls

- `loan_status` is dropped from the features after `default` is created, otherwise the model would read the answer.
- `issue_d` is dropped from the features (it is a date string) but kept in the data to allow a time-based split.
- `grade`, `sub_grade` and `int_rate` are assigned by LendingClub's own risk model. They are legitimate application-time features but overlap heavily with FICO, so they can mask what drives risk.
- No post-loan columns are used.

### 3.5 Train/test split

80/20 split, stratified on `default` so both sets keep the same class balance (`random_state=2`).

### 3.6 Model

An sklearn `Pipeline`:

1. `StandardScaler`: standardizes the numeric features. XGBoost does not need scaling, so this does not change results, but it is kept for pipeline consistency and would matter if the classifier were swapped for a linear model.
2. `XGBClassifier` with `tree_method="hist"` and `eval_metric="logloss"`.

**Class imbalance:** `scale_pos_weight = (number of paid) / (number of defaults)` makes the model weight the rarer default class more heavily. This improves recall on defaults at the cost of accuracy and calibrated probabilities.

### 3.7 Hyperparameter tuning

`RandomizedSearchCV` with 3-fold stratified cross-validation, 10 iterations, scored on ROC AUC.

| Parameter | Values searched |
|---|---|
| `max_depth` | 3, 4, 5 |
| `learning_rate` | 0.01, 0.05, 0.1 |
| `n_estimators` | 100, 200, 300 |
| `subsample` | 0.7, 0.8, 1.0 |
| `colsample_bytree` | 0.7, 0.8, 1.0 |

With about 1.3 million rows, the search can be slow. For a faster first run, tune on a 200,000-row sample, then refit the best settings on the full training set.

### 3.8 Evaluation

- **ROC AUC:** main metric, threshold-independent.
- **Classification report:** precision, recall and F1 per class.
- **Confusion matrix:** heatmap of predicted vs actual.
- **Accuracy:** reported for reference only. A model that always predicts "paid" would score about 80%.

Typical credit-risk models on application-time data reach an AUC of about 0.70 to 0.72. Defaults are only partly predictable from this information, so a much higher score would suggest data leakage.

---

## 4. How to run

Run the steps in this order (the pipeline fails on text or NaN values otherwise):

```python
data = pd.read_csv("loan.csv")

# 1. build the target and keep only finished loans
data["default"] = data["loan_status"].map(status_map)
data = data.dropna(subset=["default"]).copy()
data["default"] = data["default"].astype(int)

# 2. clean
data = data.dropna(subset=["annual_inc", "delinq_2yrs", "open_acc", "pub_rec", "total_acc"])
data = data.drop(columns="emp_length")
data = fill_column(data)       # median-fill dti, revol_util
data = encode_columns(data)    # term, grade, sub_grade, one-hot columns

# 3. model
best_model = Machine_Model(data)
```

Requirements: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`.

---

## 5. Limitations

- **Random split vs time:** the current split is random. Loan performance depends on the issue year, so a time-based split (train on older loans, test on newer) better reflects real use.
- **Survivorship skew:** dropping the 878k "Current" loans biases the data toward older loans, since recent loans have not finished.
- **Uncalibrated probabilities:** `scale_pos_weight` pushes predicted probabilities upward. Use them to rank loans by risk. Remove the weight if true probabilities are needed.
- **Median imputation before the split:** `dti` and `revol_util` are filled using the full dataset's median. The effect is tiny at this scale, but the stricter approach is to fit the median on the training set only.
- **Redundant features:** `grade`, `sub_grade`, `int_rate` and FICO carry overlapping information.
- **Dropped `emp_length`:** the missing values may themselves carry signal. Keeping them as an "Unknown" category is an alternative.
- **No macroeconomic data:** the model sees only borrower-level information.

---

## 6. Possible next steps

- Switch to a time-based train/test split using `issue_d`.
- Compare against logistic regression as a baseline.
- Calibrate probabilities (e.g. `CalibratedClassifierCV`) and choose a decision threshold based on business cost.
- Inspect feature importance or SHAP values to explain what drives predictions.
- Estimate expected loss per loan by combining default probability, loan amount and interest rate.
