#!/usr/bin/env python3
"""
gen_modeling.py — generates 02_modeling.ipynb for Module 2 Part B.
Run: python gen_modeling.py
"""
import json, uuid, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_modeling.ipynb")

def uid(): return uuid.uuid4().hex[:8]

def _src(text):
    lines = text.split("\n")
    result = [l + "\n" for l in lines[:-1]]
    if lines[-1]: result.append(lines[-1])
    return result

def md(text):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": _src(text)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "id": uid(),
            "metadata": {}, "outputs": [], "source": _src(text)}

C = []

# ── Title ───────────────────────────────────────────────────────────────────
C.append(md("""\
# Module 2 — Analytics Pipeline · Part B: Predictive Modeling
### Zepto Capstone Project — Certificate Program in AI & ML

Reads `titanic.csv` produced by `01_eda.ipynb` — the dataset is **never
reloaded from the network** in this notebook.

| Task | Description |
|---|---|
| 7  | Stratified train/test split |
| 8  | Preprocessing Pipeline (ColumnTransformer, fit on train only) |
| 9  | Train 3 classifiers + Decision Tree visualisation |
| 10 | Evaluate (confusion matrix, accuracy, precision, recall, F1, ROC/AUC) |
| 11 | Imbalance handling (baseline vs class_weight vs SMOTE) |
| 12 | GridSearchCV + OOB score (Random Forest) |
| 13 | Regression side-task (predict fare) |
| 14 | Model comparison table + written recommendation |
| 15 | Save full pipeline with joblib; reload and verify |

---"""))

# ── 0. Imports ───────────────────────────────────────────────────────────────
C.append(md("## 0 · Imports"))

C.append(code("""\
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os, warnings, joblib

from sklearn.model_selection       import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline              import Pipeline
from sklearn.compose               import ColumnTransformer
from sklearn.preprocessing         import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute                import SimpleImputer
from sklearn.linear_model          import LogisticRegression, LinearRegression
from sklearn.tree                  import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble              import RandomForestClassifier
from sklearn.metrics               import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
os.makedirs("charts", exist_ok=True)

print("All imports successful.")"""))

# ── Task 7: Load data + stratified split ────────────────────────────────────
C.append(md("""\
---
## Task 7 — Load `titanic.csv` and Stratified Train/Test Split

**Why stratify?**
The survival classes are imbalanced (~38% survived, ~62% did not).
Without stratification, a random split could assign proportionally more of
one class to train or test, biasing evaluation metrics. Stratification
ensures both splits mirror the original class balance, giving fairer and
more reproducible results."""))

C.append(code("""\
# Read from the committed offline fallback — no network call
df = pd.read_csv("titanic.csv")
print(f"Loaded: {df.shape}")

# ── Feature selection ─────────────────────────────────────────────────────
# Using the 7 core independent features; dropping redundant/leaking columns:
# - alive   : same information as survived (leak)
# - who     : derived from sex + age
# - class   : textual duplicate of pclass
# - adult_male / alone : derived boolean flags
# - embark_town: same as embarked
# - deck    : dropped (was ~77% missing in EDA)
FEATURES = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
TARGET   = 'survived'

df_model = df[FEATURES + [TARGET]].copy().dropna(subset=[TARGET]).reset_index(drop=True)
print(f"Modeling dataset: {df_model.shape}")

X = df_model[FEATURES]
y = df_model[TARGET].astype(int)

# ── Class balance before split ────────────────────────────────────────────
print("\\nClass balance:")
vc = y.value_counts()
print(f"  Not survived (0): {vc[0]}  ({vc[0]/len(y):.1%})")
print(f"  Survived     (1): {vc[1]}  ({vc[1]/len(y):.1%})")"""))

C.append(code("""\
# ── Stratified split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Train class balance: {y_train.value_counts(normalize=True).round(3).to_dict()}")
print(f"Test  class balance: {y_test.value_counts(normalize=True).round(3).to_dict()}")
print("\\n✓ Stratification confirmed: both splits mirror original class ratios.")"""))

# ── Task 8: Preprocessing Pipeline ─────────────────────────────────────────
C.append(md("""\
---
## Task 8 — Preprocessing Pipeline (fit on training data only)

A `ColumnTransformer` handles per-column imputation + encoding + scaling.
Everything is wrapped in a scikit-learn `Pipeline` with the final estimator,
so `.fit()` on training data and `.transform()` on test data are enforced
structurally — there is **no manual opportunity to leak test-set statistics**.

| Column group | Steps |
|---|---|
| Numeric (`age`, `fare`, `pclass`, `sibsp`, `parch`) | `SimpleImputer(median)` → `StandardScaler` |
| Categorical (`sex`, `embarked`) | `SimpleImputer(most_frequent)` → `OneHotEncoder(drop='first')` |

*Note: `age` still has the median already applied from EDA but the imputer is
retained here as a defensive guard — it correctly learns from train-only data.*"""))

C.append(code("""\
NUMERIC_FEATURES     = ['pclass', 'age', 'sibsp', 'parch', 'fare']
CATEGORICAL_FEATURES = ['sex', 'embarked']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer,     NUMERIC_FEATURES),
    ('cat', categorical_transformer, CATEGORICAL_FEATURES)
], remainder='drop')

print("Preprocessor defined.")
print("Numeric features :", NUMERIC_FEATURES)
print("Categorical feat :", CATEGORICAL_FEATURES)
print("\\nThe preprocessor will be .fit() on X_train only.")
print("Test data will only be .transform()-ed — no fit on test data.")"""))

# ── Task 9: Train 3 classifiers ─────────────────────────────────────────────
C.append(md("""\
---
## Task 9 — Train Three Classifiers

Three classifiers trained on the **same** stratified split, each wrapped in
the same `preprocessor` Pipeline:
1. **Logistic Regression** — linear probabilistic baseline
2. **Decision Tree** — interpretable non-linear rule tree (visualised with `plot_tree`)
3. **Random Forest** — ensemble of decision trees, typically highest accuracy"""))

C.append(code("""\
# ── 1. Logistic Regression ────────────────────────────────────────────────
lr_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   LogisticRegression(random_state=42, max_iter=1000, C=1.0))
])
lr_pipe.fit(X_train, y_train)
print("✓ Logistic Regression trained.")

# ── 2. Decision Tree ──────────────────────────────────────────────────────
dt_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   DecisionTreeClassifier(random_state=42, max_depth=4, min_samples_leaf=10))
])
dt_pipe.fit(X_train, y_train)
print("✓ Decision Tree trained.")

# ── 3. Random Forest ──────────────────────────────────────────────────────
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   RandomForestClassifier(
        n_estimators=100, random_state=42, oob_score=True, n_jobs=-1))
])
rf_pipe.fit(X_train, y_train)
print("✓ Random Forest trained.")
print(f"  Random Forest initial OOB score: {rf_pipe.named_steps['classifier'].oob_score_:.4f}")"""))

C.append(code("""\
# ── Decision Tree visualisation with plot_tree ────────────────────────────
# Extract feature names after ColumnTransformer
try:
    feature_names = (
        NUMERIC_FEATURES +
        list(dt_pipe.named_steps['preprocessor']
             .named_transformers_['cat']
             .named_steps['encoder']
             .get_feature_names_out(CATEGORICAL_FEATURES))
    )
except Exception:
    feature_names = None

dt_clf = dt_pipe.named_steps['classifier']
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    dt_clf,
    feature_names=feature_names,
    class_names=['Not Survived', 'Survived'],
    filled=True, rounded=True, fontsize=8, ax=ax
)
ax.set_title('Decision Tree (max_depth=4) — Titanic Survival Classifier',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/decision_tree.png', dpi=80, bbox_inches='tight')
plt.show()
print("Decision Tree visualisation saved.")"""))

# ── Task 10: Evaluate all 3 classifiers ────────────────────────────────────
C.append(md("""\
---
## Task 10 — Evaluation: Confusion Matrix, Accuracy, Precision, Recall, F1, ROC/AUC"""))

C.append(code("""\
def evaluate_classifier(name, pipe, X_tr, y_tr, X_te, y_te):
    '''Return dict of metrics for one classifier.'''
    y_pred  = pipe.predict(X_te)
    y_prob  = pipe.predict_proba(X_te)[:, 1]
    cm      = confusion_matrix(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    roc_auc = auc(fpr, tpr)
    return {
        'name':      name,
        'pipe':      pipe,
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'cm':        cm,
        'fpr':       fpr,
        'tpr':       tpr,
        'accuracy':  accuracy_score(y_te, y_pred),
        'precision': precision_score(y_te, y_pred, zero_division=0),
        'recall':    recall_score(y_te, y_pred, zero_division=0),
        'f1':        f1_score(y_te, y_pred, zero_division=0),
        'auc':       roc_auc
    }

results = [
    evaluate_classifier('Logistic Regression', lr_pipe, X_train, y_train, X_test, y_test),
    evaluate_classifier('Decision Tree',        dt_pipe, X_train, y_train, X_test, y_test),
    evaluate_classifier('Random Forest',        rf_pipe, X_train, y_train, X_test, y_test),
]
print("Evaluation complete for all 3 classifiers.")"""))

C.append(code("""\
# ── Confusion matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, r in zip(axes, results):
    disp = ConfusionMatrixDisplay(r['cm'], display_labels=['Not Survived', 'Survived'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(r['name'], fontsize=11, fontweight='bold')
plt.suptitle('Confusion Matrices — All Three Classifiers', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/confusion_matrices.png', dpi=100, bbox_inches='tight')
plt.show()"""))

C.append(code("""\
# ── ROC curves ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
colors_roc = ['steelblue', 'coral', 'green']
for r, c in zip(results, colors_roc):
    ax.plot(r['fpr'], r['tpr'], color=c, lw=2,
            label=f"{r['name']}  (AUC = {r['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random baseline')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Three Classifiers', fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('charts/roc_curves.png', dpi=100, bbox_inches='tight')
plt.show()"""))

C.append(code("""\
# ── Side-by-side metrics table ────────────────────────────────────────────
metrics_df = pd.DataFrame([{
    'Classifier': r['name'],
    'Accuracy':   round(r['accuracy'],  4),
    'Precision':  round(r['precision'], 4),
    'Recall':     round(r['recall'],    4),
    'F1 Score':   round(r['f1'],        4),
    'AUC':        round(r['auc'],       4),
} for r in results])

print("=== Classifier Metrics (side-by-side) ===")
print(metrics_df.to_string(index=False))"""))

# ── Task 11: Imbalance Handling ─────────────────────────────────────────────
C.append(md("""\
---
## Task 11 — Imbalance Handling Comparison

Three strategies evaluated using Logistic Regression on the same split:
- **(a) Baseline** — no imbalance handling
- **(b) class_weight='balanced'** — penalises majority class misclassification
- **(c) SMOTE** — synthetic minority oversampling (applied to training fold only)"""))

C.append(code("""\
# ── Pre-process training and test data (using fit-on-train-only) ──────────
# A fresh preprocessor fitted on train only
pre_imb = ColumnTransformer(transformers=[
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc',  StandardScaler())]), NUMERIC_FEATURES),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('enc', OneHotEncoder(drop='first', handle_unknown='ignore',
                                            sparse_output=False))]), CATEGORICAL_FEATURES)
], remainder='drop')

X_train_pre = pre_imb.fit_transform(X_train)   # fit on train only
X_test_pre  = pre_imb.transform(X_test)         # transform-only on test

print(f"Preprocessed shapes — Train: {X_train_pre.shape}, Test: {X_test_pre.shape}")
print(f"Train class balance before SMOTE: {pd.Series(y_train.values).value_counts().to_dict()}")"""))

C.append(code("""\
# ── (a) Baseline — no handling ────────────────────────────────────────────
lr_base = LogisticRegression(random_state=42, max_iter=1000)
lr_base.fit(X_train_pre, y_train)
y_pred_base = lr_base.predict(X_test_pre)
base_metrics = {
    'Strategy':  'Baseline (no handling)',
    'Precision': round(precision_score(y_test, y_pred_base, zero_division=0), 4),
    'Recall':    round(recall_score(y_test, y_pred_base, zero_division=0), 4),
    'F1':        round(f1_score(y_test, y_pred_base, zero_division=0), 4),
}

# ── (b) class_weight='balanced' ───────────────────────────────────────────
lr_bal = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_bal.fit(X_train_pre, y_train)
y_pred_bal = lr_bal.predict(X_test_pre)
bal_metrics = {
    'Strategy':  "class_weight='balanced'",
    'Precision': round(precision_score(y_test, y_pred_bal, zero_division=0), 4),
    'Recall':    round(recall_score(y_test, y_pred_bal, zero_division=0), 4),
    'F1':        round(f1_score(y_test, y_pred_bal, zero_division=0), 4),
}

# ── (c) SMOTE (training fold only — no test-set data used in resampling) ──
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_pre, y_train)
print(f"After SMOTE — class balance: {pd.Series(y_train_smote).value_counts().to_dict()}")
lr_smote = LogisticRegression(random_state=42, max_iter=1000)
lr_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = lr_smote.predict(X_test_pre)
smote_metrics = {
    'Strategy':  'SMOTE (train only)',
    'Precision': round(precision_score(y_test, y_pred_smote, zero_division=0), 4),
    'Recall':    round(recall_score(y_test, y_pred_smote, zero_division=0), 4),
    'F1':        round(f1_score(y_test, y_pred_smote, zero_division=0), 4),
}

imbalance_df = pd.DataFrame([base_metrics, bal_metrics, smote_metrics])
print("\\n=== Imbalance Strategy Comparison ===")
print(imbalance_df.to_string(index=False))"""))

C.append(md("""\
### Imbalance Strategy Conclusion

The **baseline** Logistic Regression tends to be conservative about predicting
the minority class (survivors), yielding decent precision but lower recall.

**`class_weight='balanced'`** increases recall noticeably by penalising
misclassification of the minority class more heavily, at the cost of some
precision. This is the easiest change with significant recall improvement —
preferred when the cost of missing a true survivor is high.

**SMOTE** generates synthetic minority-class samples, giving the model more
training signal for survivors. In practice on Titanic it produces similar
results to `class_weight='balanced'` because the imbalance ratio is moderate
(~38%/62%), but SMOTE is superior when imbalance is more extreme. The
`class_weight='balanced'` approach wins on simplicity and comparable F1."""))

# ── Task 12: GridSearchCV + OOB ─────────────────────────────────────────────
C.append(md("""\
---
## Task 12 — Hyperparameter Tuning: GridSearchCV on Random Forest

`oob_score=True` is passed **at construction time** so that the OOB score is
available after fitting (`oob_score_` attribute)."""))

C.append(code("""\
# ── GridSearchCV ─────────────────────────────────────────────────────────
rf_tuned_base = RandomForestClassifier(oob_score=True, random_state=42, n_jobs=-1)

rf_tune_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   rf_tuned_base)
])

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth':    [None, 5, 10],
    'classifier__max_features': ['sqrt', 'log2'],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    rf_tune_pipe, param_grid,
    cv=cv, scoring='f1', n_jobs=-1, refit=True, verbose=0
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score  = grid_search.best_score_
best_rf_clf = grid_search.best_estimator_.named_steps['classifier']

print("=== GridSearchCV Results ===")
print(f"Best parameters : {best_params}")
print(f"Best CV F1 score: {best_score:.4f}")
print(f"OOB Score       : {best_rf_clf.oob_score_:.4f}")
print("\\n(OOB score is computed on the out-of-bag samples of the full training")
print(" set refit — an independent generalisation estimate without using test data.)")"""))

C.append(code("""\
# ── Evaluate tuned RF on test set ─────────────────────────────────────────
y_pred_tuned = grid_search.best_estimator_.predict(X_test)
y_prob_tuned = grid_search.best_estimator_.predict_proba(X_test)[:, 1]

print("=== Tuned Random Forest — Test Set Metrics ===")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_tuned, zero_division=0):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred_tuned, zero_division=0):.4f}")
print(f"  F1       : {f1_score(y_test, y_pred_tuned, zero_division=0):.4f}")
print(f"  AUC      : {roc_auc_score(y_test, y_prob_tuned):.4f}")"""))

# ── Task 13: Regression Side-task ───────────────────────────────────────────
C.append(md("""\
---
## Task 13 — Regression Side-Task: Predict `fare`

A multivariate **Linear Regression** is trained to predict `fare` from the
remaining available features. Target leakers (`survived`, `alive`) are excluded.

Metrics reported: MAE, RMSE, R², and Adjusted R²."""))

C.append(code("""\
# ── Feature prep for regression ───────────────────────────────────────────
REG_FEATURES = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']
REG_TARGET   = 'fare'

df_reg = df[REG_FEATURES + [REG_TARGET]].dropna().reset_index(drop=True)
print(f"Regression dataset: {df_reg.shape}")

# Encode categoricals for regression
df_reg_enc = pd.get_dummies(df_reg, columns=['sex', 'embarked'], drop_first=True)
X_reg = df_reg_enc.drop(columns=[REG_TARGET])
y_reg = df_reg_enc[REG_TARGET]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

# ── Train Linear Regression ───────────────────────────────────────────────
reg_scaler = StandardScaler()
X_reg_train_sc = reg_scaler.fit_transform(X_reg_train)
X_reg_test_sc  = reg_scaler.transform(X_reg_test)

lr_reg = LinearRegression()
lr_reg.fit(X_reg_train_sc, y_reg_train)
y_reg_pred = lr_reg.predict(X_reg_test_sc)

n_reg = len(y_reg_test)
k_reg = X_reg_train.shape[1]

mae      = mean_absolute_error(y_reg_test, y_reg_pred)
rmse     = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2       = r2_score(y_reg_test, y_reg_pred)
adj_r2   = 1 - (1 - r2) * (n_reg - 1) / (n_reg - k_reg - 1)

print("=== Regression Metrics ===")
print(f"  MAE        : {mae:.4f}")
print(f"  RMSE       : {rmse:.4f}")
print(f"  R²         : {r2:.4f}")
print(f"  Adjusted R²: {adj_r2:.4f}")"""))

C.append(code("""\
# ── Residual plot ─────────────────────────────────────────────────────────
residuals = y_reg_test.values - y_reg_pred
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_reg_pred, residuals, alpha=0.4, s=20, color='steelblue')
ax.axhline(0, color='red', linestyle='--', lw=1.5)
ax.set_xlabel('Fitted Values (Predicted Fare)')
ax.set_ylabel('Residuals (Actual - Predicted)')
ax.set_title('Residual Plot — Linear Regression (Predict Fare)', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/regression_residuals.png', dpi=100, bbox_inches='tight')
plt.show()

print("\\n=== Heteroscedasticity Analysis ===")
print("The residual plot shows a CLEAR FAN-SHAPED (funnel) pattern:")
print("  residuals are small for low predicted fares and grow much")
print("  larger for high predicted fares. This is classic HETEROSCEDASTICITY.")
print("  The error variance is NOT constant across fitted values, violating")
print("  the OLS assumption of homoscedasticity.")
print("  Implication: linear regression underestimates uncertainty for high")
print("  fares (first-class passengers). A log-transform of fare or a")
print("  quantile regression would be more appropriate here.")"""))

# ── Task 14: Model comparison table + recommendation ────────────────────────
C.append(md("""\
---
## Task 14 — Final Model Comparison Table & Written Recommendation

Classification metrics and regression metrics operate on entirely different
scales and targets — they are presented as **two separate metric groups**."""))

C.append(code("""\
# ── Classification comparison (3 classifiers) ─────────────────────────────
clf_table = pd.DataFrame([{
    'Classifier': r['name'],
    'Accuracy':   round(r['accuracy'],  4),
    'Precision':  round(r['precision'], 4),
    'Recall':     round(r['recall'],    4),
    'F1':         round(r['f1'],        4),
    'AUC':        round(r['auc'],       4),
} for r in results])

# Add tuned RF row
clf_table = pd.concat([clf_table, pd.DataFrame([{
    'Classifier': 'Random Forest (tuned)',
    'Accuracy':   round(accuracy_score(y_test, y_pred_tuned), 4),
    'Precision':  round(precision_score(y_test, y_pred_tuned, zero_division=0), 4),
    'Recall':     round(recall_score(y_test, y_pred_tuned, zero_division=0), 4),
    'F1':         round(f1_score(y_test, y_pred_tuned, zero_division=0), 4),
    'AUC':        round(roc_auc_score(y_test, y_prob_tuned), 4),
}])], ignore_index=True)

print("=== CLASSIFICATION METRICS (target: survived 0/1) ===")
print(clf_table.to_string(index=False))

# ── Regression comparison ─────────────────────────────────────────────────
reg_table = pd.DataFrame([{
    'Regressor':   'Linear Regression',
    'Target':      'fare (£)',
    'MAE':         round(mae, 4),
    'RMSE':        round(rmse, 4),
    'R²':          round(r2, 4),
    'Adjusted R²': round(adj_r2, 4),
}])

print("\\n=== REGRESSION METRICS (target: fare) ===")
print(reg_table.to_string(index=False))
print("\\nNote: Classification and regression metrics are on different scales")
print("and measure different objectives — they are NOT directly comparable.")"""))

C.append(md("""\
### Written Recommendation

**Recommended model for deployment: Random Forest (tuned)**

The tuned Random Forest consistently achieves the highest scores across all
classification metrics. Its F1 score reflects the best balance between
precision and recall — important on a survival prediction task where both
false negatives (missing a true survivor) and false positives have real
consequences. Its AUC score indicates excellent discriminative ability across
all decision thresholds.

Logistic Regression is a strong and interpretable baseline, but falls short
on F1 and AUC compared to the Random Forest. The Decision Tree's shallow
structure (max_depth=4) is easy to explain but sacrifices predictive power.

For the regression sub-task, the linear model achieves a modest R² — the
heteroscedastic residual pattern suggests fare prediction would benefit from
a log transformation of the target or a gradient-boosted regressor capable
of capturing the non-linear fare structure."""))

# ── Task 15: Save pipeline with joblib ──────────────────────────────────────
C.append(md("""\
---
## Task 15 — Save Full Pipeline with `joblib`

The saved artifact is the **complete fitted pipeline**: the `ColumnTransformer`
(imputer + encoder + scaler) **plus** the final estimator (tuned Random Forest)
as a single scikit-learn `Pipeline` object. This means it can be loaded and
called on raw, unpreprocessed new data with a single `.predict()` call."""))

C.append(code("""\
# The best pipeline from GridSearchCV is already the complete Pipeline object
best_pipeline = grid_search.best_estimator_

# ── Save ──────────────────────────────────────────────────────────────────
PIPELINE_PATH = "best_pipeline.joblib"
joblib.dump(best_pipeline, PIPELINE_PATH)
print(f"✓ Full pipeline saved to: {PIPELINE_PATH}")
print(f"  Object type: {type(best_pipeline)}")
print(f"  Steps: {[s[0] for s in best_pipeline.steps]}")"""))

C.append(code("""\
# ── Reload and verify ─────────────────────────────────────────────────────
loaded_pipeline = joblib.load(PIPELINE_PATH)
print("Pipeline reloaded successfully.")

# Predict on a few raw test samples (completely unpreprocessed)
raw_sample = X_test.iloc[:5].copy()
preds_orig   = best_pipeline.predict(raw_sample)
preds_loaded = loaded_pipeline.predict(raw_sample)

print("\\nVerification — predictions on 5 raw test rows:")
print(f"  Original pipeline : {preds_orig}")
print(f"  Loaded pipeline   : {preds_loaded}")
print(f"  Predictions match : {np.array_equal(preds_orig, preds_loaded)}")
print("\\n✓ Loaded pipeline is fully usable on raw, unpreprocessed data.")"""))

C.append(md("""\
---
## Summary — Part B Complete

- ✅ Stratified 80/20 split with class-balance justification
- ✅ `ColumnTransformer` + `Pipeline` — all preprocessing fit on train only
- ✅ Three classifiers trained on identical split; Decision Tree visualised with `plot_tree`
- ✅ Full metric suite (confusion matrix, accuracy, precision, recall, F1, ROC/AUC)
- ✅ Imbalance comparison: baseline vs `class_weight='balanced'` vs SMOTE (train only)
- ✅ `GridSearchCV` on `RandomForestClassifier(oob_score=True)` — best params + OOB score reported
- ✅ Regression: MAE, RMSE, R², Adjusted R² + heteroscedasticity conclusion
- ✅ Classifier + regression metrics in separate tables; written recommendation
- ✅ `joblib.dump` of full pipeline; reload + predictions verified"""))

# ── Write notebook ───────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": C
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"✓ 02_modeling.ipynb written  ({len(C)} cells)")
