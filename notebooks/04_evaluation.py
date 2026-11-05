# %% [markdown]
# # Week 4: Evaluation and Improvement
# Goal: Measure model quality and improve the best model (Random Forest).

# %% Settings
import sys
from pathlib import Path

_root = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
sys.path.insert(0, str(_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import seaborn as sns

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

from src.preprocess import load_processed, split_and_scale

ROOT = _root
TARGET = "loan_status"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR = ROOT / "models"
for d in (REPORT_DIR, FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)


def evaluate(name, model, X_te, y_te):
    pred = model.predict(X_te)
    return {"Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred),
            "Recall": recall_score(y_te, pred),
            "F1-Score": f1_score(y_te, pred)}


def plot_confusion(model, X_te, y_te, title, filename):
    cm = confusion_matrix(y_te, model.predict(X_te))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()


# %% Step 1: Load data and split (same split as Week 3), then scale on training data only
df = load_processed()
X = df.drop(TARGET, axis=1)
X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# %% Step 2: BEFORE model (default Random Forest, all features)
rf_before = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_before.fit(X_train, y_train)
before = evaluate("Before (default RF, all features)", rf_before, X_test, y_test)
print(before)

# %% Step 3: Classification report and confusion matrix (BEFORE)
print("\nCLASSIFICATION REPORT (before), test data")
print(classification_report(y_test, rf_before.predict(X_test),
                            target_names=["Rejected", "Approved"]))
plot_confusion(rf_before, X_test, y_test,
               "Confusion Matrix (Before)", "13_confusion_matrix_before.png")

# %% Step 4: Cross-validation (5 splits, training data only)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(rf_before, X_train, y_train, cv=cv, scoring="accuracy")
cv_f1 = cross_val_score(rf_before, X_train, y_train, cv=cv, scoring="f1")
print(f"CV accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
print(f"CV F1:       {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")

# %% Step 5: Feature selection with feature_importances_
importances = pd.Series(rf_before.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances:")
print(importances.round(4))

plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(FIG_DIR / "15_feature_importances.png", dpi=150)
plt.close()

selected = importances[importances >= 0.01].index.tolist()
print(f"\nKept {len(selected)} of {len(X.columns)} features:", selected)

rf_selected = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_selected.fit(X_train[selected], y_train)
after_selection = evaluate("After feature selection", rf_selected, X_test[selected], y_test)
print(after_selection)

# %% Step 5b: Choose the feature set to carry forward
if after_selection["F1-Score"] >= before["F1-Score"]:
    use_cols = selected
    print("Feature selection did not hurt: using the selected features.")
else:
    use_cols = list(X.columns)
    print("Feature selection lowered F1: keeping ALL features for tuning.")

# %% Step 6: Hyperparameter tuning (GridSearchCV on training data, scored by F1)
params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 2],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                    params, cv=5, scoring="f1")
grid.fit(X_train[use_cols], y_train)
print("\nBest parameters:", grid.best_params_)
print("Best CV F1:", round(grid.best_score_, 4))

best_model = grid.best_estimator_
after_tuning = evaluate("After tuning (final model)", best_model, X_test[use_cols], y_test)
print(after_tuning)

# %% Step 7: Evaluation report (BEFORE vs AFTER, all on TEST data)
report = pd.DataFrame([before, after_selection, after_tuning]).set_index("Model").round(4)
print("\nEVALUATION REPORT: BEFORE vs AFTER (test data)")
print(report)
report.to_csv(REPORT_DIR / "evaluation_report.csv")

print("\nCLASSIFICATION REPORT (final model), test data")
final_text = classification_report(y_test, best_model.predict(X_test[use_cols]),
                                   target_names=["Rejected", "Approved"])
print(final_text)
with open(REPORT_DIR / "evaluation_report.txt", "w") as f:
    f.write("EVALUATION REPORT: BEFORE vs AFTER (test data)\n")
    f.write(report.to_string())
    f.write(f"\n\nBest parameters: {grid.best_params_}\n")
    f.write(f"Features used ({len(use_cols)}): {use_cols}\n\n")
    f.write("CLASSIFICATION REPORT (final model)\n")
    f.write(final_text)

plot_confusion(best_model, X_test[use_cols], y_test,
               "Confusion Matrix (After)", "14_confusion_matrix_after.png")

# %% Step 8: Save the optimized model (used again in Week 6)
joblib.dump(best_model, MODEL_DIR / "model.pkl", compress=3)
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(use_cols, MODEL_DIR / "features.pkl")
joblib.dump(list(X.columns), MODEL_DIR / "all_columns.pkl")
print("Saved model, scaler and feature lists to models/")

# %% [markdown]
# ## Evaluation Summary
# - Before vs after: tuning raised F1 only slightly (0.8280 to 0.8287) and accuracy
#   from 0.9290 to 0.9291. The default Random Forest was already close to its best.
#   Feature selection lowered F1 to 0.8083, so all 22 features were kept.
# - Confusion matrix: the final model finds about 77% of approved loans (recall 0.7715),
#   so roughly 457 of 2,000 approved test loans are missed. Precision is 0.895, so
#   when it predicts Approved it is usually right.
# - Top 3 features: previous_loan_defaults_on_file_Yes (0.224), loan_percent_income
#   (0.164), loan_int_rate (0.158). This matches the EDA findings.
# - Best hyperparameters: n_estimators=200, max_depth=None, min_samples_leaf=1.
# - Accuracy alone is not enough because 78% of loans are rejected, so a model that
#   always predicts Rejected would score about 78%. Precision, recall and F1 show how
#   well the rare Approved class is found.