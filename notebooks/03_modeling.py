# %% [markdown]
# # Week 3: Model Development
# Problem type: Classification (predict loan approved = 1 or rejected = 0)
# Goal: Train 3 ML models and compare them on TEST data.

# %% Settings
import sys
from pathlib import Path

_root = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
sys.path.insert(0, str(_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.preprocess import load_processed, split_and_scale

ROOT = _root
TARGET = "loan_status"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
REPORT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% Step 1 and 2: Load cleaned data, split FIRST, then scale (functions from src/preprocess.py)
df = load_processed()
print("Shape:", df.shape)
X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# %% Step 3: Baseline to beat (a "model" that always predicts Rejected)
baseline = (y_test == 0).mean()
print(f"Baseline accuracy (always predict Rejected): {baseline:.4f}")

# %% Step 4: Train 3 models (training data is used ONLY to fit), evaluate on TEST data
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred),
        "Recall": recall_score(y_test, test_pred),
        "F1-Score": f1_score(y_test, test_pred),
    })
    print(f"{name} trained.")

# %% Step 5: Comparison table (model vs accuracy), ALL values on TEST data
comparison = pd.DataFrame(results).set_index("Model").round(4)
comparison = comparison.sort_values("Test Accuracy", ascending=False)
print("\nMODEL COMPARISON (all metrics on test data)")
print(comparison)
print(f"\nBaseline (always Rejected): {baseline:.4f}")
comparison.to_csv(REPORT_DIR / "model_comparison.csv")

# %% Step 6: Chart of test accuracy
plt.figure(figsize=(7, 4))
sns.barplot(x=comparison.index, y=comparison["Test Accuracy"])
plt.axhline(baseline, color="red", linestyle="--", label="Baseline (always Rejected)")
plt.ylim(0.6, 1.0)
plt.title("Test Accuracy by Model")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "12_model_comparison.png", dpi=150)
plt.close()

# %% [markdown]
# ## Model Comparison Summary
# - Best model on test data: Random Forest (accuracy 92.9%, F1 0.828).
# - Clearly better than the baseline? Yes: baseline (always Rejected) is about 77.8%,
#   and all three models beat it.
# - Best recall: Decision Tree (0.7795), but its precision is low (0.7557). Random
#   Forest has the best balance (precision 0.8968, recall 0.769, F1 0.828).
# - Model for Week 4: Random Forest, because it is best on accuracy, precision and F1,
#   and it gives feature importances for feature selection.