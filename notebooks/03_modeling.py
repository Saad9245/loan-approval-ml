# %% [markdown]
# # Week 3: Model Development
# Problem type: Classification (predict loan approved = 1 or rejected = 0)
# Goal: Train 3 ML models and compare them on TEST data.

# %% Settings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

TARGET = "loan_status"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
REPORT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% Step 1: Load the cleaned data from Week 1
df = pd.read_csv(ROOT / "data" / "processed" / "loan_cleaned.csv")

# Safety: make sure impossible values are removed (same rule as Week 1)
df = df[(df["person_age"] <= 100) &
        (df["person_emp_exp"] <= df["person_age"] - 16)].reset_index(drop=True)
print("Shape:", df.shape)

# %% Step 2: Split FIRST, then scale (fit scaler on training data only)
X = df.drop(TARGET, axis=1)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
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
    model.fit(X_train, y_train)          # learn from training data
    test_pred = model.predict(X_test)    # judge on unseen test data
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
# ## Model Comparison Summary (write in your own words after running)
# - Best model on test data: ...
# - Is the best model clearly better than the baseline? ...
# - Which model has the best recall and F1 (finds the most approved loans)? ...
# - Which model do I take to Week 4 for improvement, and why? ...