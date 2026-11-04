# %% [markdown]
# # Week 1: Dataset and Preprocessing
# Project: Loan Approval Prediction
# Goal: Load the dataset, understand it, and clean it for machine learning.

# %% Settings
from pathlib import Path
import pandas as pd

# Find the project root (works whether you run from the root or from notebooks/)
ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

DATA_PATH = ROOT / "data" / "raw" / "loan_data.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "loan_cleaned.csv"
TARGET = "loan_status"

# %% Step 1: Load the dataset
df = pd.read_csv(DATA_PATH)
print(df.head())

# %% Step 2: Understand the data
print("Shape (rows, columns):", df.shape)
df.info()
print(df.describe())

print("\nTarget distribution:")
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True).round(3))

# MY NOTES (write in your own words):
# - What does each column mean?
# - Target = loan_status (1 = approved, 0 = rejected)
# - Is the target balanced or imbalanced?

# %% Step 3a: Missing values
print(df.isnull().sum())

num_cols = df.select_dtypes(include="number").columns.drop(TARGET, errors="ignore")
cat_cols = df.select_dtypes(exclude="number").columns.drop(TARGET, errors="ignore")

for c in num_cols:
    df[c] = df[c].fillna(df[c].median())      # numeric: median
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])     # text: mode

df = df.dropna(subset=[TARGET])
print("Missing values left:", df.isnull().sum().sum())

# %% Step 3b: Duplicates
print("Duplicates before:", df.duplicated().sum())
df = df.drop_duplicates()
print("Shape after:", df.shape)

# %% Step 3c: Drop ID columns
id_cols = [c for c in df.columns if c.lower() in ("loan_id", "id", "customer_id")]
df = df.drop(columns=id_cols)
print("Dropped:", id_cols)

# %% Step 3d: Make the target numeric (0/1)
if df[TARGET].dtype == "object":
    df[TARGET] = (df[TARGET].astype(str).str.strip().str.upper()
                  .map({"Y": 1, "N": 0, "YES": 1, "NO": 0,
                        "APPROVED": 1, "REJECTED": 0}))
df[TARGET] = df[TARGET].astype(int)
print(df[TARGET].value_counts())

# %% Step 3e: Remove impossible values (outliers)
before = len(df)
df = df[(df["person_age"] <= 100) &
        (df["person_emp_exp"] <= df["person_age"] - 16)]   # worked since age 16 at the earliest
df = df.reset_index(drop=True)
print("Rows removed:", before - len(df), "| Shape now:", df.shape)

# %% Step 4: Encode text columns (models only understand numbers)
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
print("Text columns:", cat_cols)
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
print(df.head())

# %% Step 5: Save cleaned dataset (WEEK 1 DELIVERABLE)
CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_PATH, index=False)
print("Saved:", CLEAN_PATH, "| Final shape:", df.shape)

# %% Step 6: Split, THEN scale (fit the scaler on training data only)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop(TARGET, axis=1)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit ONLY on training data
X_test = scaler.transform(X_test)

print("Train:", X_train.shape, "| Test:", X_test.shape)

# %% [markdown]
# ## Summary (fill in)
# - Rows after cleaning: ...
# - Missing values: median (numeric), mode (text)
# - Duplicates removed: ...
# - Encoding: one-hot (get_dummies)
# - Split: 80% train / 20% test, scaler fitted on train only