"""Reusable preprocessing functions for the Loan Approval project."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "loan_status"


def get_root():
    """Return the project root folder (works from the root or from notebooks/)."""
    root = Path.cwd()
    if not (root / "data").exists():
        root = root.parent
    return root


def remove_impossible_values(df):
    """Drop rows with impossible age or work experience."""
    df = df[(df["person_age"] <= 100) &
            (df["person_emp_exp"] <= df["person_age"] - 16)]
    return df.reset_index(drop=True)


def clean_data(df):
    """Full cleaning: missing values, duplicates, IDs, target, outliers, encoding."""
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns.drop(TARGET, errors="ignore")
    cat_cols = df.select_dtypes(exclude="number").columns.drop(TARGET, errors="ignore")
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])
    df = df.dropna(subset=[TARGET]).drop_duplicates()

    id_cols = [c for c in df.columns if c.lower() in ("loan_id", "id", "customer_id")]
    df = df.drop(columns=id_cols)
    df[TARGET] = df[TARGET].astype(int)

    df = remove_impossible_values(df)
    text_cols = df.select_dtypes(exclude="number").columns.tolist()
    return pd.get_dummies(df, columns=text_cols, drop_first=True, dtype=int)


def load_processed():
    """Load the cleaned CSV saved in Week 1."""
    df = pd.read_csv(get_root() / "data" / "processed" / "loan_cleaned.csv")
    return remove_impossible_values(df)


def split_and_scale(df, test_size=0.2, random_state=42):
    """Split FIRST, then scale (scaler is fitted on training data only)."""
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True), scaler