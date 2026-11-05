"""Predict loan approval for one applicant using the saved model."""
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def predict_loan(applicant: dict):
    """Return (label, probability_of_approval) for one applicant (raw, unscaled values)."""
    model = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    features = joblib.load(MODEL_DIR / "features.pkl")
    all_columns = joblib.load(MODEL_DIR / "all_columns.pkl")

    row = pd.get_dummies(pd.DataFrame([applicant]), dtype=int)
    row = row.reindex(columns=all_columns, fill_value=0)   # same columns as training
    scaled = pd.DataFrame(scaler.transform(row), columns=all_columns)

    prob = model.predict_proba(scaled[features])[0][1]
    return ("Approved" if prob >= 0.5 else "Rejected"), float(prob)


if __name__ == "__main__":
    sample = {
        "person_age": 30, "person_gender": "male", "person_education": "Bachelor",
        "person_income": 60000, "person_emp_exp": 5, "person_home_ownership": "RENT",
        "loan_amnt": 10000, "loan_intent": "PERSONAL", "loan_int_rate": 11.0,
        "loan_percent_income": 0.17, "cb_person_cred_hist_length": 5,
        "credit_score": 650, "previous_loan_defaults_on_file": "No",
    }
    print(predict_loan(sample))