# Loan Approval Prediction

Machine learning project built during the Verstech AI/ML Internship (Track 1).

## Progress
- [x] Week 1: Dataset and Preprocessing
- [x] Week 2: Exploratory Data Analysis
- [x] Week 3: Model Development
- [x] Week 4: Evaluation and Improvement
- [ ] Week 5: End-to-End Project
- [ ] Week 6: Deployment and Documentation

## Dataset
Loan Approval Classification Dataset (Kaggle), 45,000 rows, 14 columns.
Target column: `loan_status` (1 = approved, 0 = rejected).
The target is imbalanced: about 78% rejected and 22% approved.

## Project Structure
- `data/raw/` : original dataset
- `data/processed/` : cleaned dataset (`loan_cleaned.csv`)
- `notebooks/` : Week 1 preprocessing, Week 2 EDA
- `reports/figures/` : EDA charts
- `reports/model_comparison.csv` : Week 3 model comparison table
- `reports/evaluation_report.csv` : Week 4 before vs after report