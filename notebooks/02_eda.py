# %% [markdown]
# # Week 2: Exploratory Data Analysis (EDA)
# Goal: Find patterns and relationships in the data using charts.

# %% Settings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "loan_status"
SHOW = False   # False: only save PNGs (use when running from cmd). True: also display (use in VS Code cells)
sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150)
    if SHOW:
        plt.show()
    plt.close()


# %% Load data (raw categories are easier to read in charts) + same cleaning as Week 1
df = pd.read_csv(ROOT / "data" / "raw" / "loan_data.csv")
df = df[(df["person_age"] <= 100) &
        (df["person_emp_exp"] <= df["person_age"] - 16)].reset_index(drop=True)
df["status"] = df[TARGET].map({0: "Rejected", 1: "Approved"})
print(df.shape)

# %% Summary statistics
print(df.describe())

# %% Chart 1: Target balance (count plot)
plt.figure(figsize=(5, 4))
sns.countplot(x="status", data=df)
plt.title("Loan Status: Approved vs Rejected")
save("01_target_balance")

# %% Chart 2: Age distribution (histogram)
plt.figure(figsize=(6, 4))
sns.histplot(df["person_age"], kde=True, bins=30)
plt.title("Applicant Age Distribution")
save("02_age_hist")

# %% Chart 3: Credit score distribution (histogram)
plt.figure(figsize=(6, 4))
sns.histplot(df["credit_score"], kde=True, bins=30)
plt.title("Credit Score Distribution")
save("03_credit_score_hist")

# %% Chart 4: Income vs loan status (boxplot, log scale because of extreme incomes)
plt.figure(figsize=(6, 4))
sns.boxplot(x="status", y="person_income", data=df)
plt.yscale("log")
plt.title("Income by Loan Status (log scale)")
save("04_income_vs_status")

# %% Chart 5: Interest rate vs loan status (boxplot)
plt.figure(figsize=(6, 4))
sns.boxplot(x="status", y="loan_int_rate", data=df)
plt.title("Interest Rate by Loan Status")
save("05_interest_rate_vs_status")

# %% Chart 6: Loan as % of income vs loan status (boxplot)
plt.figure(figsize=(6, 4))
sns.boxplot(x="status", y="loan_percent_income", data=df)
plt.title("Loan Amount as % of Income by Loan Status")
save("06_loan_percent_income_vs_status")

# %% Chart 7: Approval rate by previous defaults (bar plot)
plt.figure(figsize=(5, 4))
sns.barplot(x="previous_loan_defaults_on_file", y=TARGET, data=df, errorbar=None)
plt.ylabel("Approval rate")
plt.title("Approval Rate by Previous Loan Default")
save("07_approval_by_previous_default")

# %% Chart 8: Approval rate by home ownership (bar plot)
plt.figure(figsize=(6, 4))
sns.barplot(x="person_home_ownership", y=TARGET, data=df, errorbar=None)
plt.ylabel("Approval rate")
plt.title("Approval Rate by Home Ownership")
save("08_approval_by_home_ownership")

# %% Chart 9: Approval rate by loan purpose (bar plot)
plt.figure(figsize=(8, 4))
sns.barplot(x="loan_intent", y=TARGET, data=df, errorbar=None)
plt.xticks(rotation=30)
plt.ylabel("Approval rate")
plt.title("Approval Rate by Loan Purpose")
save("09_approval_by_loan_intent")

# %% Chart 10: Loan amount vs income (scatter, sample of 3000 for speed)
sample = df.sample(3000, random_state=42)
plt.figure(figsize=(7, 5))
sns.scatterplot(x="person_income", y="loan_amnt", hue="status", data=sample, alpha=0.5)
plt.xscale("log")
plt.title("Loan Amount vs Income (log income)")
save("10_loan_vs_income_scatter")

# %% Chart 11: Correlation heatmap (numeric columns)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
save("11_correlation_heatmap")

# %% Numbers to support your insights
print("\nApproval rate by previous default:")
print(df.groupby("previous_loan_defaults_on_file")[TARGET].mean().round(3))
print("\nApproval rate by home ownership:")
print(df.groupby("person_home_ownership")[TARGET].mean().round(3))
print("\nApproval rate by loan purpose:")
print(df.groupby("loan_intent")[TARGET].mean().round(3))
print("\nAverages by status:")
print(df.groupby("status")[["person_income", "loan_int_rate",
                            "loan_percent_income", "credit_score"]].mean().round(2))
print("\nCorrelation of each numeric column with the target:")
print(df.corr(numeric_only=True)[TARGET].sort_values(ascending=False).round(3))

# %% [markdown]
# ## Key Insights
# 1. Applicants with a previous loan default on file were approved 0% of the time,
#    versus 45.2% for applicants with no previous default (Chart 07). This is the
#    strongest single pattern in the data.
# 2. The target is imbalanced: only about 22% of applications are approved and 78%
#    are rejected (Chart 01), so accuracy alone will be misleading in Weeks 3 and 4.
# 3. The loan amount as a share of income matters most among the numeric columns:
#    approved applicants averaged 0.20 vs 0.12 for rejected ones, and it has the
#    highest correlation with loan status (0.385) (Chart 06 and Chart 11).
# 4. Interest rate is the second strongest numeric factor: approved loans averaged
#    12.86% vs 10.48% for rejected loans (correlation 0.332) (Chart 05).
# 5. Income is negatively related to approval (correlation -0.169): approved
#    applicants earned about 59,900 on average vs about 85,600 for rejected ones
#    (Chart 04).
# 6. Approval rates differ by home ownership: OTHER (33.3%) and RENT (32.4%) are far
#    higher than MORTGAGE (11.6%) and OWN (7.5%) (Chart 08).
# 7. Approval rates differ by loan purpose: debt consolidation is highest (30.3%)
#    and venture loans lowest (14.4%) (Chart 09). Credit score shows almost no
#    relationship with approval (correlation -0.007), which is surprising.