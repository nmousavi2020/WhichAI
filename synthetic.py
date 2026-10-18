import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1500

df = pd.DataFrame({
    "age": rng.integers(21, 70, n),
    "annual_income": rng.integers(15000, 150001, n),
    "employment_years": rng.integers(0, 41, n),
    "credit_score": rng.integers(300, 851, n),
    "loan_amount": rng.integers(1000, 50001, n),
    "existing_debt": rng.integers(0, 40001, n),
    "loan_term_months": rng.choice([12, 24, 36, 48, 60], n),
    "employment_type": rng.choice(["employed", "self_employed", "unemployed"], n),
    "home_ownership": rng.choice(["own", "rent", "mortgage"], n),
    "loan_purpose": rng.choice(["car", "education", "business", "personal"], n),
})

risk = (
    0.000045 * df["loan_amount"]
    + 0.000035 * df["existing_debt"]
    - 0.006 * (df["credit_score"] - 600)
    - 0.000012 * (df["annual_income"] - 50000)
    - 0.025 * df["employment_years"]
    + 0.7 * (df["employment_type"] == "unemployed")
    + rng.normal(0, 0.8, n)
)

probability = 1 / (1 + np.exp(-risk))
df["default"] = rng.binomial(1, probability)

for col in ["annual_income", "credit_score", "existing_debt"]:
    df.loc[rng.choice(n, 30, replace=False), col] = np.nan

df.to_csv("credit_risk_1500.csv", index=False)

print(df.shape)
print(df["default"].value_counts())
print("Saved: credit_risk_1500.csv")
