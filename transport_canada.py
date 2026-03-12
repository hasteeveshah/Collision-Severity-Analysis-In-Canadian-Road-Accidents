# import necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( 
    accuracy_score, classification_report, confusion_matrix, 
    f1_score, mean_absolute_error, balanced_accuracy_score, roc_auc_score)

# Optional: makes outputs easier to read
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

RQ1_TITLE = "Environment and Traffic Context Factors Affecting Crash Severity"

# Load the dataset with specified dtypes to optimize memory usage

# df = pd.read_csv("NCDB_1999_to_2014.csv", low_memory=False)
# # Save as Parquet for faster future loading
# df.to_parquet("NCDB.parquet", compression="snappy")

# Now we can load the Parquet version for faster access in future runs 
df = pd.read_parquet("NCDB.parquet")
print(df.shape)

# Select only the columns needed for our analysis to save memory and speed up processing
cols_needed = [
    "P_ISEV", "P_SAFE", "P_USER", "P_AGE", "P_SEX",
    "V_TYPE", "V_YEAR",
    "C_WTHR", "C_RSUR", "C_RALN", "C_TRAF", "C_MNTH", "C_WDAY", "C_HOUR", "C_VEHS",
    "C_YEAR","C_SEV"
]

df = df[cols_needed].copy()

# Convert Categorical column to numerical coulumn

numeric_cols = ["P_AGE", "C_MNTH", "C_WDAY", "C_HOUR", "C_VEHS", "V_YEAR", "P_ISEV", "C_SEV"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert int64 and float64 to int32 and float32 to save memory
for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int32")

for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")

# print("Data types after conversion:")
# print(df.dtypes)

# RQ1: Logestic Regression using C_SEV (EXPLORATORY model Fetal Vs Non-Fetal) Due to extreme class imbalance

# subset the data for RQ1 and RQ2 based on the severity levels

print("RQ1: collision-level EXPLORATORY model")

# For RQ1, we focus on collision-level data with C_SEV as the outcome.
df_rq1 = df[df["C_SEV"].isin([1,2])].copy()

# Make binary outcome: 1 = fatal collision, 0 = non-fatal injury collision
df_rq1["Y_FATAL_COLLISION"] = (df_rq1["C_SEV"] == 1).astype(int)

print("\nCollision severity distribution (RQ1):")
print(df_rq1["Y_FATAL_COLLISION"].value_counts())
print(df_rq1["Y_FATAL_COLLISION"].value_counts(normalize=True))

# Predictor set (environment + time + vehicles involved)
rq1_features = ["C_WTHR","C_RSUR","C_RALN","C_TRAF","C_MNTH","C_WDAY","C_HOUR","C_VEHS"]

# Drop missing rows for modeling columns
df_rq1 = df_rq1.dropna(subset=rq1_features + ["Y_FATAL_COLLISION"]).copy()

def group_rare_levels_local(dataframe, col, min_count=2000):
    vc = dataframe[col].value_counts(dropna=False)
    rare_levels = vc[vc < min_count].index
    dataframe[col] = dataframe[col].where(~dataframe[col].isin(rare_levels), other="RARE")
    return dataframe

for c in ["C_WTHR", "C_RSUR", "C_RALN", "C_TRAF"]:
    df_rq1 = group_rare_levels_local(df_rq1, c, min_count=2000)

# Sample
df_rq1 = df_rq1.sample(200_000, random_state=42)

# Build X (design matrix) and y
y_rq1 = df_rq1["Y_FATAL_COLLISION"].astype(int)

# One-hot encode categorical variables (drop_first reduces multicollinearity)
X_rq1 = pd.get_dummies(df_rq1[rq1_features], drop_first=True)

# Ensure numeric matrix
X_rq1 = X_rq1.apply(pd.to_numeric, errors="coerce").astype("float32")

# Add intercept (constant)
X_rq1 = sm.add_constant(X_rq1, has_constant="add")

# Train/test split (stratify ensures fatal rate stays similar)
X_train_rq1, X_test_rq1, y_train_rq1, y_test_rq1 = train_test_split(
    X_rq1, y_rq1, test_size=0.30, random_state=42, stratify=y_rq1
)

print("\nRQ1 shapes:", X_train_rq1.shape, X_test_rq1.shape)

# Fit REGULARIZED logistic regression
rq1_model = sm.Logit(y_train_rq1, X_train_rq1)
rq1_res = rq1_model.fit_regularized(
    method="l1",
    alpha=0.05,
    maxiter=200)

print("\nRQ1 Regularized model fitted successfully.")
print("Non-zero coefficients:", int((rq1_res.params != 0).sum()), "out of", len(rq1_res.params))

# Predict probabilities on test set
p_test_rq1 = rq1_res.predict(X_test_rq1)

# Evaluate at default threshold (0.50)
y_pred_50 = (p_test_rq1 >= 0.50).astype(int)

print("\nRQ1 Evaluation (threshold = 0.50):")
print("Accuracy:", accuracy_score(y_test_rq1, y_pred_50))
print("Balanced Accuracy:", balanced_accuracy_score(y_test_rq1, y_pred_50))
print("Macro F1:", f1_score(y_test_rq1, y_pred_50, average="macro"))
print("ROC AUC:", roc_auc_score(y_test_rq1, p_test_rq1))
print("Confusion Matrix:\n", confusion_matrix(y_test_rq1, y_pred_50))

# Threshold tuning for imbalanced data
thresholds = np.linspace(0.01, 0.50, 50)
best_t, best_bacc = None, -1

for t in thresholds:
    y_pred_t = (p_test_rq1 >= t).astype(int)
    bacc = balanced_accuracy_score(y_test_rq1, y_pred_t)
    if bacc > best_bacc:
        best_bacc, best_t = bacc, t

y_pred_best = (p_test_rq1 >= best_t).astype(int)

print(f"\nRQ1 Evaluation (best threshold for Balanced Accuracy = {best_t:.3f}):")
print("Balanced Accuracy:", balanced_accuracy_score(y_test_rq1, y_pred_best))
print("Macro F1:", f1_score(y_test_rq1, y_pred_best, average="macro"))
print("ROC AUC:", roc_auc_score(y_test_rq1, p_test_rq1))
print("Confusion Matrix:\n", confusion_matrix(y_test_rq1, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test_rq1, y_pred_best, zero_division=0))

coef = rq1_res.params.copy()
coef = coef[coef != 0].sort_values(key=np.abs, ascending=False)

print("\nTop non-zero coefficients (by absolute value):")
print(coef.head(20))

# # Check the distribution of categorical predictors
# for c in ["C_WTHR","V_TYPE","P_SAFE","P_SEX","C_TRAF"]:
#     print("\n", c, df[c].astype("string").value_counts().head(15))


# Utility: Odds Ratio Table (MLE models)
def make_or_table(res, model_name="Model", top_n=25):
    """
    Build odds-ratio table for statsmodels Logit fit (MLE).
    If model did not converge, prints warning.
    """
    # Convergence check (works for MLE)
    try:
        conv = res.mle_retvals.get("converged", None)
        if conv is False:
            print(f"\nWARNING: {model_name} did NOT converge. OR/CI may be unstable.")
    except Exception:
        pass

    params = res.params

    # Try to compute CI + p-values (may fail if regularized or Hessian problems)
    try:
        conf = res.conf_int()
        conf.columns = ["CI_low", "CI_high"]
        pvals = res.pvalues

        or_table = pd.DataFrame({
            "OR": np.exp(params),
            "CI_low": np.exp(conf["CI_low"]),
            "CI_high": np.exp(conf["CI_high"]),
            "p_value": pvals
        }).sort_values("OR", ascending=False)

    except Exception as e:
        print(f"\nNote: CI/p-values not available for {model_name} ({repr(e)}). Showing OR only.")
        or_table = pd.DataFrame({
            "OR": np.exp(params)
        }).sort_values("OR", ascending=False)

    print(f"\n================ {model_name} Odds Ratios =================")
    print(or_table.head(top_n).round(4))
    print("\nBottom 15 (most protective):")
    print(or_table.tail(15).round(4))

    return or_table

# RQ1: Environment and Traffic Context model (person-level)
print(f"\n================ RQ1: {RQ1_TITLE} =================")

# Person-level dataset
df_rq1 = df[df["P_ISEV"].isin([1, 2, 3])].copy()

# Binary outcome
df_rq1["Y_INJURY"] = df_rq1["P_ISEV"].isin([2, 3]).astype(int)

# Environmental predictors
rq1_features = [
    "C_WTHR", "C_RSUR", "C_RALN", "C_TRAF",
    "C_MNTH", "C_WDAY", "C_HOUR", "C_VEHS"
]

# Drop missing
df_rq1 = df_rq1.dropna(subset=rq1_features + ["Y_INJURY"]).copy()

# Rare grouping (environment only)
for c in ["C_WTHR", "C_RSUR", "C_RALN", "C_TRAF"]:
    vc = df_rq1[c].value_counts()
    rare = vc[vc < 3000].index
    df_rq1[c] = df_rq1[c].where(~df_rq1[c].isin(rare), "RARE")

# Sample
df_rq1 = df_rq1.sample(200_000, random_state=42)

y_rq1 = df_rq1["Y_INJURY"]
X_rq1 = pd.get_dummies(df_rq1[rq1_features], drop_first=True)

X_rq1 = X_rq1.apply(pd.to_numeric, errors="coerce").astype("float32")
X_rq1 = sm.add_constant(X_rq1)

# Split
X_train_rq1, X_test_rq1, y_train_rq1, y_test_rq1 = train_test_split(
    X_rq1, y_rq1, test_size=0.30, stratify=y_rq1, random_state=42
)

# Fit
model_rq1 = sm.Logit(y_train_rq1, X_train_rq1)
res_rq1 = model_rq1.fit(maxiter=200)

print(res_rq1.summary())

rq1_or = make_or_table(res_rq1, model_name=f"RQ1 {RQ1_TITLE}", top_n=25)

# Evaluation
p_test_rq1 = res_rq1.predict(X_test_rq1)
print("RQ1 ROC AUC:", roc_auc_score(y_test_rq1, p_test_rq1))

# RQ2: Human & Vehicle Model (person-level)

print("\n================ RQ2: Human & Vehicle Model =================")

df_rq2 = df[df["P_ISEV"].isin([1, 2, 3])].copy()
df_rq2["Y_INJURY"] = df_rq2["P_ISEV"].isin([2, 3]).astype(int)

# Vehicle age engineering
df_rq2["VEHICLE_AGE"] = df_rq2["C_YEAR"] - df_rq2["V_YEAR"]
df_rq2.loc[(df_rq2["VEHICLE_AGE"] < 0) | 
           (df_rq2["VEHICLE_AGE"] > 60), "VEHICLE_AGE"] = np.nan

rq2_features = [
    "P_SAFE", "P_USER", "P_SEX",
    "V_TYPE", "VEHICLE_AGE",
    "P_AGE"
]

df_rq2 = df_rq2.dropna(subset=rq2_features + ["Y_INJURY"]).copy()

# Rare grouping (vehicle only)
vc = df_rq2["V_TYPE"].value_counts()
rare = vc[vc < 3000].index
df_rq2["V_TYPE"] = df_rq2["V_TYPE"].where(
    ~df_rq2["V_TYPE"].isin(rare), "RARE"
)

df_rq2 = df_rq2.sample(200_000, random_state=42)

y_rq2 = df_rq2["Y_INJURY"]
X_rq2 = pd.get_dummies(df_rq2[rq2_features], drop_first=True)

X_rq2 = X_rq2.apply(pd.to_numeric, errors="coerce").astype("float32")
X_rq2 = sm.add_constant(X_rq2)

X_train_rq2, X_test_rq2, y_train_rq2, y_test_rq2 = train_test_split(
    X_rq2, y_rq2, test_size=0.30, stratify=y_rq2, random_state=42
)

model_rq2 = sm.Logit(y_train_rq2, X_train_rq2)
res_rq2 = model_rq2.fit(maxiter=200)

print(res_rq2.summary())

rq2_or = make_or_table(res_rq2, model_name="RQ2 Human/Vehicle", top_n=25)

p_test_rq2 = res_rq2.predict(X_test_rq2)
print("RQ2 ROC AUC:", roc_auc_score(y_test_rq2, p_test_rq2))

# Combined Model (person-level)

print("\n================ Combined Model =================")

df_comb = df[df["P_ISEV"].isin([1, 2, 3])].copy()
df_comb["Y_INJURY"] = df_comb["P_ISEV"].isin([2, 3]).astype(int)

df_comb["VEHICLE_AGE"] = df_comb["C_YEAR"] - df_comb["V_YEAR"]

combined_features = rq1_features + rq2_features

df_comb = df_comb.dropna(subset=combined_features + ["Y_INJURY"]).copy()
df_comb = df_comb.sample(200000, random_state=42)

y_comb = df_comb["Y_INJURY"].astype(int)
X_comb = pd.get_dummies(df_comb[combined_features], drop_first=True)

X_comb = X_comb.apply(pd.to_numeric, errors="coerce").astype("float32")
X_comb = sm.add_constant(X_comb, has_constant="add")
# Remove zero-variance and duplicate columns
X_comb = X_comb.loc[:, X_comb.nunique() > 1]
# Remove duplicate columns (if any)
X_comb = X_comb.loc[:, ~X_comb.T.duplicated()]

# split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_comb, y_comb, test_size=0.30, stratify=y_comb, random_state=42
)

print("Combined shapes:", X_train_c.shape, X_test_c.shape)

# Separation check (remove columns that perfectly predict outcome)
bad_cols = []
for col in X_train_c.columns:
    if col == "const":
        continue
    if X_train_c[col].nunique() == 2:  # binary dummy
        y1 = y_train_c[X_train_c[col] == 1]
        y0 = y_train_c[X_train_c[col] == 0]
        if (y1.nunique() <= 1 and len(y1) > 50) or (y0.nunique() <= 1 and len(y0) > 50):
            bad_cols.append(col)

if bad_cols:
    print("Dropping separation columns:", bad_cols[:30], "..." if len(bad_cols) > 30 else "")
    X_train_c = X_train_c.drop(columns=bad_cols)
    X_test_c = X_test_c.drop(columns=bad_cols)

# Fit model — try MLE first, then fall back to regularized
model_comb = sm.Logit(y_train_c, X_train_c)

try:
    res_comb = model_comb.fit(maxiter=300, disp=1)
    fit_type = "MLE"
except Exception as e:
    print("\nMLE failed (likely singular matrix / separation). Switching to regularized fit.")
    print("Error:", repr(e))
    res_comb = model_comb.fit_regularized(method="l1", alpha=0.01, maxiter=500)
    fit_type = "Regularized (L1)"

print(f"\nCombined fitted using: {fit_type}")

# Predict + ROC AUC
p_test_c = res_comb.predict(X_test_c)
print("Combined ROC AUC:", roc_auc_score(y_test_c, p_test_c))

# Odds Ratios + 95% CI (works for MLE; for regularized, CI/p-values may not be available)
print("\n================ Combined Model Odds Ratios =================")

if fit_type == "MLE":
    params = res_comb.params
    conf = res_comb.conf_int()
    conf.columns = ["CI_low", "CI_high"]

    or_table = pd.DataFrame({
        "OR": np.exp(params),
        "CI_low": np.exp(conf["CI_low"]),
        "CI_high": np.exp(conf["CI_high"]),
        "p_value": res_comb.pvalues
    }).sort_values("OR", ascending=False)

    print(or_table.head(25).round(4))
    print("\nBottom 25 (most protective):")
    print(or_table.tail(25).round(4))
else:
    # Regularized models do not provide standard CI/p-values in the same way
    or_table = pd.DataFrame({
        "OR": np.exp(res_comb.params)
    }).sort_values("OR", ascending=False)

    print(or_table.head(25).round(4))
    print("\nBottom 25 (most protective):")
    print(or_table.tail(25).round(4))

    
