# import necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import StandardScaler

# load the data
data_path = "NCDB_1999_to_2014.csv"
def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    return df

df = load_data(data_path)

# convert caterogical coulmns to numeric columns
numeric_cols = ['C_MNTH','C_WDAY','C_HOUR','C_VEHS','V_YEAR','P_AGE','P_ISEV']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# subset the data for RQ1 and RQ2 based on the severity levels

rq1_df = df[df["C_SEV"].isin([1,2])].copy()
rq2_df = df[df["P_ISEV"].isin([1,2,3])].copy()

# Sampleling the data for RQ1 and RQ2
sample_size = 250000
rq1_sample = rq1_df.sample(sample_size, random_state=42)
rq2_sample = rq2_df.sample(sample_size, random_state=42)

# Selecting predictors and targets

# RQ1: Logestic Regression

rq1_vars = ["C_WTHR","C_RSUR","C_RALN","C_TRAF","C_MNTH","C_WDAY","C_HOUR","C_VEHS"]

rq1 = rq1_sample.dropna(subset=rq1_vars + ["C_SEV"]).copy()

y_rq1 = rq1["C_SEV"].map({1: 1, 2: 0}).astype(int)
X_rq1 = pd.get_dummies(rq1[rq1_vars], drop_first=True)

# Force numeric matrix
X_rq1 = X_rq1.apply(pd.to_numeric, errors="coerce").astype("float32")
mask = X_rq1.notna().all(axis=1)
X_rq1 = X_rq1.loc[mask]
y_rq1 = y_rq1.loc[X_rq1.index]

# Add intercept (Logit needs it)
X_rq1.insert(0, "const", 1.0)

print("RQ1 final:", X_rq1.shape, y_rq1.shape)

model_rq1 = sm.Logit(y_rq1, X_rq1).fit(disp=1, maxiter=200)
print(model_rq1.summary())

# RQ2: Ordinal Logistic Regression

rq2_vars = ["P_SAFE", "P_USER", "V_TYPE", "P_AGE", "P_SEX", "C_WTHR", "C_HOUR", "C_VEHS"]

rq2 = rq2_sample.dropna(subset=rq2_vars + ["P_ISEV"]).copy()

y_rq2 = rq2["P_ISEV"].astype(int)
X_rq2 = pd.get_dummies(rq2[rq2_vars], drop_first=True)

X_rq2 = X_rq2.apply(pd.to_numeric, errors="coerce").astype("float32")
mask = X_rq2.notna().all(axis=1)
X_rq2 = X_rq2.loc[mask]
y_rq2 = y_rq2.loc[X_rq2.index]

# Drop rare dummy columns (keep continuous columns always)
min_count = 1000
freq = X_rq2.sum(axis=0)
keep = (freq >= min_count) | X_rq2.columns.isin(["P_AGE", "C_HOUR", "C_VEHS"])
X_rq2 = X_rq2.loc[:, keep]

# Drop zero-variance columns
X_rq2 = X_rq2.loc[:, X_rq2.nunique() > 1]

# IMPORTANT: re-align y after column filtering
y_rq2 = y_rq2.loc[X_rq2.index]

# Scale continuous variables (only if present)
scale_cols = [c for c in ["P_AGE", "C_HOUR", "C_VEHS"] if c in X_rq2.columns]
if scale_cols:
    scaler = StandardScaler()
    X_rq2[scale_cols] = scaler.fit_transform(X_rq2[scale_cols])

print("RQ2 final:", X_rq2.shape, y_rq2.shape)

model_ord = OrderedModel(y_rq2, X_rq2, distr="logit")
res_ord = model_ord.fit(method="lbfgs", maxiter=800, disp=True)
print(res_ord.summary())