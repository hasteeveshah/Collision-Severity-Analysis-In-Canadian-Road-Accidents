import pandas as pd
import numpy as np

# Load the dataset with specified dtypes to optimize memory usage

# df = pd.read_csv("NCDB_1999_to_2014.csv", low_memory=False)
# # Save as Parquet for faster future loading
# df.to_parquet("NCDB.parquet", compression="snappy")


# SETTINGS

DATA_PATH = "NCDB.parquet"
RANDOM_STATE = 42

# Samples for speed
N_SAMPLE_RQ1 = 75000
N_SAMPLE_MAIN = 75000

# Rare category threshold
MIN_COUNT_RARE = 3000



# LOAD DATA

df = pd.read_parquet(DATA_PATH)
print("shape:", df.shape)



# BASIC TYPE CLEANING

numeric_cols = [
    "P_AGE", "C_MNTH", "C_WDAY", "C_HOUR",
    "C_VEHS", "V_YEAR",
    "P_ISEV", "C_SEV"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# floats64 to float32
float_cols = df.select_dtypes(include=["float64"]).columns
df[float_cols] = df[float_cols].astype("float32")

# int64 to int32 
int_cols = df.select_dtypes(include=["int64"]).columns
df[int_cols] = df[int_cols].astype("int32")

print("After numeric conversion:", df.shape)



# FEATURE ENGINEERING

if "C_YEAR" in df.columns and "V_YEAR" in df.columns:
    df["VEHICLE_AGE"] = df["C_YEAR"] - df["V_YEAR"]
    df.loc[(df["VEHICLE_AGE"] < 0) | (df["VEHICLE_AGE"] > 60), "VEHICLE_AGE"] = np.nan

# print(df.dtypes)