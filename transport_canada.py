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

