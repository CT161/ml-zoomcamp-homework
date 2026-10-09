import numpy as np
import pandas as pd

# 1. Load dataset fresh
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
df = pd.read_csv(url)

# 2. Reset index to ensure clean integer row selection
df = df.reset_index(drop=True)

# 3. Features
base = [
    'engine_displacement',
    'horsepower',
    'vehicle_weight',
    'model_year'
]

def prepare_X(df_input, base_features):
    df_num = df_input[base_features].fillna(0)
    return df_num.values

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTXinv = np.linalg.inv(XTX)
    w_full = XTXinv.dot(X.T).dot(y)
    return w_full[0], w_full[1:]

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    return np.sqrt(se.mean())

# Split lengths
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

rmse_scores = []

for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    
    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val   = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    
    y_train = np.log1p(df_train['fuel_efficiency_mpg'].values)
    y_val   = np.log1p(df_val['fuel_efficiency_mpg'].values)
    
    X_train = prepare_X(df_train, base)
    X_val   = prepare_X(df_val, base)
    
    w0, w = train_linear_regression(X_train, y_train)
    y_pred = w0 + X_val.dot(w)

    score = rmse(y_val, y_pred)
    rmse_scores.append(score)

raw_std = np.std(rmse_scores)
print()
print(f"Raw RMSE Scores: {rmse_scores}")
print()
print(f"Raw Standard Deviation: {raw_std}")
print()
print(f"Rounded STD: {round(raw_std, 3)}")