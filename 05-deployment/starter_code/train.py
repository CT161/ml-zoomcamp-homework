#!/usr/bin/env python

# SYSTEM IMPORTS
import pickle

# PACKAGES
import pandas as pd
import numpy as np
import sklearn

# NAMED FUNCTIONS FROM PACKAGES
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')

# FUNCTION TO LOAD DATA
def load_data():
    data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    df = pd.read_csv(data_url)

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == 'yes').astype(int)
    return df



# FUNCTIN TO TRAIN THE DATA
def train_model(df):
    # FEATURE SELECTION LISTS
    numerical = ['tenure', 'monthlycharges', 'totalcharges']

    categorical = [
        'gender',
        'seniorcitizen',
        'partner',
        'dependents',
        'phoneservice',
        'multiplelines',
        'internetservice',
        'onlinesecurity',
        'onlinebackup',
        'deviceprotection',
        'techsupport',
        'streamingtv',
        'streamingmovies',
        'contract',
        'paperlessbilling',
        'paymentmethod',
    ]

    # CREATE TARGET AND FEATURE DICTIONARY
    y_train = df.churn
    train_dict = df[categorical + numerical].to_dict(orient='records')

    # CREATE PIPELINE
    pipeline = make_pipeline(
        DictVectorizer(),
        # LogisticRegression(solver='liblinear')
        LogisticRegression(solver='lbfgs', max_iter=5000)
    )

    # FIT THE DATA
    pipeline.fit(train_dict, y_train)

    # RETURN THE PIPELINE
    return pipeline


def save_model(pipeline, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)


# MAIN EXECUTION
df = load_data()
pipeline = train_model(df)
save_model(pipeline, 'model.bin')

print('Model saved to model.bin')