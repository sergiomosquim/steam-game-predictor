import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor

from transformers import FreqEnc, MLBTransformer, ReleaseDateTransformer

np.random.seed(123)

def load_data():
    df = joblib.load('data/data_preprocessed.pkl')
    return df

def build_preprocessor(df):
    numeric = df.select_dtypes(exclude='object').columns.drop('price').to_list()

    preprocess = ColumnTransformer(transformers=[
    ('log_price', FunctionTransformer(func=np.log1p, feature_names_out='one-to-one'), ['price']),
    ('date',ReleaseDateTransformer(), 'release_date'),
    ('genres', MLBTransformer(), 'genres'),
    ('categories', MLBTransformer(), 'categories'),
    ('developers', FreqEnc(), 'developers'),
    ('publishers', FreqEnc(), 'publishers'),
    ('num','passthrough', numeric)
    ])

    return preprocess

def build_pipeline(preprocess, best_params):
    model = XGBRegressor(random_state=234)

    pipeline = Pipeline([
        ('preprocess', preprocess),
        ('model', model)
    ])

    pipeline.set_params(**best_params) #passes the arguments inside best_params, instead of a dictionary
    return pipeline

def main():
    # load the dataset
    df = load_data()

    # drop target variable
    y = df.target
    df = df.drop(columns='target')

    # split the data
    df_full_train, df_test, y_full_train, y_test = train_test_split(df, y, random_state=123, test_size=0.2)

    # Make a dictionary of quantiles of y_full_train for making categories during prediction
    quantiles = y_full_train.quantile(q=[0.2, 0.4, 0.6, 0.8])
    quantile_dict = dict({
        '20%': float(quantiles[0.2]),
        '40%': float(quantiles[0.4]),
        '60%': float(quantiles[0.6]),
        '80%': float(quantiles[0.8])
    })
    with open('log_score_quantiles.json', 'w') as f:
        json.dump(quantile_dict, f)

    # build preprocessing
    preprocess = build_preprocessor(df_full_train)

    # load model parameters
    with open('best_xgboost_params.json', 'r') as f:
        best_params = json.load(f)

    # build pipeline
    pipeline = build_pipeline(preprocess, best_params)

    # fit on the entire training dataset
    pipeline.fit(df_full_train, y_full_train)

    # save the final model
    with open('final_xgb_model.pkl', 'wb') as f:
        joblib.dump(pipeline, f)
    print('Model saved as final_xgb_model.pkl')

    # Calculate RMSE on the test set
    y_test_pred = pipeline.predict(df_test)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    print(f'RMSE on the test set is {test_rmse}')

if __name__ == '__main__':
    main()

