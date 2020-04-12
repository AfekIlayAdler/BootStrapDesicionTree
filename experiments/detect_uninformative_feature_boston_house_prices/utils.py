import pickle
from os import makedirs

import numpy as np
import pandas as pd

from experiments.detect_uninformative_feature_boston_house_prices.config import CATEGORY_COLUMN_NAME, \
    CATEGORICAL_COLUMNS, MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE, N_EXPERIMENTS, CATEGORIES
from experiments.detect_uninformative_feature_boston_house_prices.one_hot_encoder import OneHotEncoder


def create_x_y(category_size, path):
    df = pd.read_csv(path)
    y = df['y']
    X = df.drop(columns=['y'])
    X[CATEGORY_COLUMN_NAME] = np.random.randint(0, category_size, df.shape[0])
    X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
    return X, y


def create_one_hot_x_x_val(x, x_val):
    one_hot = OneHotEncoder()
    one_hot.fit(x[CATEGORICAL_COLUMNS])
    x_one_hot = one_hot.transform(x[CATEGORICAL_COLUMNS])
    x_one_hot_val = one_hot.transform(x_val[CATEGORICAL_COLUMNS])
    for col in x.columns:
        if col not in CATEGORICAL_COLUMNS:
            x_one_hot.loc[:, col] = x[col]
            x_one_hot_val.loc[:, col] = x_val[col]
    return x_one_hot, x_one_hot_val


def create_mean_imputing_x_x_val(x, y, x_val):
    temp_x = x.copy()
    col_name = 'y'
    temp_x.loc[:, col_name] = y
    for col in CATEGORICAL_COLUMNS:
        category_to_mean = temp_x.groupby(col)[col_name].mean().to_dict()
        temp_x[col] = temp_x[col].map(category_to_mean)
        x_val[col] = x_val[col].map(category_to_mean)
        temp_x[col] = temp_x[col].astype('float')
        x_val[col] = x_val[col].astype('float')
        x_val[col] = x_val[col].fillna(x_val[col].mean())
    temp_x = temp_x.drop(columns=[col_name])
    return temp_x, x_val


def make_dirs(dirs):
    for dir in dirs:
        if not dir.exists():
            makedirs(dir)


def get_fitted_model(path, model, X, y_col_name):
    if path.exists():
        with open(path, 'rb') as input_file:
            model = pickle.load(input_file)

    else:
        model = model(y_col_name, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                      learning_rate=LEARNING_RATE)
        model.fit(X, )
    return model


def save_model(path, model):
    with open(path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def all_experiments():
    return [(exp_number, category_size) for exp_number in range(N_EXPERIMENTS) for category_size in CATEGORIES]


n_experiments = len(all_experiments())
