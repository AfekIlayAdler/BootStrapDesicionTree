import pickle
from os import makedirs

import numpy as np
import pandas as pd

from experiments.detect_uninformative_feature_simulated_data.config import N_ROWS, SIGMA, A1, CATEGORY_COLUMN_NAME, \
     MAX_DEPTH, LEARNING_RATE, N_ESTIMATORS, A2, N_EXPERIMENTS, CATEGORIES
from experiments.detect_uninformative_feature_simulated_data.one_hot_encoder import OneHotEncoder


def create_x_y(category_size):
    X = pd.DataFrame()
    X['x1'] = np.random.randn(N_ROWS)
    X['x2'] = np.random.randn(N_ROWS)
    sigma = SIGMA * np.random.randn(N_ROWS)
    y = A1 * X['x1'] + A2 * X['x2'] + sigma
    X[CATEGORY_COLUMN_NAME] = np.random.randint(0, category_size, N_ROWS)
    X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
    return X, y


def create_one_hot_x_x_val(x, x_val):
    one_hot = OneHotEncoder()
    one_hot.fit(x[CATEGORY_COLUMN_NAME].to_frame())
    x_one_hot = one_hot.transform(x[CATEGORY_COLUMN_NAME].to_frame())
    x_one_hot['x1'] = x['x1']
    x_one_hot['x2'] = x['x2']
    x_one_hot_val = one_hot.transform(x_val[CATEGORY_COLUMN_NAME].to_frame())
    x_one_hot_val['x1'] = x_val['x1']
    x_one_hot_val['x2'] = x_val['x2']
    return x_one_hot, x_one_hot_val


def create_mean_imputing_x_x_val(x, y, x_val):
    temp_x = x.copy()
    col_name = 'y'
    temp_x.loc[:,col_name] = y
    category_to_mean = temp_x.groupby(CATEGORY_COLUMN_NAME)[col_name].mean().to_dict()
    temp_x[CATEGORY_COLUMN_NAME] = temp_x[CATEGORY_COLUMN_NAME].map(category_to_mean)
    temp_x = temp_x.drop(columns=[col_name])
    temp_x[CATEGORY_COLUMN_NAME] = temp_x[CATEGORY_COLUMN_NAME].astype('float')
    x_val[CATEGORY_COLUMN_NAME] = x_val[CATEGORY_COLUMN_NAME].map(category_to_mean)
    x_val[CATEGORY_COLUMN_NAME] = x_val[CATEGORY_COLUMN_NAME].astype('float')
    x_val[CATEGORY_COLUMN_NAME] = x_val[CATEGORY_COLUMN_NAME].fillna(x_val[CATEGORY_COLUMN_NAME].mean())
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
