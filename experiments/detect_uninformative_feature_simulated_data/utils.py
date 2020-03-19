import pickle
from os import makedirs

import numpy as np
import pandas as pd

from experiments.detect_uninformative_feature_simulated_data.config import N_ROWS, SIGMA, A1, CATEGORY_COLUMN_NAME, \
    Y_COL_NAME, MAX_DEPTH, LEARNING_RATE, N_ESTIMATORS, A2, N_EXPERIMENTS, CATEGORIES


def create_x(category_size):
    X = pd.DataFrame()
    X['x1'] = np.random.randn(N_ROWS)
    X['x2'] = np.random.randn(N_ROWS)
    y = A1 * X['x1'] + A2 * X['x2'] + SIGMA * np.random.randn(N_ROWS)
    X[CATEGORY_COLUMN_NAME] = np.random.randint(0, category_size, N_ROWS)
    X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
    X[Y_COL_NAME] = y
    return X


def make_dirs(dirs):
    for dir in dirs:
        if not dir.exists():
            makedirs(dir)


def get_fitted_model(path, model, X, y_col_name):
    if path.exists():
        with open(path, 'wb') as output:
            model = pickle.load(model, output, pickle.HIGHEST_PROTOCOL)

    else:
        model = model(y_col_name, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                      learning_rate=LEARNING_RATE)
        model.fit(X)
    return model


def save_model(path, model):
    with open(path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def all_experiments():
    return [(exp_number, category_size) for exp_number in range(N_EXPERIMENTS) for category_size in CATEGORIES]


n_experiments = len(all_experiments())
