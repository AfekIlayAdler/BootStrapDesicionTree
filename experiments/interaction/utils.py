import pickle
from os import makedirs

import numpy as np
import pandas as pd

from experiments.interaction.config import N_ROWS, SIGMA, A1, CATEGORY_COLUMN_NAME, \
    Y_COL_NAME, MAX_DEPTH, LEARNING_RATE, N_ESTIMATORS, A2, N_EXPERIMENTS, CATEGORIES


def create_x_y(category_size, a1=A1, a2=A2):
    X = pd.DataFrame()
    X[CATEGORY_COLUMN_NAME] = np.random.randint(0, category_size, N_ROWS)
    X['x1'] = np.random.randn(N_ROWS)
    X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
    sigma = SIGMA * np.random.randn(N_ROWS)
    left_group = [i for i in range(category_size // 2)]
    y = a1 * X['x1'] + a2 * X[CATEGORY_COLUMN_NAME].isin(left_group) + sigma
    return X, y


def create_x(category_size, a1=A1, a2=A2):
    x, y = create_x_y(category_size, a1, a2)
    x[Y_COL_NAME] = y
    return x


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
        model.fit(X)
    return model


def save_model(path, model):
    with open(path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def all_experiments():
    return [(exp_number, category_size) for exp_number in range(N_EXPERIMENTS) for category_size in CATEGORIES]


n_experiments = len(all_experiments())
