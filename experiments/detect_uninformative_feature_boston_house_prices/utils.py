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





def all_experiments():
    return [(exp_number, category_size) for exp_number in range(N_EXPERIMENTS) for category_size in CATEGORIES]


n_experiments = len(all_experiments())
