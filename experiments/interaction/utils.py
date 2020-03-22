import pickle
from os import makedirs
from xgboost.core import Booster

import numpy as np
import pandas as pd

from Tree.node import Leaf
from experiments.interaction.config import N_ROWS, SIGMA, A1, CATEGORY_COLUMN_NAME, \
    Y_COL_NAME, MAX_DEPTH, LEARNING_RATE, N_ESTIMATORS, A2, N_EXPERIMENTS, CATEGORIES, A1_VALUES
from experiments.interaction.one_hot_encoder import OneHotEncoder
from gradient_boosting_trees.gradient_boosting_abstract import GradientBoostingMachine
from gradient_boosting_trees.gradient_boosting_classifier import GradientBoostingClassifier
from gradient_boosting_trees.gradient_boosting_regressor import GradientBoostingRegressor


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


def create_mean_imputing_x_x_val(x, y, x_val):
    temp_x = x.copy()
    temp_x.loc[:,Y_COL_NAME] = y
    category_to_mean = temp_x.groupby(CATEGORY_COLUMN_NAME)[Y_COL_NAME].mean().to_dict()
    temp_x[CATEGORY_COLUMN_NAME] = temp_x[CATEGORY_COLUMN_NAME].map(category_to_mean)
    temp_x = temp_x.drop(columns=[Y_COL_NAME])
    temp_x[CATEGORY_COLUMN_NAME] = temp_x[CATEGORY_COLUMN_NAME].astype('float')
    x_val[CATEGORY_COLUMN_NAME] = x_val[CATEGORY_COLUMN_NAME].map(category_to_mean)
    x_val[CATEGORY_COLUMN_NAME] = x_val[CATEGORY_COLUMN_NAME].astype('float')
    x_val[CATEGORY_COLUMN_NAME] = x_val[CATEGORY_COLUMN_NAME].fillna(x_val[CATEGORY_COLUMN_NAME].mean())
    return temp_x, x_val


def create_one_hot_x_x_val(x, x_val):
    one_hot = OneHotEncoder()
    one_hot.fit(x[CATEGORY_COLUMN_NAME].to_frame())
    x_one_hot = one_hot.transform(x[CATEGORY_COLUMN_NAME].to_frame())
    x_one_hot['x1'] = x['x1']
    x_one_hot_val = one_hot.transform(x_val[CATEGORY_COLUMN_NAME].to_frame())
    x_one_hot_val['x1'] = x_val['x1']
    return x_one_hot, x_one_hot_val


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
                      learning_rate=LEARNING_RATE, min_samples_leaf=5)
        model.fit(X)
    return model


def save_model(path, model):
    with open(path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def all_experiments():
    return [(exp_number, category_size, a1) for exp_number in range(N_EXPERIMENTS)
            for category_size in CATEGORIES
            for a1 in A1_VALUES]


def compute_ntrees_nleaves(gbm):
    total_number_of_trees = 0
    total_number_of_leaves = 0
    if isinstance(gbm, GradientBoostingMachine):
        for tree in gbm.trees:
            if not isinstance(tree.root, Leaf):
                total_number_of_trees += 1
                total_number_of_leaves += tree.n_leaves
    elif isinstance(gbm, GradientBoostingClassifier) or isinstance(gbm, GradientBoostingRegressor):
        total_number_of_trees = gbm.n_estimators_
        for tree in gbm.estimators_:
            total_number_of_leaves += tree[0].get_n_leaves()
    elif isinstance(gbm, Booster):
        df = gbm.trees_to_dataframe()
        total_number_of_leaves = df[df['Feature'] == 'Leaf'].shape[0]
        total_number_of_trees = df[df['Feature'] == 'Leaf']['Tree'].max()

    print(F'number of trees is {total_number_of_trees}')
    print(F'number of leaves is {total_number_of_leaves}')


n_experiments = len(all_experiments())
