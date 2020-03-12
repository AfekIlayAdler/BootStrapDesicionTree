from os import mkdir
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from gradient_boosting_trees.gradient_boosting_regressor import CartGradientBoostingRegressor, \
    CartGradientBoostingRegressorKfold

A1 = 3
A2 = 2
SIGMNA = 10
N_ROWS = 10 ** 3
CATEGORICAL_DISTRIBUTION = 'uniform'
CATEGORY_COLUMN_NAME = 'random_category'
Y_COL_NAME = 'y'
CATEGORIES = np.arange(10, 210, 10)
N_EXPERIMENTS = 5

MAX_DEPTH = 4
N_ESTIMATORS = 100
LEARNING_RATE = 0.1

# X = pd.DataFrame()
# X['x1'] = np.random.randn(N_ROWS)
# X['x2'] = np.random.randn(N_ROWS)
# X[CATEGORY_COLUMN_NAME] = np.random.randint(0, CATEGORY_SIZE, N_ROWS)
# X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
# sigma = SIGMNA * np.random.randn(N_ROWS)
# y = A1 * X['x1'] + A2 * X['x2'] + sigma
# X[Y_COL_NAME] = y

EXP_NAME = F"detect_uninformative_feature_{MAX_DEPTH}_nestimators_{N_ESTIMATORS}_learning_rate_{LEARNING_RATE}"

sorted_index = ['x1', 'x2', 'x3']

RESULTS_DIR = Path(F"results/")
if not RESULTS_DIR.exists():
    mkdir(RESULTS_DIR)


def create_x(category_size):
    X = pd.DataFrame()
    X['x1'] = np.random.randn(N_ROWS)
    X['x2'] = np.random.randn(N_ROWS)
    sigma = SIGMNA * np.random.randn(N_ROWS)
    y = A1 * X['x1'] + A2 * X['x2'] + sigma
    X[CATEGORY_COLUMN_NAME] = np.random.randint(0, category_size, N_ROWS)
    X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
    X[Y_COL_NAME] = y
    return X


if __name__ == '__main__':
    for exp in range(N_EXPERIMENTS):
        for category_size in tqdm(CATEGORIES, total=len(CATEGORIES)):
            np.random.seed(exp)
            kfold_exp_name = F"{EXP_NAME}__kfold_exp_{exp}_category_size_{category_size}"
            reg_exp_name = F"{EXP_NAME}__reg_exp_{exp}_category_size_{category_size}"
            kfold_path = RESULTS_DIR / F"{kfold_exp_name}.csv"
            reg_path = RESULTS_DIR / F"{reg_exp_name}.csv"
            if kfold_path.exists() or reg_path.exists():
                continue
            X = create_x(category_size)
            # reg
            regular_gbm = CartGradientBoostingRegressor(Y_COL_NAME, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                                                        learning_rate=LEARNING_RATE)
            regular_gbm.fit(X)
            fi = pd.Series(regular_gbm.compute_feature_importance()).sort_index()
            fi /= fi.sum()
            fi.to_csv(F"{reg_path.parent}/{reg_path.name}", header=True)
            # kfold
            kfold_gbm = CartGradientBoostingRegressorKfold(Y_COL_NAME, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                                                           learning_rate=LEARNING_RATE)
            kfold_gbm.fit(X)
            fi = pd.Series(kfold_gbm.compute_feature_importance()).sort_index()
            fi /= fi.sum()
            fi.to_csv(kfold_path, header=True)
            with open(RESULTS_DIR / F"{reg_exp_name}.pkl", 'wb') as output:
                pickle.dump(regular_gbm, output, pickle.HIGHEST_PROTOCOL)
            with open(RESULTS_DIR / F"{kfold_exp_name}.pkl", 'wb') as output:
                pickle.dump(kfold_gbm, output, pickle.HIGHEST_PROTOCOL)
