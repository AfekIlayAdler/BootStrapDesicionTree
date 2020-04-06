from pathlib import Path

import numpy as np
import pandas as pd

import time

from Tree.tree import CartRegressionTreeKFold, CartRegressionTree
from Tree.tree_visualizer import TreeVisualizer


# def create_x_y(regression=True):
#     df = pd.DataFrame()
#     n_rows = 10 ** 3
#     n_numeric_cols = 0
#     n_categorical_cols = 50
#     n_categorical_values = 50
#     for col in range(n_numeric_cols):
#         df[col] = np.random.random(n_rows)
#     for col in range(n_categorical_cols):
#         df[col + n_numeric_cols] = np.random.randint(n_categorical_values, size=n_rows)
#         df[col + n_numeric_cols] = df[col + n_numeric_cols].astype('category')
#     y = np.random.random(n_rows) if regression else np.random.randint(2, size=n_rows)
#     return df, pd.Series(y)


def create_x_y(category_size = 50, a = 5):
    a = float(a)
    N_ROWS = 100000
    X = pd.DataFrame()
    CATEGORY_COLUMN_NAME = 'cat'
    X[CATEGORY_COLUMN_NAME] = np.random.randint(0, category_size, N_ROWS)
    X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
    X['x1'] = np.random.randn(N_ROWS)
    sigma = np.random.randn(N_ROWS)
    left_group = [i for i in range(category_size // 2)]
    right_group = [i for i in range(category_size) if i not in left_group]
    left_indicator = (X['x1'] > 0)*1
    right_indicator = (X['x1'] <= 0)*1
    y = a * (left_indicator * X[CATEGORY_COLUMN_NAME].isin(left_group)*1 + right_indicator * X[CATEGORY_COLUMN_NAME].isin(
        right_group)*1) + sigma
    return X, y


if __name__ == '__main__':
    EXP = 'simulation'  # 'boston'
    KFOLD = False
    MAX_DEPTH = 5
    tree = CartRegressionTreeKFold(max_depth=MAX_DEPTH) if KFOLD else CartRegressionTree(max_depth=MAX_DEPTH)
    if EXP == 'boston':
        input_path = Path.cwd().parent / "Datasets/boston_house_prices/boston_house_prices.csv"
        dtypes = {'CRIM': 'float64',
                  'ZN': 'float64',
                  'INDUS': 'float64',
                  'CHAS': 'category',
                  'NOX': 'float64',
                  'RM': 'float64',
                  'AGE': 'float64',
                  'DIS': 'float64',
                  'RAD': 'category',
                  'TAX': 'float64',
                  'PTRATIO': 'float64',
                  'B': 'float64',
                  'LSTAT': 'float64',
                  'y': 'float64'}
        df = pd.read_csv(input_path, dtype=dtypes)
        start = time.time()
        y = df['y']
        X = df.drop(columns=['y'])
        tree.fit(X, y)
        end = time.time()
        print(end - start)
        tree_vis = TreeVisualizer()
        tree_vis.plot(tree.root)

    else:
        np.random.seed(3)
        X, y = create_x_y()
        print(np.mean(y[X['x1']>0]))
        print(np.mean(y[X['x1']<=0]))
        print(y.mean())
        start = time.time()
        tree.fit(X, y)
        end = time.time()
        print(end - start)
        tree_vis = TreeVisualizer()
        tree_vis.plot(tree.root)
