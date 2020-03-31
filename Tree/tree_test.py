from pathlib import Path
import numpy as np
import pandas as pd

from Tree.tree import CartRegressionTree, CartClassificationTree, CartRegressionTreeKFold
from Tree.tree_feature_importance import node_based_feature_importance
from Tree.tree_visualizer import TreeVisualizer
from sklearn.model_selection import train_test_split

import time


def create_x_y(regression=True):
    df = pd.DataFrame()
    n_rows = 10 ** 4
    n_numeric_cols = 0
    n_categorical_cols = 10
    n_categorical_values = 50
    for col in range(n_numeric_cols):
        df[col] = np.random.random(n_rows)
    for col in range(n_categorical_cols):
        df[col + n_numeric_cols] = np.random.randint(n_categorical_values, size=n_rows)
        df[col + n_numeric_cols] = df[col + n_numeric_cols].astype('category')
    y = np.random.random(n_rows) if regression else np.random.randint(2, size=n_rows)
    return df, pd.Series(y)


if __name__ == '__main__':
    EXP = 'boston'
    KFOLD = True
    MAX_DEPTH = 3
    tree = CartRegressionTreeKFold(label_col_name='y', max_depth=MAX_DEPTH+1) if KFOLD else CartRegressionTree(label_col_name='y', max_depth=MAX_DEPTH+1)
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
        tree.build(df)
        end = time.time()
        print(end - start)
        tree_vis = TreeVisualizer()
        tree_vis.plot(tree.root)
    else:
        np.random.seed(3)
        # tree = CartRegressionTree(max_depth=3)
        tree = CartRegressionTreeKFold(max_depth=3)
        X, y = create_x_y()
        X['y'] = y
        start = time.time()
        tree.build(X)
        end = time.time()
        print(end - start)
        tree_vis = TreeVisualizer()
        tree_vis.plot(tree.root)

