import numpy as np
import pandas as pd

from tree import CartRegressionTree, CartClassificationTree, CartRegressionTreeKFold
import time

from tree_visualizer import TreeVisualizer


def create_x_y(regression=True):
    df = pd.DataFrame()
    n_rows = 10**3
    n_numeric_cols = 0
    n_categorical_cols = 50
    n_categorical_values = 50
    for col in range(n_numeric_cols):
        df[col] = np.random.random(n_rows)
    for col in range(n_categorical_cols):
        df[col + n_numeric_cols] = np.random.randint(n_categorical_values, size=n_rows)
        df[col + n_numeric_cols] = df[col + n_numeric_cols].astype('category')
    y = np.random.random(n_rows) if regression else np.random.randint(2, size=n_rows)
    return df, pd.Series(y)


if __name__ == '__main__':
    EXP = 'simulation' # 'boston'
    KFOLD = True
    MAX_DEPTH = 3
    tree = CartRegressionTreeKFold(max_depth=MAX_DEPTH) if KFOLD else CartRegressionTree(max_depth=MAX_DEPTH)
    if EXP == 'boston':
        input_path = "boston_house_prices.csv"
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
        tree.fit(X,y)
        end = time.time()
        print(end - start)
        tree_vis = TreeVisualizer()
        tree_vis.plot(tree.root)

    else:
        np.random.seed(3)
        X, y = create_x_y()
        start = time.time()
        tree.fit(X, y)
        end = time.time()
        print(end - start)
        tree_vis = TreeVisualizer()
        tree_vis.plot(tree.root)

