from numpy import mean
from pandas import DataFrame, Series

from Tree.node import Leaf
from Tree.tree import CartRegressionTree, CartRegressionTreeKFold, MIN_SAMPLES_LEAF, MAX_DEPTH, MIN_IMPURITY_DECREASE, \
    MIN_SAMPLES_SPLIT
from Tree.tree_feature_importance import node_based_feature_importance
from gradient_boosting_trees.gradient_boosting_abstract import GradientBoostingMachine, N_ESTIMATORS, LEARNING_RATE


class GradientBoostingRegressor(GradientBoostingMachine):
    """currently supports least squares"""

    def __init__(self, base_tree,
                 n_estimators,
                 learning_rate,
                 min_samples_leaf,
                 max_depth,
                 min_impurity_decrease,
                 min_samples_split):
        self.n_estimators = n_estimators
        self.n_trees = 0
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = base_tree
        self.base_prediction = None
        self.features = None
        self.trees = []
        self.mean = None

    def compute_gradient(self, x, y):
        temp_x = x.copy()
        temp_y = y.copy()
        tree = self.tree(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            min_samples_split=self.min_samples_split)
        tree.fit(temp_x, temp_y)
        gradients = tree.predict(x.to_dict('records'))
        self.trees.append(tree)
        return gradients

    def fit(self, X, y):
        self.features = X.columns
        self.base_prediction = mean(y)
        f = mean(y)
        self.mean = f
        for m in range(self.n_estimators):
            if m > 0 and isinstance(self.trees[-1].root, Leaf):  # if the previous tree was a bark then we stop
                return
            pseudo_response = y - f
            gradients = self.compute_gradient(X, pseudo_response)
            f += self.learning_rate * gradients
            self.n_trees += 1

    def predict(self, data: DataFrame):
        X = data.copy()
        if not isinstance(X, DataFrame):
            X = DataFrame(X, columns = self.features)
        X['prediction'] = self.mean
        for tree in self.trees:
            X['prediction'] += self.learning_rate * tree.predict(X.to_dict('records'))
        return X['prediction']

    def compute_feature_importance(self, method='gain'):
        gbm_feature_importances = {feature: 0 for feature in self.features}
        # TODO : deal with the case that a tree is a bark
        for tree in self.trees:
            tree_feature_importance = node_based_feature_importance(tree, method=method)
            for feature, feature_importance in tree_feature_importance.items():
                gbm_feature_importances[feature] += feature_importance
        return gbm_feature_importances


class CartGradientBoostingRegressor(GradientBoostingRegressor):
    def __init__(self,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(
            base_tree=CartRegressionTree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split)


class CartGradientBoostingRegressorKfold(GradientBoostingRegressor):
    def __init__(self,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(
            base_tree=CartRegressionTreeKFold,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split)


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import time
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

    def create_x_y(category_size=52):
        A1 = 3
        A2 = 2
        SIGMA = 10
        N_ROWS = 10 ** 3
        CATEGORY_COLUMN_NAME = 'random_category'
        VAL_RATIO = 0.15
        X = pd.DataFrame()
        X['x1'] = np.random.randn(N_ROWS)
        X['x2'] = np.random.randn(N_ROWS)
        sigma = SIGMA * np.random.randn(N_ROWS)
        y = A1 * X['x1'] + A2 * X['x2'] + sigma
        X[CATEGORY_COLUMN_NAME] = np.random.randint(0, category_size, N_ROWS)
        X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
        return X, y


    np.seterr(all='raise')
    EXP = 'simulation'  # 'simulation'
    KFOLD = True
    MAX_DEPTH = 3
    reg = CartGradientBoostingRegressorKfold(max_depth=3) if KFOLD else CartGradientBoostingRegressor(max_depth=3)
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
        reg.fit(X, y)
        end = time.time()
        print(end - start)
        tree_vis = TreeVisualizer()
        tree_vis.plot(reg.trees[-1].root)
        print(reg.compute_feature_importance())

    else:
        np.random.seed(3)
        X, y = create_x_y()
        start = time.time()
        reg.fit(X, y)
        end = time.time()
        print(end - start)
        tree_vis = TreeVisualizer()
        tree_vis.plot(reg.trees[0].root)
        print(reg.compute_feature_importance())
