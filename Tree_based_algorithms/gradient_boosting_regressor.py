from pathlib import Path

from Tree.tree import CartRegressionTree, CartRegressionTreeKFold

from numpy import mean, array
import pandas as pd

from Tree.tree_feature_importance import weighted_variance_reduction_feature_importance
from Tree_based_algorithms.gradient_boosting_abstract import GradientBoostingMachine


class GradientBoostingRegressor(GradientBoostingMachine):
    """currently supports least squares"""

    def __init__(self, tree, label_col_name, n_estimators=100, learning_rate=0.01):
        self.label_col_name = label_col_name
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tree = tree
        self.base_prediction = None
        self.trees = []

    def compute_gradient(self, x, y):
        tree = self.tree()
        tree.build(x, y)
        gradients = tree.predict(x)
        self.trees.append(tree)
        return gradients

    def fit(self, data: pd.DataFrame):
        y = data[self.label_col_name]
        self.base_prediction = mean(y)
        f = mean(y)
        for m in range(self.n_estimators):
            pseudo_response = y - f
            gradients = self.compute_gradient(x, pseudo_response)
            f += self.learning_rate * gradients

    def predict(self, x):
        predictions = []
        for row in x:
            prediction = self.mean + self.learning_rate * array([tree.predict(row) for tree in self.trees])
            predictions.append(prediction)
        return array(predictions)


class CartGradientBoostingRegressor(GradientBoostingRegressor):
    def __init__(self, label_col_name, n_estimators=100, learning_rate=0.01):
        super().__init__(CartRegressionTree,
                         label_col_name,
                         n_estimators=n_estimators, learning_rate=learning_rate)


class CartGradientBoostingRegressorKfold(GradientBoostingRegressor):
    def __init__(self, label_col_name, n_estimators=100, learning_rate=0.01):
        super().__init__(CartRegressionTreeKFold,
                         label_col_name,
                         n_estimators=n_estimators, learning_rate=learning_rate)


if __name__ == '__main__':
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
    tree = CartRegressionTree("y", max_depth=4)
    # tree = CartRegressionTreeKFold("y", max_depth=4)
    tree.build(df)
    print(tree.predict(X_test.to_dict('records')))
    fi = weighted_variance_reduction_feature_importance(tree)
    print(pd.Series(fi) / pd.Series(fi).sum())



