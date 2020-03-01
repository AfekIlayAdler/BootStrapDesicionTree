from pathlib import Path

from Tree.tree import CartRegressionTree, CartRegressionTreeKFold, MIN_SAMPLES_LEAF, MAX_DEPTH, MIN_IMPURITY_DECREASE, \
    MIN_SAMPLES_SPLIT

from numpy import mean, array
import pandas as pd

from Tree.tree_feature_importance import weighted_variance_reduction_feature_importance
from Tree_based_algorithms.gradient_boosting_abstract import GradientBoostingMachine, N_ESTIMATORS, LEARNING_RATE, \
    GRADIENT_BOOSTING_LABEL


class GradientBoostingRegressor(GradientBoostingMachine):
    """currently supports least squares"""

    def __init__(self, base_tree,
                 label_col_name,
                 n_estimators,
                 learning_rate,
                 min_samples_leaf,
                 max_depth,
                 min_impurity_decrease,
                 min_samples_split):
        self.label_col_name = label_col_name
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = base_tree
        self.base_prediction = None
        self.trees = []

    def compute_gradient(self, x, y):
        data = x.copy()
        data[GRADIENT_BOOSTING_LABEL] = y
        tree = self.tree(
            label_col_name=GRADIENT_BOOSTING_LABEL,
            min_samples_leaf=self.label_col_name,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            min_samples_split=self.min_samples_split)
        tree.build(data)
        gradients = tree.predict(x)
        self.trees.append(tree)
        return gradients

    def fit(self, data: pd.DataFrame):
        y = data[self.label_col_name]
        x = data.drop(columns = [self.label_col_name])
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
    def __init__(self, label_col_name,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(
            base_tree=CartRegressionTree,
            label_col_name=label_col_name,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split)


class CartGradientBoostingRegressorKfold(GradientBoostingRegressor):
    def __init__(self, label_col_name,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(
            base_tree=CartRegressionTreeKFold,
            label_col_name=label_col_name,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split)


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
    test_tree = CartRegressionTree("y", max_depth=4)
    # tree = CartRegressionTreeKFold("y", max_depth=4)
    test_tree.build(df)
    fi = weighted_variance_reduction_feature_importance(test_tree)
    print(pd.Series(fi) / pd.Series(fi).sum())
