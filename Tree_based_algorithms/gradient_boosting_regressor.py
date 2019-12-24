from Tree.tree import CartRegressionTree

from numpy import mean, array

from Tree_based_algorithms.gradient_boosting_abstract import GradientBoostingClassifier


class GradientBoostingRegressor(GradientBoostingClassifier):
    """currently supports least squares"""

    def __init__(self, n_estimators, learning_rate):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tree = CartRegressionTree
        self.base_prediction = None
        self.trees = []

    def compute_gradient(self, x, y):
        tree = self.tree()
        tree.build(x, y)
        gradients = tree.predict(x)
        self.trees.append(tree)
        return gradients

    def fit(self, x, y):
        self.base_prediction = mean(y)
        f = mean(y)
        for m in range(self.n_estimators):
            pseudo_response = y - f
            gradients = self.compute_gradient(x, pseudo_response)
            f = f + self.learning_rate * gradients

    def predict(self, x):
        predictions = []
        for row in x:
            prediction = self.mean + self.learning_rate * array([tree.predict(row) for tree in self.trees])
            predictions.append(prediction)
        return array(predictions)


if __name__ == '__main__':
    reg = GradientBoostingRegressor(100, 0.01)
