from Tree.tree import CartRegressionTree

from numpy import mean, array, log, exp
from pandas import DataFrame

from Tree_based_algorithms.gradient_boosting_abstract import GradientBoostingClassifier


class GradientBoostingClassifier(GradientBoostingClassifier):
    """currently supports only binomial log likelihood as in the original paper of friedman"""

    def __init__(self, n_estimators, learning_rate):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tree = CartRegressionTree  # TODO change to classification tree
        self.base_prediction = None
        self.trees = []
        self.step_sizes = []

    @staticmethod
    def line_search(x, y):
        """x: tree predictions, y: pseudo response"""
        df = DataFrame([x, y]).T
        df.columns = ['predictions', 'pseudo_response']
        grouped_values = df.groupby('predictions')['pseudo_response'].apply(array)
        leaf_values_to_step_size_dict = grouped_values.apply(lambda x: sum(x) / sum(abs(x) / (2 - abs(x)))).to_dict()
        update = df['predictions'].map(leaf_values_to_step_size_dict).values
        return leaf_values_to_step_size_dict, update

    def compute_gradient(self, x, y):
        """x: features, y: pseudo response"""
        tree = self.tree()
        tree.build(x, y)
        predictions = tree.predict(x)
        # TODO: assumption: all leaves provide unique value
        self.trees.append(tree)
        leaf_step_size_dict, gradients = self.line_search(predictions, y)
        self.step_sizes.append(leaf_step_size_dict)
        return gradients

    def fit(self, x, y):
        self.base_prediction = 0.5 * log(1 + mean(y) / 1 - mean(y))
        f = self.base_prediction
        for m in range(self.n_estimators):
            pseudo_response = 2 * y / (1 + exp(2 * y * f))
            gradients = self.compute_gradient(x, pseudo_response)
            f = f + self.learning_rate * gradients

    def predict(self, x):
        predictions = []
        for row in x:
            prediction = self.mean + self.learning_rate * array([tree.predict() for tree in self.trees])
            predictions.append(prediction)
        prediction = array(predictions)
        return 1 / (1 + exp(2 * prediction))


if __name__ == '__main__':
    reg = GradientBoostingClassifier(100, 0.01)
    # TODO : make sure y \in {-1,1}
