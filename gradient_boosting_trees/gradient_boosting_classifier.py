from Tree.node import Leaf
from Tree.tree import CartRegressionTree

from numpy import mean, array, log, exp
from pandas import DataFrame

from gradient_boosting_trees.gradient_boosting_abstract import GradientBoostingMachine


class GradientBoostingClassifier(GradientBoostingMachine):
    """currently supports only binomial log likelihood as in the original paper of friedman"""

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
        temp_x = x.copy()
        temp_y = y.copy()
        tree = self.tree(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            min_samples_split=self.min_samples_split)
        tree.fit(temp_x, temp_y)
        predictions = tree.predict(x.to_dict('records'))
        # TODO: assumption: all leaves provide unique value
        self.trees.append(tree)
        leaf_step_size_dict, gradients = self.line_search(predictions, temp_y)
        self.step_sizes.append(leaf_step_size_dict)
        return gradients

    def fit(self, x, y):
        y = 2*y - 1
        self.base_prediction = 0.5 * log(1 + mean(y) / 1 - mean(y))
        f = self.base_prediction
        for m in range(self.n_estimators):
            if m > 0 and isinstance(self.trees[-1].root, Leaf):  # if the previous tree was a bark then we stop
                return
            pseudo_response = y - f
            gradients = self.compute_gradient(X, pseudo_response)
            f += self.learning_rate * gradients
            self.n_trees += 1

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
