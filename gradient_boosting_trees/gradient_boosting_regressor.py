from numpy import mean, array, sum

from Tree.tree import CartRegressionTree, CartRegressionTreeKFold, MIN_SAMPLES_LEAF, MAX_DEPTH, MIN_IMPURITY_DECREASE, \
    MIN_SAMPLES_SPLIT
from Tree.tree_feature_importance import weighted_variance_reduction_feature_importance
from gradient_boosting_trees.gradient_boosting_abstract import GradientBoostingMachine, N_ESTIMATORS, LEARNING_RATE, \
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
        self.features = None
        self.trees = []

    def compute_gradient(self, x, y):
        data = x.copy()
        data[GRADIENT_BOOSTING_LABEL] = y
        tree = self.tree(
            label_col_name=GRADIENT_BOOSTING_LABEL,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            min_samples_split=self.min_samples_split)
        tree.build(data)
        gradients = tree.predict(x.to_dict('records'))
        self.trees.append(tree)
        return gradients

    def fit(self, data):
        y = data[self.label_col_name]
        x = data.drop(columns=[self.label_col_name])
        self.features = x.columns
        self.base_prediction = mean(y)
        f = mean(y)
        for m in range(self.n_estimators):
            pseudo_response = y - f
            gradients = self.compute_gradient(x, pseudo_response)
            f += self.learning_rate * gradients

    def predict(self, data):
        predictions = []
        for row in data:
            prediction = self.mean + self.learning_rate * sum(array([tree.predict(row) for tree in self.trees]))
            predictions.append(prediction)
        return array(predictions)

    def compute_feature_importance(self):
        gbm_feature_importances = {feature: 0 for feature in self.features}
        # TODO : deal with the case that a tree is a bark
        for tree in self.trees:
            tree_feature_importance = weighted_variance_reduction_feature_importance(tree)
            for feature, feature_importance in tree_feature_importance.items():
                gbm_feature_importances[feature] += feature_importance
        return gbm_feature_importances


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
