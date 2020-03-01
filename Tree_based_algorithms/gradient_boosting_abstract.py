N_ESTIMATORS = 30
LEARNING_RATE = 0.01
GRADIENT_BOOSTING_LABEL = 'y_residual'


class GradientBoostingMachine:
    """currently supports only binomial log likelihood as in the original paper of friedman"""

    def compute_gradient(self, x, y):
        raise NotImplementedError

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
