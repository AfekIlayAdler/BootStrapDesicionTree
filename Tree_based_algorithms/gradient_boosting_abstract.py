class GradientBoostingMachine:
    """currently supports only binomial log likelihood as in the original paper of friedman"""

    def compute_gradient(self, x, y):
        raise NotImplementedError

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

