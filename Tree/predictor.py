import pandas as pd


class Predictor:
    def calc_impurity(self, y: pd.Sereies):
        raise NotImplementedError

    def predict_on_leaf(self, y: pd.Sereies):
        raise NotImplementedError


class Regression(Predictor):
    def calc_impurity(self, y):
        return y.std()

    def predict_on_leaf(self,y):
        return y.mean()

# class TwoClassClassificationPredictor):
