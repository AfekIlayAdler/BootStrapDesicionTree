import pandas as pd

from Tree.node import Node, Leaf
from Tree.predictor import Predictor


class Split:
    def __init__(self, col_name, purity):
        self.col_name = col_name
        self.purity = purity

    def create_node(self, train: pd.DataFrame) -> Node:
        raise NotImplementedError

    @staticmethod
    def create_leaf(y: pd.Series, predictor: Predictor) -> Leaf:
        return Leaf(predictor.predict_on_leaf(y))


class CategoricalBinarySplit(Split):
    def __init__(self, col_name, purity, left_values, right_values):
        super().__init__(col_name, purity)
        self.left_values = left_values
        self.right_values = right_values

    def create_node(self, train: pd.DataFrame) -> Node:
        raise NotImplementedError


class NumericSplit(Split):
    def __init__(self, col_name, purity, thr):
        super().__init__(col_name, purity)
        self.thr = thr

    def create_node(self, train: pd.DataFrame) -> Node:
        raise NotImplementedError
