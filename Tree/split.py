from Tree.node import Node, NumericBinaryNode
import pandas as pd

from Tree.predictor import Predictor


class Split:
    def __init__(self,col_name,score):
        self.col_name = col_name
        self.score = score

    def do_split(self, train: pd.DataFrame) -> Node:
        raise NotImplementedError

    def create_leaf(self,y: pd.Series) -> Node:
        raise NotImplementedError


class CategoricalBinarySplit(Split):
    def __init__(self, col_name, score, left_values, right_values):
        super().__init__(col_name, score)
        self.left_values = left_values
        self.right_values = right_values

    def do_split(self, train: pd.DataFrame) -> Node:
        raise NotImplementedError

    def create_leaf(self,y,predictor:Predictor) -> Node:
        prediction = predictor.predict_on_leaf(y)
        NumericBinaryNode(prediction = prediction, is_leaf = True)






        raise NotImplementedError