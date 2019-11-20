from Tree.node import CategoricalBinaryNode
from Tree.splitters.splitter_interface import Splitter


class NumericBestSplitter(Splitter):
    def _evaluate(self, x, y):
        pass

    def get_split(self, df):
        pass


class CategoricalBestSplitter(Splitter):
    def _evaluate(self, x, y):
        pass

    def get_split(self, df):
        pass


class MocCategoricalBinaryBestSplitter(Splitter):
    def __init__(self):
        self.node = CategoricalBinaryNode

    def _evaluate(self, x, y):
        pass

    def get_split(self, df):
        x = df[df.columns[0]]
        col_name = x.name
        score = 1
        right_values = x.unique()[:1]
        left_values = x.unique()[1:]
        return self.node(col_name, score, right_values, left_values)