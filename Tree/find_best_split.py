import pandas as pd

from Tree.node import Node


class FindBestSplit:
    def _evaluate(self, x, y):
        raise NotImplementedError

    def get_split(self, df: pd.DataFrame) -> Node:
        raise NotImplementedError


class NumericBestSplitter(FindBestSplit):
    def _evaluate(self, x, y):
        pass

    def get_split(self, df):
        pass


class CategoricalBestSplitter(FindBestSplit):
    def _evaluate(self, x, y):
        pass

    def get_split(self,df):
        pass


class MocCategoricalBinaryBestSplitter(FindBestSplit):
    def __init__(self):
        self.node = CategoricalBinaryNode

    def _evaluate(self, x, y):
        pass

    def get_split(self,df):
        x = df[df.columns[0]]
        col_name = x.name
        score = 1
        right_values = x.unique()[:1]
        left_values = x.unique()[1:]
        return self.node(col_name, score, right_values, left_values)


class MocNumericBestSplitter(FindBestSplit):
    def __init__(self):
        self.node = NumericSplit

    def _evaluate(self, x, y):
        pass

    def get_split(self, df) :
        col = df[df.columns[0]]
        return self.node(col.name, 1, 1)
