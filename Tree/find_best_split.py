from Tree.split import Split, CategoricalBinarySplit, NumericSplit
import pandas as pd


class FindBestSplit:
    def _evaluate(self, x, y):
        raise NotImplementedError

    def get_split(self, df:pd.DataFrame) -> Split:
        raise NotImplementedError


class NumericBestSplitter(FindBestSplit):
    def _evaluate(self, x, y):
        pass

    def get_split(self, df) -> Split:
        pass


class CategoricalBestSplitter(FindBestSplit):
    def _evaluate(self, x, y):
        pass

    def get_split(self,df) -> Split:
        pass


class MocCategoricalBinaryBestSplitter(FindBestSplit):
    def __init__(self):
        self.split = CategoricalBinarySplit

    def _evaluate(self, x, y):
        pass

    def get_split(self,df) -> Split:
        x = df[df.columns[0]]
        col_name = x.name
        score = 1
        right_values = x.unique()[:1]
        left_values = x.unique()[1:]
        return self.split(col_name, score, right_values, left_values)


class MocNumericBestSplitter(FindBestSplit):
    def __init__(self):
        self.split = NumericSplit

    def _evaluate(self, x, y):
        pass

    def get_split(self, df) -> Split:
        col = df[df.columns[0]]
        return self.split(col.name, 1, 1)
