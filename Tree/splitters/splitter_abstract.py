from pandas import DataFrame
from numpy import array


class Splitter:
    def get_split(self, df: DataFrame):
        raise NotImplementedError


class Split:
    def __init__(self, values: array, split_index: int, impurity: float):
        self.impurity = impurity
        self.split_index = split_index
        self.values = values
