from pandas import DataFrame
from numpy import array


class Splitter:
    def __init__(self, numeric_node, categorical_node,response_variable_type,min_samples_leaf):
        self.numeric_node = numeric_node
        self.categorical_node = categorical_node
        self.type = response_variable_type
        self.min_samples_leaf = min_samples_leaf

    def get_split(self, df: DataFrame):
        raise NotImplementedError


class Split:
    def __init__(self, values: array, split_index: int, impurity: float):
        self.impurity = impurity
        self.split_index = split_index
        self.values = values
