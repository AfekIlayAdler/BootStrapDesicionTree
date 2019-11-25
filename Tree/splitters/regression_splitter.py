import numpy as np
import pandas as pd

from Tree.node import NumericBinaryNode, CategoricalBinaryNode, InternalNode
from Tree.splitters.splitter_abstract import Splitter
from Tree.splitters.utils import get_numeric_node, get_categorical_node


class Regression(Splitter):
    def __init__(self, node):
        super().__init__(node)

    def evaluate(self, labels, n):
        left_sum, right_sum = 0, labels.sum()
        left_sum_square, right_sum_square = 0, np.sum(np.square(labels))
        split_index = None
        best_impurity = np.inf
        for i in range(1, n):
            value = labels[i - 1]
            left_sum += value
            right_sum -= value
            left_mean = left_sum / i
            right_mean = right_sum / (n - i)
            left_sum_square += np.square(value)
            right_sum_square -= np.square(value)
            left_var = left_sum_square - i * np.square(left_mean)
            right_var = left_sum_square - (n - i) * np.square(right_mean)
            # impurity = (i / n) * left_std + ((n - i) / n) * right_std
            impurity = left_var + right_var
            if impurity < best_impurity:
                best_impurity, split_index = impurity, i
        return best_impurity, split_index


class NumericFeatureRegressionSplitter(Regression):
    def __init__(self):
        super().__init__(NumericBinaryNode)

    def get_node(self, series: pd.Series, n: int, col_name: str) -> InternalNode:
        return get_numeric_node(self,series, n, col_name)


class CategoricalFeatureRegressionSplitter(Regression):
    def __init__(self):
        super().__init__(CategoricalBinaryNode)

    def get_node(self, series, n, col_name):
        return get_categorical_node(self, series, n, col_name)
