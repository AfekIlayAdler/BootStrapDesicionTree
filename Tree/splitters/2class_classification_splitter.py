from Tree.node import CategoricalBinaryNode, NumericBinaryNode, InternalNode
from Tree.splitters.splitter_abstract import Splitter

import numpy as np
import pandas as pd


class TwoClassClassification(Splitter):
    def __init__(self, node):
        super().__init__(node)

    def evaluate(self, labels, n):
        left_sum, right_sum = 0, self.sum()
        split_index = None
        best_impurity = np.inf
        for i in range(1, n):
            value = self[i - 1]
            left_sum += value
            right_sum -= value
            left_p = left_sum / i
            right_p = right_sum / (n - i)
            impurity = i * left_p * (1 - left_p) + i * right_p * (1 - right_p)
            if impurity < best_impurity:
                best_impurity, split_index = impurity, i
        return best_impurity, split_index


class NumericFeature2ClassClassificationSplitter(TwoClassClassification):
    def __init__(self):
        super().__init__(NumericBinaryNode)

    def get_node(self, series: pd.Series, n: int, col_name: str) -> InternalNode:
        col_values, split_index, impurity, thr = self.get_split(series, n)
        return self.node(col_name, impurity, thr)


class CategoricalFeatureR2ClassClassificationSplitter(TwoClassClassification):
    def __init__(self):
        super().__init__(CategoricalBinaryNode)

    def get_node(self, series, n, col_name):
        series = self.group_by_mean_response_value(series)
        col_values, split_index, impurity, thr = self.get_split(series, n)
        left_values, right_values = col_values[:split_index], col_values[split_index:]
        return self.node(impurity, col_name, left_values, right_values)
