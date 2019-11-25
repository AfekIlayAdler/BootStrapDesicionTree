from Tree.node import CategoricalBinaryNode, NumericBinaryNode, InternalNode
from Tree.splitters.splitter_abstract import Splitter

import numpy as np
import pandas as pd

from Tree.splitters.utils import get_numeric_node, get_categorical_node


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
        self.create_node = get_numeric_node

    def get_node(self, series: pd.Series, n: int, col_name: str) -> InternalNode:
        return self.create_node(self, series, n, col_name)


class CategoricalFeatureR2ClassClassificationSplitter(TwoClassClassification):
    def __init__(self):
        super().__init__(CategoricalBinaryNode)
        self.create_node = get_categorical_node

    def get_node(self, series, n, col_name):
        return self.create_node(self, series, n, col_name)
