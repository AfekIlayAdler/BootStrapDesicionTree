import numpy as np

from Tree.node import NumericBinaryNode, CategoricalBinaryNode
from Tree.splitters.splitter_abstract import Splitter


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

    def get_node(self, series, n, col_name):
        col_values, split_index, impurity, thr = self.get_split(series, n)
        return self.node(col_name, impurity, thr)


class CategoricalFeatureRegressionSplitter(Regression):
    def __init__(self):
        super().__init__(CategoricalBinaryNode)

    def get_node(self, series, n, col_name):
        series = self.group_by_mean_response_value(series)
        col_values, split_index, impurity, thr = self.get_split(series, n)
        left_values, right_values = col_values[:split_index], col_values[split_index:]
        return self.node(impurity, col_name, left_values, right_values)
