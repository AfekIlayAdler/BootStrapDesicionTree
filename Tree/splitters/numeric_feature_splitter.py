from Tree.node import NumericBinaryNode
from Tree.splitters.splitter_interface import Splitter

import numpy as np


class NumericFeatureRegressionSplitter(Splitter):
    def __init__(self):
        self.node = NumericBinaryNode

    def _evaluate(self, x, y):
        pass

    def get_split(self, df, n, col):
        df = df.sort_values(col)
        col_values, labels = df.iloc[:, 0], df.iloc[:, 1]
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
        # TODO : handle missing values
        thr = (col_values[split_index - 1] + col_values[split_index]) / 2
        return self.node(col_values.name, labels.std(), thr)


class NumericFeature2ClassClassificationSplitter(Splitter):
    def __init__(self):
        self.node = NumericBinaryNode

    def _evaluate(self, x, y):
        pass

    def get_split(self, df, n, col):
        """
        assume labels are {0,1} and no other values there
        """
        df = df.sort_values(col)
        col_values, labels = df.iloc[:, 0], df.iloc[:, 1]
        assert len()
        left_sum, right_sum = 0, labels.sum()
        split_index = None
        best_impurity = np.inf
        for i in range(1, n):
            value = labels[i - 1]
            left_sum += value
            right_sum -= value
            left_p = left_sum / i
            right_p = right_sum / (n - i)
            impurity = i * left_p * (1 - left_p) + i * right_p * (1 - right_p)
            if impurity < best_impurity:
                best_impurity, split_index = impurity, i
        # TODO : handle missing values
        thr = (col_values[split_index - 1] + col_values[split_index]) / 2
        return self.node(col_values.name, labels.std(), thr)
