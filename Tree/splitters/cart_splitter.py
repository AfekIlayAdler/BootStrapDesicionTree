import numpy as np
import pandas as pd

from Tree.node import NumericBinaryNode, CategoricalBinaryNode
from Tree.splitters.splitter_abstract import Splitter, Split


# Todo: handle cases where min_samples_leaf makes the method to return nothing

class CartRegressionSplitter(Splitter):
    def __init__(self, min_samples_leaf):
        super().__init__(NumericBinaryNode, CategoricalBinaryNode, 'regression', min_samples_leaf)

    @staticmethod
    def evaluate(mrv, mrv_square, counts, min_samples_leaf):
        left_sum, left_counts, left_sum_squares = 0., 0., 0.
        split_index, best_impurity = None, np.inf
        total_sum, total_sum_squares, total_counts = np.sum(counts * mrv), np.sum(counts * mrv_square), np.sum(counts)
        for i in range(1, mrv.size):
            left_sum += mrv[i - 1] * counts[i - 1]
            left_sum_squares += mrv_square[i - 1] * counts[i - 1]
            left_counts += counts[i - 1]
            if min(left_counts, (total_counts - left_counts)) < min_samples_leaf:
                continue
            left_mean = left_sum / left_counts
            right_mean = (total_sum - left_sum) / (total_counts - left_counts)
            left_var = left_sum_squares - left_counts * np.square(left_mean)
            right_var = (total_sum_squares - left_sum_squares) - (total_counts - left_counts) * np.square(right_mean)
            impurity = left_var + right_var
            if impurity < best_impurity:
                best_impurity, split_index = impurity, i
        return best_impurity, split_index

    def get_split(self, df: pd.DataFrame):
        """
        :param df: data frame with columns [MEAN_RESPONSE_VALUE, MEAN_RESPONSE_VALUE_SQUARED, COUNT_COL_NAME]
        and index which is the column values (splitting column)
        """
        data = df.values
        mrv, mrv_square, counts, index = data[:, 0], data[:, 1], data[:, 2], df.index.values
        best_impurity, split_index = self.evaluate(mrv, mrv_square, counts, self.min_samples_leaf)
        return Split(index, split_index, best_impurity)


class CartTwoClassClassificationSplitter(Splitter):
    def __init__(self, min_samples_leaf):
        super().__init__(NumericBinaryNode, CategoricalBinaryNode, 'classification', min_samples_leaf)

    @staticmethod
    def evaluate(mrv, counts, min_samples_leaf):
        # we can look at left sum for example as number of succes in the left split
        left_sum, left_counts = 0., 0.
        split_index, best_impurity = None, np.inf
        total_sum, total_counts = np.sum(counts * mrv), np.sum(counts)
        for i in range(1, mrv.size):
            left_sum += mrv[i - 1] * counts[i - 1]
            left_counts += counts[i - 1]
            if min(left_counts, (total_counts - left_counts)) < min_samples_leaf:
                continue
            left_p = (left_sum / left_counts)
            left_var = left_counts * left_p * (1 - left_p)
            right_p = (total_sum - left_sum) / (total_counts - left_counts)
            right_var = (total_counts - left_counts) * right_p * (1 - right_p)
            impurity = left_var + right_var
            if impurity < best_impurity:
                best_impurity, split_index = impurity, i
        return best_impurity, split_index

    def get_split(self, df: pd.DataFrame):
        """
        :param df: data frame with columns [MEAN_RESPONSE_VALUE, COUNT_COL_NAME]
        and index which is the column values (splitting column)
        """
        data = df.values
        mrv, counts, index = data[:, 0], data[:, 1], df.index.values
        best_impurity, split_index = self.evaluate(mrv, counts, self.min_samples_leaf)
        return Split(index, split_index, best_impurity)
