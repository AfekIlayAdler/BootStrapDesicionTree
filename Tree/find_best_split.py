import pandas as pd
import numpy as np

from Tree.node import Node, NumericBinaryNode, CategoricalBinaryNode, InternalNode


class FindBestSplit:
    def _evaluate(self, x, y):
        raise NotImplementedError

    def get_split(self, df: pd.DataFrame) -> InternalNode:
        raise NotImplementedError


class NumericBestSplitter(FindBestSplit):
    def _evaluate(self, x, y):
        pass

    def get_split(self, df):
        pass


class CategoricalBestSplitter(FindBestSplit):
    def _evaluate(self, x, y):
        pass

    def get_split(self, df):
        pass


class MocCategoricalBinaryBestSplitter(FindBestSplit):
    def __init__(self):
        self.node = CategoricalBinaryNode

    def _evaluate(self, x, y):
        pass

    def get_split(self, df):
        x = df[df.columns[0]]
        col_name = x.name
        score = 1
        right_values = x.unique()[:1]
        left_values = x.unique()[1:]
        return self.node(col_name, score, right_values, left_values)


class MocNumericBinarySplitter(FindBestSplit):
    def __init__(self):
        self.node = NumericBinaryNode

    def _evaluate(self, x, y):
        pass

    def get_split(self, df):
        col = df.columns[0]
        df = df.sort_values(col)
        indexes, col_values, labels = df.index, df.iloc[:, 0], df.iloc[:, 1]
        sigma_values = labels.sum()
        n = df.shape[0]
        left_sum, right_sum = 0 ,sigma_values
        left_sum_square,right_sum_square = 0,np.sum(np.square(labels))
        split_index = None
        best_impurity = np.inf
        for i in range(1,n):
            value = labels[i-1]
            left_sum += value
            right_sum -= value
            left_mean = left_sum/i
            right_mean = right_sum/(n-i)
            #TODO : can be faster???
            left_sum_square += np.square(value)
            right_sum_square -= np.square(value)
            left_std = np.sqrt((left_sum_square/i)-np.square(left_mean )/((n-i)/n))
            left_std = np.sqrt((left_sum_square/i)-np.square(left_mean )/((n-i)/n))
            impurity = (i/n)*left_std + ((n-i)/n)*right_std
            if impurity < best_impurity:
                best_impurity = impurity
                split_index = i
        # TODO : fix cases where i is in the end/ begining
        thr = (col_values[split_index] + col_values[split_index+1])/2
        return self.node(col_values.name, labels.std(), thr)
