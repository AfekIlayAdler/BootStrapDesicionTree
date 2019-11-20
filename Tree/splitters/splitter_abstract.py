import pandas as pd
import numpy as np


# TODO : handle missing values

class Splitter:
    def __init__(self, node):
        self.node = node

    @staticmethod
    def sort(series: pd.Series) -> pd.Series:
        return series.sort_values()

    @staticmethod
    def group_by_mean_response_value(series: pd.Series) -> pd.Series:
        return series.groupby(level=0).mean()

    def evaluate(self, labels: np.array, n: int) -> tuple:
        pass

    def get_split(self, series: pd.Series, n: int):
        series = self.sort(series)
        col_values, labels = series.index, series.values
        best_impurity, split_index = self.evaluate(labels, n)
        thr = (col_values[split_index - 1] + col_values[split_index]) / 2
        return col_values, split_index, best_impurity, thr

    def get_node(self, series, n, col_name):
        pass
