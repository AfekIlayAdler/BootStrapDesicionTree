from typing import Optional

import pandas as pd
import numpy as np

from Tree.config import COUNT_COL_NAME, MEAN_RESPONSE_VALUE_SQUARED, MEAN_RESPONSE_VALUE
from sklearn.model_selection import KFold

from Tree.node import InternalNode

column_order = [MEAN_RESPONSE_VALUE, MEAN_RESPONSE_VALUE_SQUARED, COUNT_COL_NAME]


def general_preprocess(df: pd.DataFrame, col_name: str, label_col_name: str) -> pd.DataFrame:
    df = df[df[col_name].notna()]
    df[COUNT_COL_NAME] = 1
    return df.rename(columns={label_col_name: MEAN_RESPONSE_VALUE})


class GetNode:
    def __init__(self, splitter, col_name, label_col_name, col_type):
        self.col_name = col_name
        self.col_type = col_type
        self.label_col_name = label_col_name
        self.splitter = splitter

    def sort(self, df) -> pd.DataFrame:
        if self.col_type == 'categorical':
            return df.sort_values(by=[MEAN_RESPONSE_VALUE])
        # col_type = 'numeric'
        return df.sort_index()

    def create_node(self, split) -> InternalNode:
        if self.col_type == 'numeric':
            thr = (split.values[split.split_index - 1] + split.values[split.split_index]) / 2
            return self.splitter.numeric_node(self.col_name, split.impurity, thr)
        else:
            left_values, right_values = split.values[:split.split_index], split.values[split.split_index:]
            return self.splitter.categorical_node(self.col_name, split.impurity, left_values, right_values)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = general_preprocess(df, self.col_name, self.label_col_name)
        if self.splitter.type == 'regression':
            df[MEAN_RESPONSE_VALUE_SQUARED] = np.square(df[MEAN_RESPONSE_VALUE])
            return df.groupby(self.col_name, observed=True).agg(
                {MEAN_RESPONSE_VALUE: 'mean', MEAN_RESPONSE_VALUE_SQUARED: 'mean', COUNT_COL_NAME: 'sum'})
        else:
            return df.groupby(self.col_name).agg(
                {MEAN_RESPONSE_VALUE: 'mean', COUNT_COL_NAME: 'sum'})

    def __get(self, df) -> Optional[InternalNode]:
        df = self.preprocess(df)
        if df.shape[0] == 1:
            # it is a pure leaf, we can't split on this node
            return None
        df = self.sort(df)
        split = self.splitter.get_split(df)
        if split.split_index is None:
            # no split that holds min_samples_leaf constraint
            return None
        return self.create_node(split)

    def get(self, df) -> tuple:
        node = self.__get(df)
        if not node:
            return None, None
        return node, node.purity


class KFoldGetNode(GetNode):
    def __init__(self, splitter, col_name, label_col_name, col_type, k_folds=5):
        super().__init__(splitter, col_name, label_col_name, col_type)
        self.k_folds = k_folds

    def calculate_fold_error(self, node, new_samples):
        return None

    def get(self, df):
        best_node = self.__get(df)
        # now we will calculate a real estimate for this impurity using kfold
        error = 0
        kf = KFold(n_splits=self.k_folds)
        for train_index, validation_index in kf.split(df):
            train, validation = df[train_index], df[validation_index]
            node = self.__get(train)
            error += self.calculate_fold_error(node, train, validation)
        mean_error = error / self.k_folds
        setattr(best_node, 'purity', mean_error)
        return best_node
