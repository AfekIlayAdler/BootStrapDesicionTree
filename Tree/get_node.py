import pandas as pd
import numpy as np

from Tree.config import COUNT_COL_NAME, MEAN_RESPONSE_VALUE_SQUARED, MEAN_RESPONSE_VALUE

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

    # TODO understand if groupby is avoidable
    def preprocess_data_for_regression(self, df: pd.DataFrame) -> pd.DataFrame:
        df = general_preprocess(df, self.col_name, self.label_col_name)
        df[MEAN_RESPONSE_VALUE_SQUARED] = np.square(df[MEAN_RESPONSE_VALUE])
        return df.groupby(self.col_name, observed=True).agg(
            {MEAN_RESPONSE_VALUE: 'mean', MEAN_RESPONSE_VALUE_SQUARED: 'mean', COUNT_COL_NAME: 'sum'})

    def preprocess_data_for_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        df = general_preprocess(df, self.col_name, self.label_col_name)
        return df.groupby(self.col_name).agg(
            {MEAN_RESPONSE_VALUE: 'mean', COUNT_COL_NAME: 'sum'})
        # return df.set_index(self.col_name)[column_order]

    def create_node(self, split):
        # Todo change self.splitter.node to self.splitter.numeric node and categorical node
        if self.col_type == 'numeric':
            thr = (split.values[split.split_index - 1] + split.values[split.split_index]) / 2
            return self.splitter.numeric_node(self.col_name, split.impurity, thr)
        else:
            left_values, right_values = split.values[:split.split_index], split.values[split.split_index:]
            return self.splitter.categorical_node(self.col_name, split.impurity, left_values, right_values)

    def get_preprocessor(self):
        if self.splitter.type == 'regression':
            return self.preprocess_data_for_regression
        return self.preprocess_data_for_classification

    def get(self, df):
        preprocessor = self.get_preprocessor()
        df = preprocessor(df)
        if df.shape[0] == 1:
            # it is a pure leaf, we can't split on this node
            return None
        df.sort_values(by=[MEAN_RESPONSE_VALUE], inplace=True)
        split = self.splitter.get_split(df)
        if split.split_index is None:
            # no split that holds min_samples_leaf constraint
            return None
        return self.create_node(split)
