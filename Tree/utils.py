from typing import Dict
import numpy as np
import pandas as pd

from Tree.config import PANDAS_CATEGORICAL_COLS


def get_cols_dtypes(df: pd.DataFrame, y_col_name) -> Dict:
    return {i: v.name for i, v in df.dtypes.to_dict().items() if i != y_col_name}


def get_col_type(col_type):
    if col_type in PANDAS_CATEGORICAL_COLS:
        return 'categorical'
    return 'numeric'


def regression_impurity(y: pd.Series):
    return np.sum(np.square(y) - np.mean(y))


def classification_impurity(y: pd.Series):
    p = np.sum(y)/y.size
    n = y.size
    return n * p * (1 - p)


impurity_dict = {'regression': regression_impurity, 'classification': classification_impurity}
