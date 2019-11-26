from typing import Dict
import numpy as np
import pandas as pd


def get_cols_dtypes(df: pd.DataFrame, y_col_name) -> Dict:
    return {i: v.name for i, v in df.dtypes.to_dict().items() if i != y_col_name}


def regression_impurity(y: pd.Series):
    return np.var(y)


def classification_impurity(y: pd.Series):
    p = np.sum(y)
    n = y.size
    return n * p * (1 - p)


impurity_dict = {'regression': regression_impurity, 'classification': classification_impurity}
