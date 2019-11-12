from typing import Dict

import pandas as pd


def get_cols_dtypes(df: pd.DataFrame,y_col_name) -> Dict:
    return {i: v.name for i, v in df.dtypes.to_dict().items() if i != y_col_name}
