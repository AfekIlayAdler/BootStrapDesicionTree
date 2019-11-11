from typing import Dict

import pandas as pd


def get_cols_dtypes(df: pd.DataFrame) -> Dict:
    return {i: v.name for i, v in df.dtypes.to_dict().items()}
