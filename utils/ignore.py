from typing import List

import pandas as pd


def ignore(df: pd.DataFrame, ignore_columns: List[str] = None):
    if not ignore_columns:
        ignore_columns = []

    return df.drop(columns=ignore_columns)