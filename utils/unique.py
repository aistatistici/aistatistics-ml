import pandas as pd


def unique(df: pd.DataFrame, column: str = None):
    print(column)
    if not column:
        return df
    return df.drop_duplicates(subset=column)
