from typing import Tuple, List, Any, Union, Dict

import pandas as pd


def flatten(l: List[List[Any]]):
    return [item for sublist in l for item in sublist]


def split(df: pd.DataFrame, columns: Dict[str, Union[List[Any], str]]) -> \
        List[Tuple[dict, pd.DataFrame]]:
    subsets = [({}, df)]
    for key, values in columns.items():
        subsets = flatten([split_column(s, extra, key, values) for extra, s in subsets])

    return subsets


def split_column(df: pd.DataFrame, extra: Dict[str, Any], key: str, values: Union[List[Any], str]) -> \
        List[Tuple[dict, pd.DataFrame]]:
    if values == '__all__':
        values = df[key] \
            .unique()

    subsets = []

    for v in values:
        dfs = df[df[key] == v]
        dfs = dfs.drop(columns=[key], axis=1)
        subsets.append((dict([(key, v)], **extra), dfs))

    return subsets
