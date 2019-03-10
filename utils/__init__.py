from typing import Dict, Any, Tuple, List

import pandas as pd

from ml.utils.augument import augment
from ml.utils.ignore import ignore
from ml.utils.process import process
from ml.utils.split import SplitColumn, split
from ml.utils.unique import unique

__all__ = ['process_data']


def process_data(df: pd.DataFrame, split_columns: Dict[str, SplitColumn] = None,
                 augment_columns: Dict[str, Any] = None, process_columns: Dict[str, str] = None,
                 unique_column: str = None, ignore_columns: List[str] = None):
    if not split_columns:
        split_columns = {}
    if not augment_columns:
        augment_columns = {}
    if not process_columns:
        process_columns = {}
    if not ignore_columns:
        ignore_columns = []

    subsets = split(df, split_columns)

    subsets = [(t, process(augment(ignore(unique(s, unique_column), ignore_columns), augment_columns), process_columns)) for t, s in subsets]

    return subsets
