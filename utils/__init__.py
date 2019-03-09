from typing import Dict, Any, Tuple, List

import pandas as pd

from ml.utils.augument import augment
from ml.utils.process import process
from ml.utils.split import SplitColumn, split

__all__ = ['process_data']


def process_data(df: pd.DataFrame, split_columns: Dict[str, SplitColumn] = None,
                 augment_columns: Dict[str, Any] = None, process_columns: Dict[str, str] = None):
    if not split_columns:
        split_columns = {}
    if not augment_columns:
        augment_columns = {}
    if not process_columns:
        process_columns = {}

    df, column_data = process(df, process_columns)
    df = augment(df, augment_columns)
    return split(df, split_columns), column_data
