from math import nan
from typing import Dict

import numpy as np
import pandas as pd


def min_max_normalization(data: np.ndarray):
    tmp = np.sort(data)
    mi = tmp[0]
    ma = tmp[-1]

    i = ma - mi

    mi = mi + i * 10 / 100
    ma = ma - i * 10 / 100
    return (data - mi) / i, {
        'min': mi,
        'max': ma
    }


def inverse(data: np.ndarray):
    ret = np.zeros(data.shape)
    ret.fill(nan)
    inds = data > 0
    ret[inds] = 1 / data[inds]
    return ret, {}


PROCESS_MAP = {
    "inverse": inverse,
    "abs": lambda x: (np.abs(x), {}),
    "log": lambda x: (np.log10(x), {}),
    "min_max_normalization": min_max_normalization
}


def process(df: pd.DataFrame, columns: Dict[str, str]):
    column_data = {}
    for col, f in columns.items():
        df[col], tmp = PROCESS_MAP[f](df[col].values)
        column_data[col] = {
            'process': dict([(f, tmp)])
        }
    return df, column_data
