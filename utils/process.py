from math import nan
from typing import Dict

import numpy as np
import pandas as pd


def min_max_normalization(data: np.ndarray):
    tmp = data.sort()
    mi = data[int(10 / 100 * tmp.size)]
    ma = data[int(90 / 100 * tmp.size)]

    return (data - mi) / (ma - mi), {
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
        df[col], column_data[col] = PROCESS_MAP[f](df[col])
    return df, column_data
