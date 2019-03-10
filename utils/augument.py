from datetime import datetime
from typing import List, Tuple, Any, Dict

import pandas as pd
import numpy as np
from forex_python.converter import CurrencyRates

CURRENCY_MAP = {
    1: 'EUR',
    2: 'USD',
    3: 'RON',
    4: 'GBP'
}

c = CurrencyRates()

augmented_date_headers = ["Day", "Mouth", "Year",
                          *("Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()),
                          *("Winter Spring Summer Autumn".split()),
                          "Exchange Rate"]


def augment_date(column: np.ndarray, currency: int):
    column = column.astype(int)
    def func(date: int):
        ns = 1e-9
        date = datetime.utcfromtimestamp(date * ns)
        weekday = date.weekday()
        weekdays = [0] * 7
        weekdays[weekday] = 1
        season = [0] * 4
        season[(date.month % 12) // 3] = 1
        exchange_rate = 0 #c.get_rate(CURRENCY_MAP[currency], 'RON', date_obj=date)
        day = date.day
        month = date.month
        year = date.year

        return [day, month, year, *weekdays, *season, exchange_rate]

    return np.array([func(f) for f in column]), augmented_date_headers


AUGMENT_MAP = [
    (datetime, augment_date),
    (np.datetime64, augment_date)
]


def augment(df: pd.DataFrame, columns: Dict[str, Any]):
    for c, v in columns.items():
        t = df[c].dtype
        func = ([f for d, f in AUGMENT_MAP if np.issubdtype(t, d)][0:1] or [None])[0]
        if not func:
            continue
        col_data = df[c].values
        aug_data, aug_headers = func(col_data, v)
        df = df.drop(columns=[c])
        for i in range(aug_data.shape[1]):
            df[aug_headers[i]] = aug_data[:, i]
    return df
