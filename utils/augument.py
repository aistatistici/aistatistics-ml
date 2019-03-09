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
    def func(date: datetime):
        weekday = date.weekday()
        weekdays = [0] * 7
        weekdays[weekday] = 1
        season = [0] * 4
        season[(date.month % 12) // 3] = 1
        exchange_rate = c.get_rate(CURRENCY_MAP[currency], 'RON', date_obj=date)
        day = date.day
        month = date.month
        year = date.year

        return np.array([day, month, year, *weekdays, *season, exchange_rate])

    return np.vectorize(func)(column), augmented_date_headers


AUGMENT_MAP = [
    (datetime, augmented_date_headers)
]


def augment(df: pd.DataFrame, columns: Dict[str, Any]):
    types = df.dtypes

    for c, v in columns:
        t = types[c]
        func = [f for c, f in AUGMENT_MAP if c == t][0][1]
        col_data = df[c].values
        aug_data, aug_headers = func(col_data, v)
        df.drop(c)
        for i in range(aug_data.shape[0]):
            df[aug_headers[i]] = aug_data[i]
    return df
