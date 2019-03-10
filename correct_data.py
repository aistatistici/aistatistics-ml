import os
from datetime import datetime

from ml.utils.csv import open_csv
import numpy as np
import time
import pandas as pd


def date_fom_str(x):
    return datetime.fromtimestamp(time.mktime(time.strptime(x, "%m/%d/%Y")))

def to_float(x):
    return float(x) if x != '' else 0

def to_int(x):
    return int(x) if x != '' else 0


csv_init = open_csv(os.path.abspath("./data/balances.csv"))

head = csv_init[0]
data = csv_init[1:]
data = data[data[:, 1].astype(float) > 0]

data_processed = {}

data_processed[head[0]] = np.vectorize(date_fom_str)(data[:, 0])

data_processed[head[1]] = np.vectorize(to_float)(data[:, 1])
data_processed[head[2]] = np.vectorize(to_float)(data[:, 2])

data_processed[head[3]] = np.vectorize(to_int)(data[:, 3])
data_processed[head[4]] = np.vectorize(to_int)(data[:, 4])


pd.DataFrame(data=data_processed).to_csv("./prepared_data/balances.csv", index=None)