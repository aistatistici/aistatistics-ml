import numpy as np
import pandas as pd


def open_csv(path):
    with open(path, "r") as f:
        return np.array([l.strip().split(',') for l in f.readlines()])


def open_csv_as_data_frame(path):
    return pd.read_csv(path)
