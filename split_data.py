import os
import numpy as np

from ml.utils import process_data
from ml.utils.csv import open_csv_as_data_frame

df = open_csv_as_data_frame(os.path.abspath("./prepared_data/balances.csv"), ["Date"])
print(df["Date"].dtypes.type == np.datetime64)
subsets = process_data(df, {
    "Type": [12, 13],
    "Currency": '__all__'
}, {
}, {
    "Avg Rate (for Balance)": 'inverse',
    "Balance": 'min_max_normalization'
}, "Date", ["Date"])

folder = os.path.abspath("./prepared_data/balances")
for d, (s, _) in subsets:
    if not os.path.exists(folder):
        os.mkdir(os.path.abspath(folder))
    s.to_csv(os.path.join(folder, f"balance_Type_{d['Type']}_Currency_{d['Currency']}.csv"), index=False)
