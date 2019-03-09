import os

from ml.utils.csv import open_csv_as_data_frame
from ml.utils.split import split

df = open_csv_as_data_frame(os.path.abspath("./prepared_data/balances.csv"))

subsets = split(df, {
    "Type": [12, 13],
    "Currency": '__all__'
})

folder = os.path.abspath("./prepared_data/balances")
for s in subsets:
    if not os.path.exists(folder):
        os.mkdir(os.path.abspath(folder))
    s[1].to_csv(os.path.join(folder, f"balance_Type_{s[0]['Type']}_Currency_{s[0]['Currency']}.csv"), index=False)
