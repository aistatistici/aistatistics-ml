import os
import numpy as np
from keras.engine.saving import load_model
from matplotlib import pyplot
from keras.losses import mean_squared_error
import keras.backend as K

from ml.utils.csv import open_csv_as_data_frame
from ml.utils.model import dataset_to_time_series

model = load_model('./models/test3.h5')

csv = open_csv_as_data_frame(os.path.abspath('./prepared_data/balances/balance_Type_12_Currency_3.csv'))

out_cols = ['Balance']
nr_out = len(out_cols)
n_in = 30
n_out = 1
n_features = csv.columns.values.size
out = dataset_to_time_series(csv, n_in=n_in, n_out=n_out, n_out_columns=out_cols)
values = out.values
# np.random.shuffle(values)
train_count = int(values.shape[0] * 80 / 100)
train = values[:train_count, :]
test = values[train_count:, :]
# split into input and outputs
train_X, train_y = train[:, :(n_features * n_in)], train[:, -(n_out * nr_out):]
test_X, test_y = test[:, :(n_features * n_in)], test[:, -(n_out * nr_out):]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_in, n_features))
test_X = test_X.reshape((test_X.shape[0], n_in, n_features))


preds = []

for i in range(test_y.shape[0]):
    inp = test_X[i].copy()
    if i > 0:
        for j in range(0, min(i, n_in)):
            inp[-j][0] = preds[-j][0]
    inp = inp.reshape(-1, n_in, n_features)
    p = model.predict(inp)
    preds.append(p[0])

pred = np.asarray(preds)
print(pred.shape)

pyplot.plot(pred[:, 0], label='prediction')
pyplot.plot(test_y[:, 0], 'r', label='ground_truth')
pyplot.legend()
print(K.eval(mean_squared_error(pred[:, 0], test_y[:, 0])))
pyplot.show()
