import os
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot

from ml.utils.csv import open_csv_as_data_frame
from ml.utils.model import dataset_to_time_series

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


model = Sequential()
model.add(LSTM(40, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(nr_out * n_out))
model.compile(loss='mse', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(test_X, test_y), shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

if not os.path.exists('./models'):
    os.mkdir(os.path.abspath('./models'))
model.save('./models/test3.h5')

print(model.predict(test_X))