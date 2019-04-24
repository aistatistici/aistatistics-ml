import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dense


def dataset_to_time_series(df: pd.DataFrame, n_in, n_out, n_out_columns, dropnan=True):
    # convert series to supervised learning
    if n_out_columns is None:
        n_out_columns = []
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, -1, -1):
        if i == 0:
            tmp = df.shift(i)
            for i in n_out_columns:
                tmp[i] = np.zeros(tmp[i].values.shape)

            cols.append(tmp)
            names += [('%s(t)' % j) for j in df.columns.values]
        else:
            cols.append(df.shift(i))
            names += [('%s(t-%d)' % (j, i)) for j in df.columns.values]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        tmp = df.shift(-i)[n_out_columns]
        cols.append(tmp)
        if i == 0:
            names += [('Output %s(t)' % j) for j in n_out_columns]
        else:
            names += [('Output %s(t+%d)' % (j, i)) for j in n_out_columns]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def get_nr_features(df: pd.DataFrame):
    return df.columns.values.size


def split_data(df: pd.DataFrame, n_features, n_in, n_out, n_out_columns, ratio=0.8):
    values = df.values
    # np.random.shuffle(values)
    nr_out = len(n_out_columns)
    train_count = int(values.shape[0] * 80 / 100)
    train = values[:train_count, :]
    test = values[train_count:, :]
    # split into input and outputs
    train_X, train_y = train[:, :(n_features * n_in)], train[:, -(n_out * nr_out):]
    test_X, test_y = test[:, :(n_features * n_in)], test[:, -(n_out * nr_out):]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_in, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_in, n_features))

    return (train_X, train_y), (test_X, test_y)


def generate_model(n_features, n_in, n_out, n_out_columns):
    nr_out = len(n_out_columns)
    model = Sequential()
    model.add(LSTM(40, input_shape=(n_in, n_features)))
    model.add(Dense(nr_out * n_out))
    model.compile(loss='mse', optimizer='adam')

    return model


def train_model(model, train_X, train_y, test_X, test_y):
    return model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(test_X, test_y), shuffle=False)


def predict(model, test_X, n_features, n_in, with_ground_truth=True):
    if not with_ground_truth:
        preds = []
        for i in range(test_X.shape[0]):
            inp = test_X[i].copy()
            if i > 0:
                for j in range(0, min(i, n_in)):
                    inp[-j][0] = preds[-j][0]
            inp = inp.reshape(-1, n_in, n_features)
            p = model.predict(inp)
            preds.append(p[0])
        return np.asarray(preds)
    else:
        return model.predict(test_X)
