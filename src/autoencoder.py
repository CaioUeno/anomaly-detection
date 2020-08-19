from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, LSTM, Dense, Bidirectional, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils import *


data = pd.read_csv('../data/deaths_data.csv', index_col=['date'])
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

window_size = 7

#get window over time series data
X_train = get_window(data[:127], window_size)
X_test = get_window(data[127:], window_size)

# define model
input_window = Input(shape=(window_size, 1))
bi_lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(input_window)
lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(bi_lstm)
time_dense = TimeDistributed(Dense(16))(lstm)
out = TimeDistributed(Dense(1))(time_dense) 

autoencoder = Model(input_window, out)
autoencoder.compile(loss='mae', optimizer='adam')
autoencoder.summary()

callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=10), 
             ModelCheckpoint('autoencoder.h5', monitor='val_loss', mode='min', save_best_only=False)]

history = autoencoder.fit(X_train, X_train, validation_split=0.2, epochs=150, batch_size=4, shuffle=False, callbacks=callbacks)

pred_train = autoencoder.predict(X_train)
mae_train = np.mean(abs(pred_train - X_train), axis=1)

pred_test = autoencoder.predict(X_test)
mae_test = np.mean(abs(pred_test - X_test), axis=1)

np.save('../data/mae_train.npy', mae_train)
np.save('../data/mae_test.npy', mae_test)