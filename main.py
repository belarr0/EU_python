import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

data = pd.read_csv('temperature_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

features = data[['Temperature']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

X, y = [], []
time_step = 10
for i in range(len(scaled_features) - time_step - 1):
    X.append(scaled_features[i:(i + time_step), 0])
    y.append(scaled_features[i + time_step, 0])
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=16)

predicted = model.predict(X_test)
y_test_reshaped = y_test.reshape(-1, 1)
predicted_reshaped = predicted.reshape(-1, 1)
actual = scaler.inverse_transform(y_test_reshaped)
predicted = scaler.inverse_transform(predicted_reshaped)

results = pd.DataFrame({'Actual': actual.flatten(), 'Predicted': predicted.flatten()})
print(results)