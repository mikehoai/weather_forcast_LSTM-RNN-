import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Preprocess data
data = pd.read_csv('weather_hanoi_test.csv')
data['datetime'] = pd.to_datetime(data['datetime']) 
data = data.set_index('datetime')
data = data.fillna(data.mean(numeric_only=True))

target_variable = 'temp'
data_to_use = data[[target_variable]].values

# Use MinMaxScaler to scale the data to the range of 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_to_use)

# Prepair data to LSTM model
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60

X, y = create_sequences(scaled_data, seq_length)

# Train data
train_size = int(len(X) * 0.8)
# Validation data
val_size = int(len(X) * 0.1)
# Test data
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]


# Bulid LSTM model
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1)) 
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Save the trained model
model.save('weather_hanoi.keras')

# Predict
y_pred = model.predict(X_test)
y_pred_original = scaler.inverse_transform(y_pred)
y_test_original = scaler.inverse_transform(y_test)

# Validate model performance using RMSE
rmse = np.sqrt(np.mean((y_pred_original - y_test_original)**2))
print(f"RMSE: {rmse}")

# Visualize the training and validation loss and reward
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot the actual vs predicted temperature
plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label='Actual')
plt.plot(y_pred_original, label='Predicted')
plt.legend()
plt.title('Temperature Prediction with LSTM')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.show()

# Predict 7-day temperature
last_sequence = scaled_data[-seq_length:] 
predictions = []

for _ in range(7):
    next_pred = model.predict(last_sequence.reshape(1, seq_length, 1))
    predictions.append(next_pred[0, 0])
    last_sequence = np.append(last_sequence[1:], next_pred, axis=0)

predictions_original = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(predictions_original, label='Predicted Temperature')
plt.title('7-Day Temperature Prediction')
plt.xlabel('Days')
plt.ylabel('Temperature')
plt.legend()
plt.show()
