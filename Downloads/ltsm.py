import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Configurations
STOCK_TICKER = "AAPL"
LOOKBACK = 60  # Days to look back for prediction
EPOCHS = 20
BATCH_SIZE = 16

# Fetch Stock Data
def fetch_stock_data(stock_ticker):
    data = yf.download(stock_ticker, start="2015-01-01", end="2024-01-01")
    data.to_csv("data/stock_data.csv")
    return data

# Preprocessing
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        X.append(scaled_data[i-LOOKBACK:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # LSTM expects 3D input
    
    return X, y, scaler

# Build LSTM Model
def build_lstm_model():
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Train & Predict
def train_and_predict():
    if not os.path.exists("data/stock_data.csv"):
        data = fetch_stock_data(STOCK_TICKER)
    else:
        data = pd.read_csv("data/stock_data.csv", index_col=0)
    
    X, y, scaler = preprocess_data(data)
    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    
    model = build_lstm_model()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    
    # Plot Results
    plt.figure(figsize=(10,5))
    plt.plot(data.index[split + LOOKBACK:], data["Close"][split + LOOKBACK:], label="Actual Price")
    plt.plot(data.index[split + LOOKBACK:], predictions, label="Predicted Price")
    plt.legend()
    plt.show()
    
    model.save("models/stock_lstm.h5")

if __name__ == "__main__":
    train_and_predict()
