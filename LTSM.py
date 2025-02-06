import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Configurations
STOCK_TICKER = "AAPL"
LOOKBACK = 60  # Days to look back for prediction
EPOCHS = 50  # Increased epochs for better training
BATCH_SIZE = 16
MODEL_PATH = "models/stock_lstm.h5"

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Fetch Stock Data
def fetch_stock_data(stock_ticker):
    data = yf.download(stock_ticker, start="2015-01-01", end="2024-01-01")
    data.to_csv("data/stock_data.csv")
    return data

# Preprocessing
def preprocess_data(data):
    # Ensure correct datatype
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

    # Fill missing values if any
    data["Close"].fillna(method="ffill", inplace=True)
    data = data.dropna()  # Drop remaining NaN rows (if any)

    # Scale data
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
    # Load or fetch stock data
    if not os.path.exists("data/stock_data.csv"):
        data = fetch_stock_data(STOCK_TICKER)
    else:
        data = pd.read_csv("data/stock_data.csv", index_col=0, parse_dates=True)

    # Ensure index is in datetime format (handle potential errors)
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors="coerce")  # Coerce errors to NaT
        data = data.dropna(subset=["Close"])  # Drop rows where date conversion failed
    
    # Debugging print after date conversion
    print("First 5 rows after cleaning:\n", data.head())

    # Preprocess data
    X, y, scaler = preprocess_data(data)

    # Train-test split (80% training, 20% testing)
    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    # Check if the model already exists
    if os.path.exists(MODEL_PATH):
        # Load the pre-trained model
        model = load_model(MODEL_PATH)
        print("Loaded existing model.")
    else:
        # Build and train model
        model = build_lstm_model()

        # Implement Early Stopping to stop training if no improvement
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Fit the model with early stopping
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test),
                  verbose=1, callbacks=[early_stopping])

        # Save the trained model
        model.save(MODEL_PATH)
        print(f"Model saved at {MODEL_PATH}")

    # Predict
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Plot Results
    plt.figure(figsize=(10, 5))
    plt.plot(data.index[split + LOOKBACK:], data["Close"][split + LOOKBACK:], label="Actual Price")

    # Ensure predictions align with the correct dates in the test set
    predictions = predictions.flatten()  # Convert from (441, 1) to (441,)
    plt.plot(data.index[split + LOOKBACK:split + LOOKBACK + len(predictions)], predictions, label="Predicted Price", linestyle="dashed")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_and_predict()
