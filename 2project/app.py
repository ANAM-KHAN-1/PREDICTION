from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import io, base64
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data(ticker):
    df = yf.download(ticker, start="2015-01-01", end="2025-01-01")
    return df[['Close']]

def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    def create_sequences(data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, y_train, X_test, y_test, scaler

def build_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future(model, last_data, scaler, days=30):
    future_predictions = []
    last_data = last_data.reshape(1, -1, 1)
    
    for _ in range(days):
        pred = model.predict(last_data)
        future_predictions.append(pred[0, 0])
        last_data = np.append(last_data[:, 1:, :], [[[pred[0, 0]]]], axis=1)
    
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

def plot_predictions(df, actual, predicted, future_prices):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-len(actual):], actual, label='Actual Price', color='blue')
    plt.plot(df.index[-len(predicted):], predicted, label='Predicted Price', color='red')
    future_dates = pd.date_range(start=df.index[-1], periods=len(future_prices) + 1)[1:]

    plt.plot(future_dates, future_prices, label='Future Prediction', color='green')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    df = fetch_stock_data(ticker)
    X_train, y_train, X_test, y_test, scaler = prepare_data(df)
    model = build_lstm_model()
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    future_prices = predict_future(model, X_test[-1], scaler)
    plot_url = plot_predictions(df, actual_prices, predictions, future_prices)
    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)
