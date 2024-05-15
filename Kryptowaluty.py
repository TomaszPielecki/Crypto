# -*- coding: utf-8 -*-
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from numpy import array, reshape
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.models import load_model



def main():
    print("Prediction Algorithm: Cryptocurrency Yahoo API")
    prediction_future_date = dt.datetime.now() + dt.timedelta(days=prediction_days)
    print(
        f"Your cryptocurrency prediction for {crypto} on {prediction_future_date.strftime('%Y-%m-%d')} is {prediction[0][0]:.2f} {against}")


def get_user_input():
    while True:
        print("Enter days to predict cryptocurrency exchange rate (type 'exit' to quit):")
        future_days = input()
        if future_days.lower() == 'exit':
            return None, None, None

        try:
            future_days = int(future_days)
            if future_days <= 0:
                raise ValueError("Number of days must be greater than zero.")

            print("Enter cryptocurrency symbol (e.g., BTC, ETH, LTC):")
            crypto = input().upper().strip()
            if any(char.isdigit() for char in crypto):
                raise ValueError("Invalid cryptocurrency symbol.")

            print("Enter currency symbol (e.g., USD, EUR):")
            against = input().upper().strip()
            if any(char.isdigit() for char in against):
                raise ValueError("Invalid currency symbol.")

            return future_days, crypto, against
        except ValueError as ve:
            print(f"Error: {ve}. Please try again.")


def download_data(crypto, against, start_date, end_date):
    data = yf.download(f'{crypto}-{against}', start=start_date, end=end_date, interval='1d')
    if data.empty:
        raise ValueError(f"No data available for {crypto} in {against} during this time period.")
    return data


def prepare_data(data, prediction_days, future_days):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data) - future_days):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x + future_days, 0])

    x_train, y_train = array(x_train), array(y_train)
    x_train = reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler


def build_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),

        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model


def plot_results(test_data, actual_price, prediction_prices, crypto, prediction, future_days):
    plt.figure(figsize=(12, 6))
    plt.title(f'Price forecast of {crypto}')
    plt.plot(test_data.index, actual_price, color='black', label=f'Real prices of {crypto}')
    plt.plot(test_data.index, prediction_prices, color='green', label=f'Predicted prices of {crypto}')
    plt.xlabel(f'Time\nPrediction in {crypto} value = {prediction[0][0]:.2f}, Days now {future_days}')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()


def predict_future(model, scaler, scaled_data, prediction_days):
    future_data = np.array([scaled_data[-prediction_days:, 0]])
    future_data = np.reshape(future_data, (future_data.shape[0], future_data.shape[1], 1))
    future_prediction = model.predict(future_data)
    future_prediction = scaler.inverse_transform(future_prediction)
    return future_prediction


def run_prediction():
    future_days, crypto, against = get_user_input()
    while future_days is not None:
        start_date = dt.datetime.now() - dt.timedelta(365 * 4)
        end_date = dt.datetime.now()

        data = download_data(crypto, against, start_date, end_date)

        prediction_days = 60
        x_train, y_train, scaler = prepare_data(data, prediction_days, future_days)

        model = build_model((x_train.shape[1], 1))
        model.fit(x_train, y_train, epochs=50, batch_size=32)

        test_data = yf.download(f'{crypto}-{against}', start=start_date, end=end_date)
        actual_price = test_data['Adj Close'].values

        total_dataset = concat((data['Adj Close'], test_data['Adj Close']), axis=0)
        model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        scaler.fit(model_inputs)
        model_inputs = scaler.transform(model_inputs)

        x_test = []
        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x - prediction_days:x, 0])

        x_test = array(x_test)
        x_test = reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        prediction_prices = model.predict(x_test)
        prediction_prices = scaler.inverse_transform(prediction_prices)

        prediction = predict_future(model, scaler, model_inputs, prediction_days)

        plot_results(test_data, actual_price, prediction_prices, crypto, prediction, future_days)

        future_days, crypto, against = get_user_input()


if __name__ == '__main__':
    try:
        run_prediction()
        print("End of program.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
