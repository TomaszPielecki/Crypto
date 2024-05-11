# -*- coding: utf-8 -*-
import datetime as dt
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from numpy import array, reshape
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, LSTM, Dense
# from keras.layers import Dropout, LSTM, Dense
# from keras.models import Sequential
# import os
#
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def main():
    print(
        "Prediction Algorithm: Cryptocurrency Yahoo API")
    print(f"Your cryptocurrency prediction {crypto},\non {prediction_future_days} is {prediction}")


def get_user_input():
    while True:
        print("Prediction Algorithm: Enter days to predict cryptocurrency exchange rate (type 'exit' to quit):")
        future_days = input()
        if future_days.lower() == 'exit':
            return None, None, None

        try:
            future_days = int(future_days)
            if future_days <= 0:
                raise ValueError("Number of days must be greater than zero.")

            print("Enter cryptocurrency symbol e.g. BTC, ETH, LTC:")
            crypto = input().upper()
            if any(char.isdigit() for char in crypto):
                raise ValueError("Entered cryptocurrency name is invalid.")

            print("Enter currency symbol e.g. USD, EUR:")
            against = input().upper()
            if any(char.isdigit() for char in against):
                raise ValueError("Entered currency name is invalid.")

            return future_days, crypto, against
        except ValueError as ve:
            print(f"Error: {ve}. Please try again.")


try:
    future_days, crypto, against = get_user_input()
    while future_days is not None:
        start_date = dt.datetime.now() - dt.timedelta(365 * 4)
        end_date = dt.datetime.now()

        data = yf.download(f'{crypto}-{against}', start=start_date, end=end_date, interval='1d')
        if data.empty:
            raise ValueError(
                f"No data available for cryptocurrency {crypto} in currency {against} during this time period.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

        prediction_days = 60

        x_train, y_train = [], []

        for x in range(prediction_days, len(scaled_data) - future_days):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x + future_days, 0])

        x_train, y_train = array(x_train), array(y_train)
        x_train = reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()

        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

        model.fit(x_train, y_train, epochs=50, batch_size=32)

        test_start = dt.datetime.now() - dt.timedelta(365 * 4)
        test_end = dt.datetime.now()
        test_data = yf.download(f'{crypto}-{against}', start=test_start, end=test_end)
        actual_price = test_data['Adj Close'].values

        total_dataset = concat((data['Adj Close'], test_data['Adj '
                                                             'Close']), axis=0)
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

        # Add future prediction
        future_data = np.array([scaled_data[-prediction_days:, 0]])
        future_data = np.reshape(future_data, (future_data.shape[0], future_data.shape[1], 1))
        future_prediction = model.predict(future_data)
        future_prediction = scaler.inverse_transform(future_prediction)
        prediction = future_prediction[0][0]

        real_data = [model_inputs[len(model_inputs) + future_days - prediction_days:len(model_inputs) + 1, 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        prediction_future_days = dt.datetime.now() + dt.timedelta(future_days)

        # Plot prediction prices
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, actual_price, color='black', label=f'Real prices of {crypto}')
        plt.plot(test_data.index, prediction_prices, color='green', label=f'Predicted prices of {crypto}')
        plt.xlabel(f'Time\nPrediction  in {crypto} value = {prediction}, Days now {prediction_future_days}')
        plt.ylabel('Price')
        plt.title(f'Price forecast of {crypto} for {future_days} days')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

        future_days, crypto, against = get_user_input()
        main()
    print("End of program.")

except ValueError as ve:
    print(f"Error: {ve}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
