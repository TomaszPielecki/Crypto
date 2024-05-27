# Cryptocurrency Prediction Project

This project uses Long Short-Term Memory (LSTM) neural networks to predict cryptocurrency prices based on historical data. It leverages Yahoo Finance API for data retrieval and Keras for building the predictive model. The project is released under the MIT License.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/crypto-prediction.git
    cd crypto-prediction
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the prediction script**:
    ```sh
    python crypto_prediction.py
    ```

2. **Follow the prompts** to enter the number of days to predict, the cryptocurrency symbol (e.g., BTC for Bitcoin), and the currency symbol (e.g., USD for US Dollar).

3. **Scheduled Updates**: The script includes scheduling functionality to update predictions daily. This is handled by the `schedule` library, which runs the `update_data_and_predict` function at midnight every day.

## Project Structure

- **crypto_prediction.py**: The main script that handles data download, preprocessing, model training, prediction, and results visualization.
- **models/**: Directory where trained models are saved.
- **crypto_predictions.db**: SQLite database to store prediction results.

### Main Functions

```python
def main(prediction_days, crypto, against, prediction):
    print(f'Prediction Algorithm: Cryptocurrency Yahoo API')
    prediction_future_date = dt.datetime.now() + dt.timedelta(days=prediction_days)
    print(f'Your cryptocurrency prediction for {crypto} on {prediction_future_date.strftime('%Y-%m-%d')} is {prediction[0][0]:.2f} {against}')


[wybor_Kryptowaluty](https://github.com/TomaszPielecki/Crypto/blob/Kryptowaluty/wybor_Kryptowaluty.png)





