# trading-simulator
## Project Overview
This project is a Financial Trading Strategies Simulator aimed at helping users backtest, optimize, and predict trading strategies based on historical stock data pulled from Yahoo Finance. It uses a machine learning model (specifically LSTM) to predict stock prices and incorporates technical indicators like RSI for strategy optimization. Users can input stock ticker symbols, start & end dates, to visualize the performance of different trading strategies.

## Installation & Setup

### Setup
Clone the repository:

```sh
git clone https://github.com/yourusername/trading-simulator.git
cd trading-simulator
```

Set up a virtual environment (optional but recommended):

```sh
python -m venv simulator
source simulator/bin/activate  # On Windows: .\env\Scripts\simulator
```

Install dependencies:

```sh
pip install -r requirements.txt
```

**Optional**: If you want to train the model yourself (optional since the trained model and scaler are already there under the models folder):
1. Open train_model.ipynb in Google Colab
2. Run the notebook to train the LSTM model and download trading_model.h5 and scaler.pkl
3. Move trading_model.h5 and scaler.pkl to the models/ directory in your local project

Run the Flask application:
```sh
python simulator_app.py
```

Access the application:

Open your web browser and go to http://127.0.0.1:5000.

## Usage

Home Page:

Backtesting Section:
1. Enter a stock ticker symbol, start date, end date, and transaction cost (all dates in format year-month-day)
2. Click 'Run Backtest' to see the performance of the backtest strategy

Optimize Strategy:
1. Click 'Optimize Parameters to find the best parameters for the RSI strategy

Predict Prices
1. Enter a stock ticker symbol, start date, end date (all dates in format year-month-day and limited to 1990-01-01 until 2024-01-01)
2. Click 'Predict' to get the predicted stock prices using the pre-trained LSTM model

## Project Structure
simulator_app.py: The main Flask application
static/: CSS files for styling
templates/: HTML template for home screen
models/: Directory for saving the trained models (trading_model.h5 and scaler.pkl)
train_model.ipynb: Jupyter notebook for training the LSTM model
requirements.txt: List of dependencies to install

## Explanation of Algorithms
1. Backtesting with RSI
Relative Strength Index (RSI): This indicator measures the speed and change of price movements, it indentifies overbought or oversold conditions in a market
Strategy: The RSI strategy buys when the RSI crosses above 30 (indicating an oversold condition) and sells when the RSI crosses below 70 (indicating an overbought condition)
2. Optimization
Parameter Optimization: The optimize_params function iterates over a predefined grid of parameters to find the best-performing set
4. Machine Learning Prediction with LSTM
LSTM Model: Long Short-Term Memory (LSTM) networks can learn order dependence in sequence prediction problems so they are well-suited to time series forecasting because they can capture temporal dependencies.
Data Processing: Historical stock data form the yfinance library in Python using data from Yahoo Finance is scaled and split into sequences to train the model
Prediction: The trained LSTM model predicts future stock prices based on historical data


## Technologies Used
**Backend**: Python, Flask
**Data Processing**: numpy, scikit-learn, yfinance
**Machine Learning**: TensorFlow, Keras
**Frontend**: HTML, CSS, JavaScript
**Model Training**: Google Colab
**Deployment**: Local Flask server
