from flask import Flask, request, render_template, jsonify
import numpy as np
import yfinance as yf
import tensorflow as tf
import pickle

app = Flask(__name__)

# loading pre-trained LSTM model that was trained in Google Colab
# print("before")
model = tf.keras.models.load_model('models/trading_model.h5')
# print("done")

# loading minmaxscaler for scaling input data
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# optimization function using grid search
def optimize_params(rsi, grid):
    best_params = None
    best_performance = -(np.inf)

    for params in grid:
        # calling rsi with the params
        current = rsi(**params, window = 10)
        if current > best_performance:
            # updating with best values
            best_performance = current
            best_params = params
    return best_params

# prediction function using the model
def model_predict(data):
    # scaling data
    scaled = scaler.transform(data)
    # making predictions
    predictions = model.predict(scaled)

    return predictions

# using RSI for trading strategy
def rsi(data, window):
    # just doing RSI because its a more common trading strategy
    # but implementing a more simple version of it since I'm a beginner

    # calculating price changes
    delta = data['Close'].diff()

    # separating gains & losses
    gains = delta.where(delta > 0, 0)
    losses = delta.where(delta < 0, 0)
    losses = -losses

    # calculating average gains & losses
    avg_gain = gains[:window].mean()
    avg_loss = losses[:window].mean()

    # calculating relative strength and preventing divide by 0 error
    if avg_loss == 0:
        RS = -np.inf
    else:
        RS = avg_gain / avg_loss

    # calculating RSI
    RS += 1
    RSI = 100 - (100 / RS)

    # getting buy/sell signals
    buy = RSI < 30
    sell = RSI > 70

    # calculating performance metric
    buys = data['Close'][buy]
    sells = data['Close'][sell]

    # updating performance value
    if len(buys) > 0:
        performance = len(sells) / len(buys)
    else:
        performance = 0
 
    return performance

# backtesting function which accounts for transaction cost since that's a major factor to daily returns
def backtest_func(data, transaction_cost):

    close_prices = data['Close'].values

    # getting daily returns
    temp = close_prices[:-1]
    returns = np.diff(close_prices) / temp

    # deducting transaction costs:
    returns -= transaction_cost

    # getting average daily return
    performance = np.mean(returns)

    return performance

@app.route('/')
def home():
    # rendering home page
    return render_template('home.html')

@app.route('/backtest_func', methods=['POST'])
def backtest():
    # getting user input to form data
    symbol = request.form['symbol']
    start = request.form['start_date']
    end = request.form['end_date']
    data = yf.download(symbol, start, end)
    cost = float(request.form['transaction_cost'])

    # doing actual backtesting
    performance = backtest_func(data, cost)

    return jsonify({'performance': performance})

@app.route('/optimize_params', methods=['POST'])
def optimize():
    # grid for optimizations
    grid = [
        {'param1': 10, 'param2': 0.01},
        {'param1': 20, 'param2': 0.02},
    ]

    # optimizing the parameters
    best_params = optimize_params(rsi, grid)

    return jsonify(best_params) 

@app.route('/model_predict', methods=['POST'])
def predict():
    # getting user input to form data
    symbol = request.form['symbol']
    start = request.form['start_date']
    end = request.form['end_date']
    data = yf.download(symbol, start, end)

    # predictions using ML model
    predictions = model_predict(data)
    predictions = predictions.tolist()

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
