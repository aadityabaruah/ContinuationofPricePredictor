from django.shortcuts import render
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf


def home(request):
    return render(request, 'prediction/home.html')


def predict(request):
    date = request.GET.get('date', '')
    date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
    predicted_price = predict_bitcoin_price(date_obj)
    return render(request, 'prediction/predict.html', {'predicted_price': predicted_price})


def predict_bitcoin_price(date_obj):
    # Download historical Bitcoin prices using yfinance
    bitcoin_data = yf.download(tickers='BTC-USD', start='2015-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'))

    # Prepare the data
    bitcoin_data = bitcoin_data.reset_index()
    bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])
    bitcoin_data['Timestamp'] = bitcoin_data['Date'].apply(lambda x: x.timestamp())
    X = bitcoin_data['Timestamp'].values.reshape(-1, 1)
    y = bitcoin_data['Close'].values

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the Bitcoin price for the given date
    timestamp = np.array([date_obj.timestamp()]).reshape(-1, 1)
    predicted_price = model.predict(timestamp)

    return predicted_price[0]