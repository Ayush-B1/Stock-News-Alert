from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_price_prediction_model(stock_data):
    # Ensure "Close" is in the right format
    X = stock_data[["Close"]].values.reshape(-1, 1)
    y = stock_data["Close"].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    return model, pd.Series(predictions, index=stock_data.index[-len(predictions):])
