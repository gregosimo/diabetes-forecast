'''
A module to predict the number of diabetes cases in all states for the next
three years.

This module uses a linear regression to predict the next several values.'''

import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import cdcdata

FULL_DATA = cdcdata.read_Diabetes_Data_file()

def predict_n(n, state):
    '''Predict the next n years using the linear regression.
    
    This takes in a number of years to predict past the last data point. It
    also requires a state found in the FULL_DATA dataset. It will then run a
    linear regression on all the years in the full dataset.
    
    Predict_n will return a 2-tuple. The first element will be the predictions,
    an array of length n. The second element will be the mean square error of
    the validation set.'''
    # If a row is missing data, ignore it?
    # The next level is to impute it, but I don't believe in that.
    state_data = FULL_DATA[np.logical_and(
        FULL_DATA["State"] == state, ~pandas.isnull(FULL_DATA["Number"]))]
    last_year = int(max(state_data["Year"]))
    next_years = np.arange(n).reshape(-1, 1) + last_year
    year_train, year_test, diabetes_train, diabetes_test = train_test_split(
        state_data["Year"], state_data["Number"], test_size=0.3,
        random_state=53019)
    # Look more into Standardization here.
    lr = LinearRegression(normalize=True)
    year_train = year_train.values.reshape(-1, 1)
    year_test = year_test.values.reshape(-1, 1)
    diabetes_train = diabetes_train.values.reshape(-1, 1)
    diabetes_test = diabetes_test.values.reshape(-1, 1)
    lr.fit(year_train, diabetes_train)
    diabetes_pred = lr.predict(year_test)
    mean_square_error = mean_squared_error(diabetes_test, diabetes_pred)
    diabetes_new = lr.predict(next_years)
    return diabetes_new, mean_square_error

def predict_states():
    '''Predict the next 3 years of diabetes in states.'''
    states = sorted(FULL_DATA["State"].cat.categories)
    datatab = np.zeros((3, len(states)))
    errors = np.zeros(len(states))
    for i, s in enumerate(states):
        preds, mse = predict_n(3, s)
        datatab[:,i] = preds.flatten()
        errors[i] = mse
    return datatab



