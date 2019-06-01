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
import matplotlib.pyplot as plt

FULL_DATA = cdcdata.read_Diabetes_Data_file()

def predict_n(n, state):
    '''Predict the next n years using the linear regression.
    
    This takes in a number of years to predict past the last data point. It
    also requires a state found in the FULL_DATA dataset. It will then run a
    linear regression on all the years in the full dataset.
    
    Predict_n will return a 2-tuple. The first element will be the predictions,
    an array of length n. The second element will be the mean square error of
    the validation set.'''
    last_year = int(max(state_data["Year"]))
    next_years = np.arange(n).reshape(-1, 1) + last_year
    diabetes_pred = lr.predict(year_test)
    mean_square_error = mean_squared_error(diabetes_test, diabetes_pred)
    diabetes_new = lr.predict(next_years)
    return diabetes_new, mean_square_error

def read_test_train_full(state, test_size=0.3):
    '''Get a testing and training set from the full dataset.

    This function reads the year and diabetes number from FULL_DATA for a
    specific state, and then splits it into a testing and training set. The
    fraction of testing points should be specified in test_size.
    
    This function will return a 4-tuple containing:
    
    (year_train, year_test, diabetes_train, diabetes_test)'''
    # If a row is missing data, ignore it?
    # The next level is to impute it, but I don't believe in that.
    state_data = FULL_DATA[np.logical_and(
        FULL_DATA["State"] == state, ~pandas.isnull(FULL_DATA["Number"]))]
    # Split the data randomly into testing and training sets 70/30 and return
    # the 4-tuple.
    return train_test_split(
        state_data["Year"], state_data["Number"], test_size=test_size,
        random_state=53019)

def full_linear_model(year_train, diabetes_train):
    '''Train a linear model on the given diabetes dataset.

    This model will be trained on all the data stored in FULL_DATA for the
    specific state. It will return a LinearModel instance.'''

    # The normalize=True keyword normalizes the X-dimension of the data. I
    # don't think it really matters here, but other sources on data science
    # note that you should do a process to scale the data so that it looks like
    # a normal distribution, or a flat distribution. It may be important for
    # more sophisticated analyses.
    lr = LinearRegression(normalize=True)
    # lr.fit requires 2-D arrays. The first dimension should be the samples (16
    # in this case). The second dimension should be the number of predictors
    # (1; just the year).
    year_train = year_train.values.reshape(-1, 1)
    diabetes_train = diabetes_train.values.reshape(-1, 1)
    # Now this fits the training data to the linear model.
    lr.fit(year_train, diabetes_train)
    return lr


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

def graph_state(state):
    '''Graph the historical and predicted data for a given state.'''
    # Get the testing and training data
    year_train, year_test, diabetes_train, diabetes_test = read_test_train_full(
        state, test_size=0.3)
    # Get the lienar model for the trained data.
    lr = full_linear_model(year_train, diabetes_train)

    # Now rejoin to make the full dataset.
    full_year = np.concatenate([year_train, year_test])
    full_diabetes = np.concatenate([diabetes_train, diabetes_test])
    print(full_year)
    print(full_diabetes)

    plt.plot(full_year, full_diabetes, 'bd')
