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

def read_test_train_since2011(state, test_size=0.3):
    '''Get a testing and training set only since 2011.
    
    This function reads the year and diabetes number for all FULL_DATA records
    starting with the year 2011. This year marked the first main revision of 
    the BFRSS dataset.'''
    # If a row is missing data, ignore it?
    # The next level is to impute it, but I don't believe in that.
    state_data = FULL_DATA[np.logical_and(
        np.logical_and(
            FULL_DATA["State"] == state, ~pandas.isnull(FULL_DATA["Number"])), 
        FULL_DATA["Year"] >= 2011.0)]
    # Split the data randomly into testing and training sets 70/30 and return
    # the 4-tuple.
    return train_test_split(
        state_data["Year"], state_data["Number"], test_size=test_size,
        random_state=53019)

def linear_model(year_train, diabetes_train):
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

def predict_lr(lr, preddata):
    '''Make a prediction for a linear regression model.

    Return the predicted values of preddata for the linear regression model.
    While lr.predict(preddata) should nominally work, this function
    automatically converts preddata into a 2-d array, which is needed by lr.'''
    # If preddata is 1-d, we need to make it 2-d.
    if len(preddata.shape) == 1:
        # If it's a Series, then it should have a values attribute.
        try:
            preddata = preddata.values.reshape(-1, 1)
        except AttributeError:
            preddata = preddata.reshape(-1, 1)
    return lr.predict(preddata)




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
    year2011_train, year2011_test, diabetes2011_train, diabetes2011_test = \
        read_test_train_since2011(state, test_size=0.3)
    # Get the linear model for the trained data.
    # Get the linear model for the trained data.
    fulllr = linear_model(year_train, diabetes_train)
    since2011lr = linear_model(year2011_train, diabetes2011_train)

    # Now rejoin to make the full dataset.
    full_year = np.concatenate([year_train, year_test])
    full_diabetes = np.concatenate([diabetes_train, diabetes_test])
    pre2011 = full_year < 2011.0
    post2011 = full_year >= 2011.0

    # Make a new window
    plt.figure()

    # Plot the pre-2011 data in red and the post-2011 data in blue.
    plt.plot(full_year[pre2011], full_diabetes[pre2011], 'rd')
    plt.plot(full_year[post2011], full_diabetes[post2011], 'bd')

    # Now plot the fits.
    res_dates = np.linspace(1994, 2019, 2)
    full_fit = predict_lr(fulllr, res_dates)
    since2011_fit = predict_lr(since2011lr, res_dates)

    plt.plot(res_dates, full_fit, 'r-')
    plt.plot(res_dates, since2011_fit, 'b-')

    # Now make the predictions.
    predyears = np.arange(2017, 2020)
    fullpred = predict_lr(fulllr, predyears)
    since2011pred = predict_lr(since2011lr, predyears)

    # Plot the predictions.
    plt.plot(predyears, fullpred, 'rd')
    plt.plot(predyears, since2011pred, 'bd')

    # Separate the data from the predictions
    yedges = plt.ylim()
    plt.plot([2016.5, 2016.5], list(yedges), 'k--')

    plt.xlabel("Year")
    plt.ylabel("Number of Diabetes Cases")
    plt.title(state)

def compare_mse_distributions():
    '''Compare the mean square error for using full data vs post-2011.'''
    # I want to calculate the Mean Square Error for all states using the full
    # dataset and using just the 2011 data. I make a copy of it by running
    # list.
    allstates = list(FULL_DATA["State"].cat.categories)
    # I found that the "Virgin Islands of the US" has basically no data between
    # 2011 and now. So I'm going to skip that one.
    allstates.remove("Virgin Islands of the U.S.")
    msefull_array = np.zeros(len(allstates))
    msesince2011_array = np.zeros(len(allstates))
    for i, state in enumerate(allstates):
        # Get the testing and training data
        year_train, year_test, diabetes_train, diabetes_test = \
            read_test_train_full(state, test_size=0.3)
        year2011_train, year2011_test, diabetes2011_train, diabetes2011_test = \
            read_test_train_since2011(state, test_size=0.3)
        # Get the linear model for the trained data.
        # Get the linear model for the trained data.
        fulllr = linear_model(year_train, diabetes_train)
        since2011lr = linear_model(year2011_train, diabetes2011_train)

        # Now predict the test data with the model.
        fullpred_test = predict_lr(fulllr, year_test)
        since2011pred_test = predict_lr(since2011lr, year2011_test)

        # Save the values of the MSE in the array.
        full_mse = mean_squared_error(
            fullpred_test, diabetes_test)
        since2011_mse = mean_squared_error(
            since2011pred_test, diabetes2011_test)

        # We need to normalize the MSE before comparing between states. The
        # reason for this is that the MSE for a state with 100000 cases will be
        # a very different scale for a state with only 1000 cases.
        allyear = np.concatenate([year_train, year_test])
        alldiabetes = np.concatenate([diabetes_train, diabetes_test])
        normalized_fullmse = full_mse / (
            max(alldiabetes) - min(alldiabetes))**2
        normalized_since2011mse = since2011_mse / (
            max(alldiabetes[allyear>=2011.0]) -
            min(alldiabetes[allyear>=2011.0]))**2
        msesince2011_array[i] = normalized_fullmse
        msefull_array[i] = normalized_since2011mse

    # Make a new window.
    plt.figure()
    plt.hist([msefull_array, msesince2011_array], color=["red", "blue"],
             label=["Full Data", "Since 2011"], density=False, bins=50)

    # I Calculate the Median MSE for each set instead of the mean because
    # looking at the distribution of MSE shows that there are a lot of really
    # large outliers. So I want something a bit more robust.
    medmse_full = np.median(msefull_array)
    medmse_since2011 = np.median(msesince2011_array)
    plt.text(
        0.35, 30, "Median MSE: {0:.3f}".format(medmse_full), color="red")
    plt.text(
        0.35, 25, "Median MSE: {0:.3f}".format(medmse_since2011), color="blue")
    # Write the medians in the figure.
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Distribution")
    plt.legend(loc="upper right")
