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

def read_test_train_full(state, test_size=0.3):
    '''Get a testing and training set only since 2011.
    
    This function reads the year and diabetes number for all FULL_DATA records
    starting with the year 2011. This year marked the first main revision of 
    the BFRSS dataset.'''
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
    plt.figure(figsize=(11, 8))

    # Plot the pre-2011 data in red and the post-2011 data in blue.
    # I want the figure to be in thousands, so I will be dividing the cases by
    # 1000 in these plots.
    plt.plot(full_year[pre2011], full_diabetes[pre2011]/1e3, 'rd')
    plt.plot(full_year[post2011], full_diabetes[post2011]/1e3, 'bd')

    # Now plot the fits.
    res_dates = np.linspace(1994, 2019, 2)
    full_fit = predict_lr(fulllr, res_dates)
    since2011_fit = predict_lr(since2011lr, res_dates)

    plt.plot(res_dates, full_fit/1e3, 'r-')
    plt.plot(res_dates, since2011_fit/1e3, 'b-')

    # Now make the predictions.
    predyears = np.arange(2017, 2020)
    fullpred = predict_lr(fulllr, predyears)
    since2011pred = predict_lr(since2011lr, predyears)

    # Plot the predictions.
    plt.plot(predyears, fullpred/1e3, 'rd')
    plt.plot(predyears, since2011pred/1e3, 'bd')

    # Separate the data from the predictions
    yedges = plt.ylim()
    plt.plot([2016.5, 2016.5], yedges, 'k--')

    plt.xlabel("Year")
    plt.ylabel("Number of Diabetes Cases (Thousands)")
    plt.title(state)
    plt.ylim(yedges)


def graph_residual(state):
    '''Graph the residual after regressing for a given state.
    
    I think this may be informative to determine if there's apparent structure
    beyond random noise.'''
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
    plt.figure(figsize=(11, 8))

    # Calculate the predicted values for all of the years.
    # Predict yields a 2-dimensional array, so I want to reduce it back to 1
    # dimension.
    fullmodel_diabetes = predict_lr(fulllr, full_year)[:,0]
    since2011_diabetes = predict_lr(since2011lr, full_year[post2011])[:,0]

    # Calculate the residuals.
    full_residual = fullmodel_diabetes - full_diabetes
    since2011_residual = since2011_diabetes - full_diabetes[post2011]
    print(since2011_diabetes.shape)

    # Plot the residuals
    plt.plot(full_year, full_residual/1e3, 'rd')
    plt.plot(full_year[post2011], since2011_residual/1e3, 'bd')

    # Plot a zero-line
    xvals = plt.xlim()
    plt.plot(xvals, [0, 0], 'k--')
    plt.xlim(xvals)

    plt.xlabel("Year")
    plt.ylabel("Residual (Thousands)")

def compare_mse_distributions():
    '''Compare the mean square error for using full data vs post-2011.
    
    This will output a graph that shows a histogram of the MSE with the full
    data and with the post-2011 data for all states. In order to reliably
    compare MSEs for state, the data is normalized to lie between the maximum
    and minimum value of the dataset for that state.'''
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
    plt.figure(figsize=(11, 8))
    plt.hist([msefull_array, msesince2011_array], color=["red", "blue"],
             label=["Full Data", "Since 2011"], density=False, bins=50)

    # I Calculate the Median MSE for each set instead of the mean because
    # looking at the distribution of MSE shows that there are a lot of really
    # large outliers. So I want something a bit more robust.
    medmse_full = np.median(msefull_array)
    medmse_since2011 = np.median(msesince2011_array)
    # Write the medians in the figure.
    plt.text(
        0.35, 30, "Median MSE: {0:.3f}".format(medmse_full), color="red")
    plt.text(
        0.35, 25, "Median MSE: {0:.3f}".format(medmse_since2011), color="blue")
    plt.xlabel("Normalized Mean Squared Error")
    plt.ylabel("Distribution")
    plt.legend(loc="upper right")
    plt.title("{0} Residual".format(state))

def write_prediction_table(
        fulloutput="./predictions/Full_Predictions.csv",
        shortoutput="./predictions/2011_Predictions.csv"):
    '''Write a table of predictions.

    Takes the predictions from the linear regression model and writes them to a
    csv.'''
    # I want to store the prediction values for all states. Make a list copy of
    # te States to avoid some weird problem.
    allstates = list(FULL_DATA["State"].cat.categories)
    # The Virgin Islands have really spotty data. So I am going to omit the
    # territories in general. They can be added back in.
    allstates.remove("Virgin Islands of the U.S.")
    allstates.remove("Puerto Rico")
    allstates.remove("Guam")
    allstates.remove("District of Columbia")
    # These are the arrays that will hold the predicted years. Each state will
    # be a row. Each column will be a year.
    full_array = np.zeros((len(allstates), 3))
    since2011_array = np.zeros((len(allstates), 3))
    # The years that we are predicting.
    predyears = np.arange(2017, 2020)
    for i, state in enumerate(allstates):

        year_train, year_test, diabetes_train, diabetes_test = \
            read_test_train_full(state, test_size=0.3)
        year2011_train, year2011_test, diabetes2011_train, diabetes2011_test = \
            read_test_train_since2011(state, test_size=0.3)
        # Get the linear model for the trained data.
        fulllr = linear_model(year_train, diabetes_train)
        since2011lr = linear_model(year2011_train, diabetes2011_train)

        # Now make the predictions.
        fullpred = predict_lr(fulllr, predyears)
        since2011pred = predict_lr(since2011lr, predyears)

        # The way fullpred is done. I just need to take the transpose and it
        # will align with the arrays.
        full_array[i,:] = fullpred.T
        since2011_array[i,:] = since2011pred.T

    # Place the data into a dataframe.
    # The output should be an integer. I am adding 0.5 so it rounds correctly
    # rather than just taking the floor.
    fullframe = pandas.DataFrame(
        full_array+0.5, columns=predyears.astype("str"), dtype="int")
    since2011frame = pandas.DataFrame(
        since2011_array+0.5, columns=predyears.astype("str"), dtype="int")
    # Add the State information into the dataframe.
    fullframe.insert(0, "State", allstates)
    since2011frame.insert(0, "State", allstates)

    # Now write the files
    fullframe.to_csv(fulloutput)
    since2011frame.to_csv(shortoutput)


if __name__ == "__main__":

    # Plot the data and prediction for a particular state.
    # In many cases this looks reasonable. The two lines fit well to the data.
    # However, there are some cases where the data looks more like a timeseries
    # than a linear model with noise. Take a look at Wyoming. Wyoming seems
    # like a more appropriate case for a timeseries analysis.
    # This image is stored in "Ohio_Diabetes.png".
    graph_state("Ohio")
    plt.show()

    # Compare the Mean Squared Error calculated for the test samples when the
    # full sample is used vs when only the first 5 years are used. This plot
    # clearly shows that the MSE is uch lower when you only use the most recent
    # data, which may support that the data pre-2011 is substantially different
    # from the data post-2011. However, I also noticed that since we only have
    # 5 data points for 2011+, the test sample (30%) will only be 1 or 2 points.
    # So I'm not sure whether that actually implies that the fit is better. 
    # The image corresponding to this command is "MSE_DIST.png".
    compare_mse_distributions()
    plt.show()

    # Finally, plot the residual between the model and the data. I think these
    # look pretty reasonable by eye for Ohio (but not Wyoming). 
    # The image corresponding to this command is "Ohio_Residuals.png.
    graph_residual("Ohio")
    plt.show()

    # Now write the predictions. The predictions will be in the form of a csv
    # file. The headers will be "State", "2017", "2018", and "2019",
    # corresponding to the State, and the predictions for each coming year.
    # The predictions using the full dataset is in "Full_Predictions.csv".
    # The predictions using just the 2011+ dataset is in
    # "2011_Predictions.csv".
    write_prediction_table()
