'''This module is focused on reading in the CDC data from a directory.

The CDC data will be stored as a series of files, one file per year.'''

import pandas
from pathlib import Path

CDC_FOLDER = Path.home() / "CDCData"

def read_Diabetes_Data_file(datapath=CDC_FOLDER / "Diabetes_State.csv"):
    '''Read in a single file of Diabetes rates.
    
    The file shouuld be taken from the aggregated CDC Diabetes Atlas at:
    
    https://gis.cdc.gov/grasp/diabetes/DiabetesAtlas.html#

    The file should be stored in CDC_FOLDER. This function wil return the
    dataframe holding all of the data.'''
    # Read the CSV.
    diabetes_data = pandas.read_csv(datapath, header=2)

    # The last row of this file is just text. I want to drop it. However, it
    # messes up ALL the datatypes, so we have to reset it.
    diabetes_data.drop(diabetes_data.tail(1).index, inplace=True)

    # This table has missing values, which are denoted by "No Value". This
    # function remaps the "Number" column to be false where the "Number" is a
    # string instead of a number.
    diabetes_data["Number"] = diabetes_data["Number"].where(
        diabetes_data["Number"].str.isnumeric())

    # Converting the state column to be categorical makes it possible to get a
    # list of states from the data to iterate through.
    diabetes_data["State"] = diabetes_data["State"].astype("category")

    # Convert the "Year" and "Number" columns to floats. They should 
    # be integers, but integers can't hold a NaN value in pandas.
    diabetes_data["Year"] = diabetes_data["Year"].astype(float)
    diabetes_data["Number"] = diabetes_data["Number"].astype(float)
    return diabetes_data


