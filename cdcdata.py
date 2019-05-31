'''This module is focused on reading in the CDC data from a directory.

The CDC data will be stored as a series of files, one file per year.'''

import pandas
from pathlib import Path

CDC_FOLDER = Path.home() / "CDCData"

def read_Diabetes_Data_file(datapath=CDC_FOLDER / "Diabetes_State.csv"):
    '''Read in a single file of Diabetes rates.
    
    The file should be stored in CDC_FOLDER. This file should be taken from the
    aggregated CDC data map.'''
    diabetes_data = pandas.read_csv(datapath, header=2)
    diabetes_data.drop(diabetes_data.tail(1).index, inplace=True)
    diabetes_data["State"] = diabetes_data["State"].astype("category")
    diabetes_data["Number"] = diabetes_data["Number"].where(
        diabetes_data["Number"].str.isnumeric())
    return diabetes_data


