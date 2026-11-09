'''This module is focused on reading in the CDC data from a directory.

The CDC data will be stored as a series of files, one file per year.'''

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

# NOTE: I included the data in the Git repository, which is generally a no-no.
# Please don't make a habit of putting large datasets in the Git repository!
CDC_FOLDER = Path.cwd() / "data"

CATEGORIES = [
    "18-44", "45-64", "65-74", "75+",
    "Asian", "Black", "White", "Hispanic",
    "Female", "Male",
    "LessHS", "HS", "MoreHS",
    "Total",
]
NUMBER_FORMATS = ["Number", "Percentage"]
MISSING_VALUES = ["Suppressed", "No Data"]

# Map each file type's source columns -> NumberType labels in the tidy table
COLUMN_MAP = {
    "Number": {"Number": "Number"},
    "Percentage": {
        "Percentage": "Percentage",
        "Lower Limit": "LowerLim",
        "Upper Limit": "UpperLim",
    },
}

def _filename(cat: str, numformat: str) -> str:
    return f"DiabetesAtlas_AllStates_{cat}_{numformat}.csv"

def check_input_files(datadir=CDC_FOLDER) -> list:
    """Check that every {cat} x {numformat} file exists in datadir.

    Returns a list of missing file paths (empty if everything is present)
    and prints a summary.
    """
    datadir = Path(datadir)
    missing = [
        datadir / _filename(cat, fmt)
        for cat in CATEGORIES
        for fmt in NUMBER_FORMATS
        if not (datadir / _filename(cat, fmt)).is_file()
    ]

    total = len(CATEGORIES) * len(NUMBER_FORMATS)
    if missing:
        print(f"Missing {len(missing)} of {total} expected files in {datadir}:")
        for p in missing:
            print(f"  - {p.name}")
    else:
        print(f"All {total} expected files found in {datadir}.")
    return missing


def _read_one(path: Path, numformat: str, cat: str) -> pd.DataFrame:
    """Read one Diabetes Atlas CSV and return it in tidy (long) form."""
    df = pd.read_csv(
        path,
        skiprows=2,                 # header is on line 3 (1-indexed)
        na_values=MISSING_VALUES,   # "Suppressed" / "No Data" -> NaN
        thousands=",",              # in case counts look like "1,234"
        skipinitialspace=True,
    )
    df.columns = df.columns.str.strip()  # e.g. " Upper Limit" -> "Upper Limit"

    colmap = COLUMN_MAP[numformat]
    expected = ["Year", "State", *colmap]
    absent = [c for c in expected if c not in df.columns]
    if absent:
        raise ValueError(
            f"{path.name}: expected columns {expected}, "
            f"missing {absent}; found {list(df.columns)}"
        )

    df = df[expected].rename(columns=colmap)

    tidy = df.melt(
        id_vars=["Year", "State"],
        value_vars=list(colmap.values()),
        var_name="NumberType",
        value_name="Value",
    )
    tidy["Value"] = pd.to_numeric(tidy["Value"], errors="coerce")
    tidy["Category"] = cat
    return tidy


def tidy_table(inputpath=CDC_FOLDER, output="Diabetes_State.csv") -> pd.DataFrame:
    """Combine all Diabetes Atlas files into one tidy CSV.

    Output columns: Year, State, Value, NumberType, Category
    Saved to inputpath / output. Also returns the DataFrame.
    """
    inputpath = Path(inputpath)

    missing = check_input_files(inputpath)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} input file(s) missing from {inputpath}; "
            "see list printed above."
        )

    frames = [
        _read_one(inputpath / _filename(cat, fmt), fmt, cat)
        for cat in CATEGORIES
        for fmt in NUMBER_FORMATS
    ]
    tidy = pd.concat(frames, ignore_index=True)

    # Defensive: drop any non-data rows (e.g., footnotes) lacking a numeric year
    tidy["Year"] = pd.to_numeric(tidy["Year"], errors="coerce")
    tidy = tidy.dropna(subset=["Year"])
    tidy["Year"] = tidy["Year"].astype(int)

    tidy = tidy[["Year", "State", "Value", "NumberType", "Category"]]
    tidy.to_csv(inputpath / output, index=False)
    return tidy


def read_Diabetes_Data_file(datapath=CDC_FOLDER / "Diabetes_State.csv"):
    '''Read in a single file of Diabetes rates.
    
    The file shouuld be taken from the aggregated CDC Diabetes Atlas at:
    
    https://gis.cdc.gov/grasp/diabetes/DiabetesAtlas.html#

    The file should be stored in CDC_FOLDER. This function wil return the
    dataframe holding all of the data.'''
    # Read the CSV.
    df = pd.read_csv(
            datapath, dtype={
                "State": "category", 
                "NumberType": "category",
                "Category": "category",
            },
        na_values=[""],          # empty cell -> NaN
        keep_default_na=False,   # don't treat strings like "NA" or "None" as missing
    )

    # Value is float64 so NaN is representable; coerce in case of stray text
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Year should never be missing, so an int dtype is safe. This raises
    # an error if it is missing, which is what we want.
    df["Year"] = df["Year"].astype(int)

    return df


    # Convert the "Year" and "Number" columns to floats. They should 
    # be integers, but integers can't hold a NaN value in pandas.
    #diabetes_data["Year"] = diabetes_data["Year"].astype(float)
    #diabetes_data["Number"] = diabetes_data["Number"].astype(float)
    #return diabetes_data

def missing_data_table(fulldata):
    '''Calculate fraction of missing data for CATEGORIES and STATE.'''

    # Empty frame: one row per state, one column per category
    missing_fracs = pd.DataFrame(index=STATES, columns=CATEGORIES, dtype=float)

    # Iterate over the outer product of STATES x CATEGORIES
    for state, category in product(STATES, CATEGORIES):
        numbers = fulldata[fulldata["NumberType"] == "Number"]
        matching_years = numbers[np.logical_and(
            numbers["State"] == state, numbers["Category"] == category)]
        missing_years = np.count_nonzero(
                np.isnan(matching_years["Value"]))
        value = missing_years / len(matching_years)
        missing_fracs.at[state, category] = value
        print(state)
        print(category)
        print(missing_fracs.at[state, category]) 
    return missing_fracs


def check_na_consistency(fulldata):
    """Check that every (Category, State, Year) combination appears in all
    NumberType subsets and that "Value" is NA in all of them or in none.

    Returns a DataFrame of problem rows (empty if everything is consistent).
    """
    keys = ["Category", "State", "Year"]
    types = ["Number", "Percentage", "UpperLim", "LowerLim"]
    merged = None

    for numtype in types:
        subset = fulldata[fulldata["NumberType"] == numtype]

        # Fail early with a readable message if the key isn't unique
        dupes = subset[subset.duplicated(keys, keep=False)]
        if not dupes.empty:
            raise ValueError(
                f"{len(dupes)} rows of type {numtype!r} share the same "
                f"{keys}; another column probably belongs in the key.\n"
                f"{dupes.sort_values(keys).head(10)}"
            )

        # Keep keys + Value, and add a flag so we can tell a missing row
        # apart from a row whose Value is NA
        part = (
            subset[keys + ["Value"]]
            .rename(columns={"Value": f"Value_{numtype}"})
            .assign(**{f"in_{numtype}": True})
        )

        if merged is None:
            merged = part
        else:
            merged = merged.merge(
                part, on=keys, how="outer", validate="one_to_one"
            )

    in_cols = [f"in_{t}" for t in types]
    value_cols = [f"Value_{t}" for t in types]

    # After an outer merge, flags for absent rows are NaN, and NaN == True is False
    present = merged[in_cols].eq(True)
    all_present = present.all(axis=1)

    # NA consistency only makes sense for rows present in all subsets
    # (absent rows get NaN from the merge itself)
    na = merged[value_cols].isna()
    na_inconsistent = all_present & na.any(axis=1) & ~na.all(axis=1)

    problems = merged[~all_present | na_inconsistent].copy()

    # Describe each problem
    def describe(idx):
        missing = [t for t in types if not present.at[idx, f"in_{t}"]]
        if missing:
            return "missing in: " + ", ".join(missing)
        return "inconsistent NA"

    problems["problem"] = [describe(i) for i in problems.index]

    n_missing = int((~all_present).sum())
    n_na = int(na_inconsistent.sum())
    print(f"{len(merged)} unique {keys} combinations checked")
    print(f"  missing from at least one subset: {n_missing}")
    print(f"  inconsistent NA values:           {n_na}")

    if problems.empty:
        print("OK: all subsets are consistent.")
    return problems.sort_values(keys).drop(columns=in_cols)

if __name__ == "__main__":
    tidy_table()

    cdc = read_Diabetes_Data_file()

    # The columns are Year, State, Value, NumberType, Category.
    # There are 25 years, 54 states, 4 numbertypes and 14 categories.
    expected_rows = 25*54*4*14
    print(f"We expect {expected_rows} rows. "
            f"The actual number of rows is {len(cdc)}.")

    STATES = cdc["State"].unique()

    # Now want to check for missing data.
    # Let's go through each category and State to find how many values are
    # missing.
    missing_fracs = missing_data_table(cdc)


    # Print the full table with no truncation
    print (missing_fracs.to_string())

    # Print the summary row of each category and of each state.
    print(missing_fracs.mean(axis=0))
    print(missing_fracs.mean(axis=1))

    # Let's drop the race categories and then recompute
    cdc_norace = cdc[~cdc["Category"].isin(
        ["Asian", "Black", "White", "Hispanic"])]

    missing_norace = missing_data_table(cdc_norace)

    print (missing_norace.to_string())
    print(missing_norace.mean(axis=1))

    cdc_bestdata = cdc_norace[~cdc_norace["State"].isin(
        ["Guam", "Virgin Islands of the U.S."])]

    # Cool, so now I've verified that the race data is really incomplete
    # for most states. Once that data is removed, all states except alaska
    # are >95% complete. Alaksa is around 7%, which I will consider
    # acceptable.

    # Let's double-check that all number types are equivalently missing.
    number_bestdata = cdc_bestdata[
            cdc_bestdata["NumberType"] == "Number"]
    percentage_bestdata = cdc_bestdata[
            cdc_bestdata["NumberType"] == "Percentage"]
    upperlim_bestdata = cdc_bestdata[
            cdc_bestdata["NumberType"] == "UpperLim"]
    lowerlim_bestdata = cdc_bestdata[
            cdc_bestdata["NumberType"] == "LowerLim"]

    numbervpercentage = (
            np.isnan(number_bestdata["Value"]) == 
            np.isnan(percentage_bestdata["Value"]))
    print(number_bestdata[["Year", "State"]][numbervpercentage])

    # Next let's check for gaps
