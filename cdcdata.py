'''This module is focused on reading in the CDC data from a directory.

The CDC data will be stored as a series of files, one file per year.'''

import pandas as pd
from pathlib import Path

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

if __name__ == "__main__":
    tidy_table()

