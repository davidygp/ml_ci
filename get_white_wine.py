# Get the White Wine dataset
from urllib.request import urlretrieve
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

urlretrieve(url, "winequality-white-raw.csv")

# Load the white wien dataset as a dataframe
import pandas as pd
white_wine = pd.read_csv("winequality-white-raw.csv", sep=";")

# Export as a csv file
white_wine.to_csv("winequality-white.csv", sep=",", index=False)
