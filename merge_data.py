import pandas as pd

# Load the white and red datasets
white = pd.read_csv("winequality-white.csv")
red = pd.read_csv("winequality-red.csv")

# Merge the data frames in a single file
result = pd.concat([white, red], ignore_index=True)

# Export as a csv file
result.to_csv("winequality.csv", sep=",", index=False)
