# Import wine dataset
import pandas as pd

wines = pd.read_csv("winequality-red.csv")
wines.columns = wines.columns.str.replace(" ", "_")

# Split dataset into features and target
"""
X = wines.loc[:, ["fixed_acidity", "volatile_acidity", "citric_acid", "alcohol"]]
# KeyError: "Passing list-likes to .loc or [] with any missing labels is no longer supported. The following labels were missing: Index(['fixed_acitidy'], dtype='object'). See https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike"
"""
X = wines[["fixed_acidity", "volatile_acidity", "citric_acid", "alcohol"]]
y = wines.loc[:, ["quality"]]

# Scale the variables to be within the range of -1 to 1.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(X)
X = scaler.transform(X)

# Train a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X,y)

# Print the model score
model.score(X, y)

# Export the model using pickle
import pickle
file_name = "model.pkl"
open_file = open(file_name, "wb")
pickle.dump([scaler, model], open_file)
open_file.close()
