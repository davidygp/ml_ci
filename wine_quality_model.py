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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the variables to be within the range of -1 to 1.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Compute the training and test accuracy
training_acc = model.score(X_train, y_train) * 100
test_acc = model.score(X_test, y_test) * 100

# Output the results in a .txt file
with open("results.txt", "w") as f:
    f.write(f"Training accuracy: {training_acc}\n")
    f.write(f"Test accuracy: {test_acc}\n")

# Plot a confusion matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
plot_confusion_matrix(model, X_test, y_test)
plt.savefig("confusion_matrix.png")

# Export the model using pickle
import pickle
file_name = "model.pkl"
open_file = open(file_name, "wb")
pickle.dump([scaler, model], open_file)
open_file.close()
