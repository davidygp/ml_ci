#%%
# # Import wine dataset
import pandas as pd

wines = pd.read_csv("winequality-red.csv")
wines.columns = wines.columns.str.replace(" ", "_")

# Split dataset into features and target
"""
X = wines.loc[:, ["fixed_acidity", "volatile_acidity", "citric_acid", "alcohol"]]
# KeyError: "Passing list-likes to .loc or [] with any missing labels is no longer supported. The following labels were missing: Index(['fixed_acitidy'], dtype='object'). See https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike"
"""
feature_cols = ["fixed_acidity", "volatile_acidity", "citric_acid", "alcohol"]
X = wines[feature_cols]
y = wines.loc[:, ["quality"]]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the variables to be within the range of -1 to 1.
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=7)
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

#%%
# Plot feature importances
# model_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
plt.bar(x=feature_cols, height=model.feature_importances_)
ax.set_title("Feature importances")
fig.tight_layout()
plt.savefig("feature_importance.png")

#%%
# Plot residuals
fig.clear()
y_pred = model.predict(X_test)
y_test1 = y_test["quality"].tolist()
# residuals = y_test - y_pred
residuals = [y_test1[i] - y_pred[i] for i in range(len(y_test1))]
plt.scatter(residuals, y_pred)
plt.savefig("residuals.png")

#%%
# Export the model using pickle
import pickle

file_name = "model.pkl"
open_file = open(file_name, "wb")
pickle.dump([scaler, model], open_file)
open_file.close()

# %%
