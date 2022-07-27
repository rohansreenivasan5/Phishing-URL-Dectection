# importing basic packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data0 = pd.read_csv("res.csv")
data0.head()
df_train = data0.sample(frac=0.8, random_state=1)
df_test = data0.drop(df_train.index)
feature_names = [
    "URL_Depth",  # normalize
    "Domain_Age",  # n
    "Domain_End",  # n
    "Update_Age",  # n
    "Exact Length",  # n
    "Zero Count",  # n
    "Zero Prop",  # n
    "Period Prop",  # n
    "Special Count",  # n
    "Special Prop",  # n
    "Slash Count",  # n
    "Slash Prop",  # n
    "Check Words",  # n
]
for column in feature_names:
    mean = df_train[column].mean()
    standard_dev = df_train[column].std()
    df_train[column] = (df_train[column] - df_train[column].mean()) / df_train[
        column
    ].std()
    df_test[column] = (df_test[column] - mean) / standard_dev

data0.describe()
# Dropping the Domain column
data = data0
# data = data.sample(frac=1).reset_index(drop=True)
data.head()
y = data["Label"]

y_train = df_train["Label"]
y_test = df_test["Label"]

X_train = df_train.drop("Label", axis=1)
X_test = df_test.drop("Label", axis=1)

from sklearn.metrics import accuracy_score

# Creating holders to store the model performance results
ML_Model = []
acc_train = []
acc_test = []

# function to call for storing the results
def storeResults(model, a, b):
    ML_Model.append(model)
    acc_train.append(round(a, 3))
    acc_test.append(round(b, 3))

    # Decision Tree model


from sklearn.tree import DecisionTreeClassifier

# instantiate the model
tree = DecisionTreeClassifier(max_depth=5)
# fit the model
tree.fit(X_train, y_train)
# predicting the target value from the model for the samples
y_test_tree = tree.predict(X_test)
y_train_tree = tree.predict(X_train)

# computing the accuracy of the model performance
acc_train_tree = accuracy_score(y_train, y_train_tree)
acc_test_tree = accuracy_score(y_test, y_test_tree)

print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))

# storing the results. The below mentioned order of parameter passing is important.
# Caution: Execute only once to avoid duplications.
storeResults("Decision Tree", acc_train_tree, acc_test_tree)
# Random Forest model
from sklearn.ensemble import RandomForestClassifier

# instantiate the model
forest = RandomForestClassifier(max_depth=5)

# fit the model
forest.fit(X_train, y_train)

# predicting the target value from the model for the samples
y_test_forest = forest.predict(X_test)
y_train_forest = forest.predict(X_train)

# computing the accuracy of the model performance
acc_train_forest = accuracy_score(y_train, y_train_forest)
acc_test_forest = accuracy_score(y_test, y_test_forest)

print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))

# storing the results. The below mentioned order of parameter passing is important.
# Caution: Execute only once to avoid duplications.
storeResults("Random Forest", acc_train_forest, acc_test_forest)

# Multilayer Perceptrons model
from sklearn.neural_network import MLPClassifier

# instantiate the model
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100, 100, 100]))

# fit the model
mlp.fit(X_train, y_train)

# predicting the target value from the model for the samples
y_test_mlp = mlp.predict(X_test)
y_train_mlp = mlp.predict(X_train)

# computing the accuracy of the model performance
acc_train_mlp = accuracy_score(y_train, y_train_mlp)
acc_test_mlp = accuracy_score(y_test, y_test_mlp)

print("Multilayer Perceptrons: Accuracy on training Data: {:.3f}".format(acc_train_mlp))
print("Multilayer Perceptrons: Accuracy on test Data: {:.3f}".format(acc_test_mlp))

# storing the results. The below mentioned order of parameter passing is important.
# Caution: Execute only once to avoid duplications.
storeResults("Multilayer Perceptrons", acc_train_mlp, acc_test_mlp)

# XGBoost Classification model
# from xgboost import XGBClassifier

# # instantiate the model
# xgb = XGBClassifier(learning_rate=0.4, max_depth=7)
# # fit the model
# xgb.fit(X_train, y_train)

# # predicting the target value from the model for the samples
# y_test_xgb = xgb.predict(X_test)
# y_train_xgb = xgb.predict(X_train)

# # computing the accuracy of the model performance
# acc_train_xgb = accuracy_score(y_train, y_train_xgb)
# acc_test_xgb = accuracy_score(y_test, y_test_xgb)


# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test, y_test_xgb)


# print("XGBoost: Accuracy on training Data: {:.3f}".format(acc_train_xgb))
# print("XGBoost : Accuracy on test Data: {:.3f}".format(acc_test_xgb))

# # storing the results. The below mentioned order of parameter passing is important.
# # Caution: Execute only once to avoid duplications.
# storeResults("XGBoost", acc_train_xgb, acc_test_xgb)


# Support vector machine model
from sklearn.svm import SVC

# instantiate the model
svm = SVC(kernel="linear", C=1.0, random_state=12)
# fit the model
svm.fit(X_train, y_train)

# predicting the target value from the model for the samples
y_test_svm = svm.predict(X_test)
y_train_svm = svm.predict(X_train)


# computing the accuracy of the model performance
acc_train_svm = accuracy_score(y_train, y_train_svm)
acc_test_svm = accuracy_score(y_test, y_test_svm)

print("SVM: Accuracy on training Data: {:.3f}".format(acc_train_svm))
print("SVM : Accuracy on test Data: {:.3f}".format(acc_test_svm))

# storing the results. The below mentioned order of parameter passing is important.
# Caution: Execute only once to avoid duplications.
storeResults("SVM", acc_train_svm, acc_test_svm)

# XGBoost Classification model
from xgboost import XGBClassifier

# instantiate the model
xgb = XGBClassifier(learning_rate=0.4, max_depth=7)
# fit the model
xgb.fit(X_train, y_train)

# predicting the target value from the model for the samples
y_test_xgb = xgb.predict(X_test)
y_train_xgb = xgb.predict(X_train)

# computing the accuracy of the model performance
acc_train_xgb = accuracy_score(y_train, y_train_xgb)
acc_test_xgb = accuracy_score(y_test, y_test_xgb)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_test_xgb)


print("XGBoost: Accuracy on training Data: {:.3f}".format(acc_train_xgb))
print("XGBoost : Accuracy on test Data: {:.3f}".format(acc_test_xgb))

# storing the results. The below mentioned order of parameter passing is important.
# Caution: Execute only once to avoid duplications.
storeResults("XGBoost", acc_train_xgb, acc_test_xgb)

# creating dataframe
results = pd.DataFrame(
    {"ML Model": ML_Model, "Train Accuracy": acc_train, "Test Accuracy": acc_test}
)
results

# Sorting the datafram on accuracy
results.sort_values(by=["Test Accuracy", "Train Accuracy"], ascending=False)
