# importing basic packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data0 = pd.read_csv("res.csv")
data1 = pd.read_csv("Phishing_Mitre_Dataset_Summer_of_AI.csv", encoding="latin-1")
data0.head()

data0.describe()
# Dropping the Domain column
data = data0
# data = data.sample(frac=1).reset_index(drop=True)
data.head()
y = data["Label"]
X = data.drop("Label", axis=1)
X.shape, y.shape
# Splitting the dataset into train and test sets: 80-20 split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)
X_train.shape, X_test.shape

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

df = pd.DataFrame()
df["actual"] = y_train
df["predicted"] = y_train_xgb
incorrect = df[df["actual"] != df["predicted"]]

print("XGBoost: Accuracy on training Data: {:.3f}".format(acc_train_xgb))
print("XGBoost : Accuracy on test Data: {:.3f}".format(acc_test_xgb))

# storing the results. The below mentioned order of parameter passing is important.
# Caution: Execute only once to avoid duplications.
storeResults("XGBoost", acc_train_xgb, acc_test_xgb)

print(cm)

print(incorrect)
