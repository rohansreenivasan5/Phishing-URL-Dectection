# importing required packages for this section
from urllib.parse import urlparse, urlencode
import ipaddress
import re

# maybe remove if takes too long
# 1.Domain of the URL (Domain)
def getDomain(url):
    domain = urlparse(url).netloc
    if re.match(r"^www.", domain):
        domain = domain.replace("www.", "")

    return domain


# redundent
# 2.Checks for IP address in URL (Have_IP)
def havingIP(url):
    try:
        ipaddress.ip_address(url)
        ip = 1
    except:
        ip = 0
    return ip


# 3.Checks the presence of @ in URL (Have_At)
def haveAtSign(url):
    if "@" in url:
        at = 1
    else:
        at = 0
    return at


# 4.Finding the length of URL and categorizing (URL_Length)
def getLength(url):
    if len(url) < 54:
        length = 0
    else:
        length = 1
    return length


# 5.Gives number of '/' in URL (URL_Depth)
def getDepth(url):
    s = urlparse(url).path.split("/")
    depth = 0
    for j in range(len(s)):
        if len(s[j]) != 0:
            depth = depth + 1
    return depth


# 6.Checking for redirection '//' in the url (Redirection)
def redirection(url):
    pos = url.rfind("//")
    if pos > 6:
        if pos > 7:
            return 1
        else:
            return 0
    else:
        return 0


# 7.Existence of "HTTPS" Token in the Domain Part of the URL (https_Domain)
def httpDomain(url):
    domain = urlparse(url).netloc
    if "https" in domain:
        return 1
    else:
        return 0


# listing shortening services


strs = [
    "bit",
    ".ly",
    "goo",
    ".gl",
    "shorte",
    ".st",
    "go2l",
    ".ink",
    "x",
    ".co",
    "ow",
    ".ly",
    "t",
    ".co",
    "tinyurl",
    "tr",
    ".im",
    "is",
    ".gd",
    "cli",
    ".gs",
    "yfrog",
    ".com",
    "migre",
    ".me",
    "ff",
    ".im",
    "tiny",
    ".cc",
    "url4",
    ".eu",
    "twit",
    ".ac",
    "su",
    ".pr",
    "twurl",
    ".nl",
    "snipurl",
    ".com",
    "short",
    ".to",
    "BudURL",
    ".com",
    "ping",
    ".fm",
    "post",
    ".ly",
    "Just",
    ".as",
    "bkite",
    ".com",
    "snipr",
    ".com",
    "fic",
    ".kr",
    "loopt",
    ".us",
    "doiop",
    ".com",
    "short",
    ".ie",
    "kl",
    ".am",
    "wp",
    ".me",
    "rubyurl",
    ".com",
    "om",
    ".ly",
    "to",
    ".ly",
    "bit",
    ".do",
    "t",
    ".co",
    "lnkd",
    ".in",
    "db",
    ".tt",
    "qr",
    ".ae",
    "adf",
    ".ly",
    "goo",
    ".gl",
    "bitly",
    ".com",
    "cur",
    ".lv",
    "tinyurl",
    ".com",
    "ow",
    ".ly",
    "bit",
    ".ly",
    "ity",
    ".im",
    "q",
    ".gs",
    "is",
    ".gd",
    "po",
    ".st",
    "bc",
    ".vc",
    "twitthis",
    ".com",
    "u",
    ".to",
    "j",
    ".mp",
    "buzurl",
    ".com",
    "cutt",
    ".us",
    "u",
    ".bb",
    "yourls",
    ".org",
    "x",
    ".co",
    "prettylinkpro",
    ".com",
    "scrnch",
    ".me",
    "filoops",
    ".info",
    "vzturl",
    ".com",
    "qr",
    ".net",
    ".com",
    "tweez",
    ".me",
    "v",
    ".gd",
    "tr",
    ".im",
    "link",
    ".zip",
    ".net",
]


# 9.Checking for Prefix or Suffix Separated by (-) in the Domain (Prefix/Suffix)
def prefixSuffix(url):
    if "-" in urlparse(url).netloc:
        return 1  # phishing
    else:
        return 0  # legitimate


# importing required packages for this section
import re
from bs4 import BeautifulSoup
import whois
import urllib
import urllib.request
from datetime import datetime

# 12.Web traffic (Web_Traffic)
def web_traffic(url):
    try:
        # Filling the whitespaces in the URL if any
        url = urllib.parse.quote(url)
        rank = BeautifulSoup(
            urllib.request.urlopen(
                "http://data.alexa.com/data?cli=10&dat=s&url=" + url
            ).read(),
            "xml",
        ).find("REACH")["RANK"]
        rank = int(rank)
    except Exception as e:
        return 1
    if rank < 100000:
        return 1
    else:
        return 0


# REMOVE
# 14.End time of domain: The difference between termination time and current time (Domain_End)


# importing required packages for this section
import requests

# 15. IFrame Redirection (iFrame)
def iframe(response):
    if response == "":
        return 1
    else:
        if re.findall(r"[<iframe>|<frameBorder>]", response.text):
            return 0
        else:
            return 1


# remove
# 16.Checks the effect of mouse over on status bar (Mouse_Over)
def mouseOver(response):
    if response == "":
        return 1
    else:
        if re.findall("<script>.+onmouseover.+</script>", response.text):
            return 1
        else:
            return 0


# 17.Checks the status of the right click attribute (Right_Click)
def rightClick(response):
    if response == "":
        return 1
    else:
        if re.findall(r"event.button ?== ?2", response.text):
            return 0
        else:
            return 1


# 18.Checks the number of forwardings (Web_Forwards)
def forwarding(response):
    if response == "":
        return 1
    else:
        if len(response.history) <= 2:
            return 0
        else:
            return 1


def create_age_check(num):
    if num <= 12:
        return 1

    else:
        return 0


def expiry_age_check(num):
    if num <= 3:
        return 1

    else:
        return 0


def update_age_check(num):
    if num >= 1000:
        return 1

    else:
        return 0


def exactLength(url):
    length = len(url)
    return length


def zerocount(url):
    num = url.count("0")
    return num

    # 5.11 0 proportion


def zeroprop(url):
    num = url.count("0")
    length = len(url)
    prop = num / length
    return prop


def slashcount(url):
    num = url.count("/")
    return num

    # 5.11 0 proportion


def slashprop(url):
    num = url.count("/")
    length = len(url)
    prop = num / length
    return prop


def percentcount(url):
    num = url.count("%")
    return num

    # 5.11 0 proportion


def percentprop(url):
    num = url.count("%")
    length = len(url)
    prop = num / length
    return prop


def lessthan(url):
    num = url.count(">")
    return num

    # 5.11 0 proportion


def greaterthan(url):
    num = url.count("<")
    return num


# 5.2 count periods
def periodcount(url):
    num = url.count(".")
    return num


# 5.11 0 proportion
def periodprop(url):
    num = url.count(".")
    length = len(url)
    prop = num / length
    return prop


# 5.2 count all special characters according to https://owasp.org/www-community/password-special-characters
def specialcount(url):
    num = 0
    num = num + url.count(" ")
    num = num + url.count("!")
    num = num + url.count('"')
    num = num + url.count("#")
    num = num + url.count("$")
    num = num + url.count("%")
    num = num + url.count("&")
    num = num + url.count("'")
    num = num + url.count("(")
    num = num + url.count(")")
    num = num + url.count("*")
    num = num + url.count("+")
    num = num + url.count(",")
    num = num + url.count("-")
    num = num + url.count(".")
    num = num + url.count("/")
    num = num + url.count(":")
    num = num + url.count(";")
    num = num + url.count("<")
    num = num + url.count("=")
    num = num + url.count(">")
    num = num + url.count("?")
    num = num + url.count("@")
    num = num + url.count("[")
    num = num + url.count("]")
    num = num + url.count("^")
    num = num + url.count("_")
    num = num + url.count("`")
    num = num + url.count("{")
    num = num + url.count("|")
    num = num + url.count("}")
    num = num + url.count("~")
    return num


# prop special
def specialprop(url):
    num = 0
    num = num + url.count(" ")
    num = num + url.count("!")
    num = num + url.count('"')
    num = num + url.count("#")
    num = num + url.count("$")
    num = num + url.count("%")
    num = num + url.count("&")
    num = num + url.count("'")
    num = num + url.count("(")
    num = num + url.count(")")
    num = num + url.count("*")
    num = num + url.count("+")
    num = num + url.count(",")
    num = num + url.count("-")
    num = num + url.count(".")
    num = num + url.count("/")
    num = num + url.count(":")
    num = num + url.count(";")
    num = num + url.count("<")
    num = num + url.count("=")
    num = num + url.count(">")
    num = num + url.count("?")
    num = num + url.count("@")
    num = num + url.count("[")
    num = num + url.count("]")
    num = num + url.count("^")
    num = num + url.count("_")
    num = num + url.count("`")
    num = num + url.count("{")
    num = num + url.count("|")
    num = num + url.count("}")
    num = num + url.count("~")
    length = len(url)
    prop = num / length
    return prop


def check_words(url):
    num = 0
    for str in strs:
        num = num + url.count(str)
    return num


# Function to extract features
def featureExtraction(url, label, create_age, expiry_age, update_age):

    features = []
    features.append(getLength(url))
    features.append(getDepth(url))
    features.append(create_age_check(create_age))
    features.append(expiry_age_check(expiry_age))
    features.append(update_age_check(update_age))
    features.append(exactLength(url))
    features.append(zerocount(url))
    features.append(zeroprop(url))
    features.append(periodprop(url))
    features.append(specialcount(url))
    features.append(specialprop(url))
    features.append(slashcount(url))
    features.append(slashprop(url))
    features.append(check_words(url))
    features.append(label)

    return features

print("--Loaded Feature Extraction--")

import pandas as pd

data0 = pd.read_csv("Phishing_Mitre_Dataset_Summer_of_AI.csv", encoding="latin-1")
data0.head()
features = []
label = 0

for i in range(0, 4799):
    url = data0["URL"][i]
    modified = url[:-3]
    #print(modified)
    features.append(
        featureExtraction(
            modified,
            data0["Label"][i],
            data0["create_age(months)"][i],
            data0["expiry_age(months)"][i],
            data0["update_age(days)"][i],
        )
    )

feature_names = [
    "URL_Length",
    "URL_Depth", #normalize
    "Domain_Age",#n
    "Domain_End",#n
    "Update_Age",#n
    "Exact Length",#n
    "Zero Count",#n
    "Zero Prop",#n
    "Period Prop",#n
    "Special Count",#n
    "Special Prop",#n
    "Slash Count",#n
    "Slash Prop",#n
    "Check Words",#n
    "Label",
]

featList = pd.DataFrame(features, columns=feature_names)
featList.head()
featList.to_csv("res.csv", index=False)
#print(featList)

data1 = pd.read_csv("Summer_of_AI_Test_Students.csv", encoding="latin-1")
data1 = data1.iloc[: , 1:]
data1.head()
#print(data1)
features = []
label = 0

for i in range(0, 1200):
    url = data1["URL"][i]
    modified = url[:-3]
    #print(modified)
    features.append(
        featureExtraction(
            modified,
            0,
            data1["create_age(months)"][i],
            data1["expiry_age(months)"][i],
            data1["update_age(days)"][i],
        )
    )

feature_names = [
    "URL_Length",
    "URL_Depth", #normalize
    "Domain_Age",#n
    "Domain_End",#n
    "Update_Age",#n
    "Exact Length",#n
    "Zero Count",#n
    "Zero Prop",#n
    "Period Prop",#n
    "Special Count",#n
    "Special Prop",#n
    "Slash Count",#n
    "Slash Prop",#n
    "Check Words",#n
    "Label",
]

featList = pd.DataFrame(features, columns=feature_names)
featList.head()
featList.to_csv("pred.csv", index=False)
#print(featList)

# importing basic packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data0 = pd.read_csv("res.csv")
data0.head()
df_train = data0.sample(frac=0.8, random_state=1)
df_test=data0.drop(df_train.index)
feature_names = [
    "URL_Depth", #normalize
    "Domain_Age",#n
    "Domain_End",#n
    "Update_Age",#n
    "Exact Length",#n
    "Zero Count",#n
    "Zero Prop",#n
    "Period Prop",#n
    "Special Count",#n
    "Special Prop",#n
    "Slash Count",#n
    "Slash Prop",#n
    "Check Words",#n
]
for column in feature_names:
    mean= df_train[column].mean()
    standard_dev=df_train[column].std()
    df_train[column] = (df_train[column] - df_train[column].mean()) / df_train[column].std()
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
from sklearn.metrics import f1_score

# Creating holders to store the model performance results
ML_Model = []
acc_train = []
acc_test = []

# function to call for storing the results
def storeResults(model, a, b):
    ML_Model.append(model)
    acc_train.append(round(a, 3))
    acc_test.append(round(b, 3))

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
f1_train_mlp = f1_score(y_train, y_train_mlp)
f1_test_mlp = f1_score(y_test, y_test_mlp)

print("Multilayer Perceptrons: Accuracy on training Data: {:.3f}".format(acc_train_mlp))
print("Multilayer Perceptrons: Accuracy on test Data: {:.3f}".format(acc_test_mlp))
print("Multilayer Perceptrons: F1 on training Data: {:.3f}".format(f1_train_mlp))
print("Multilayer Perceptrons: F1 on test Data: {:.3f}".format(f1_test_mlp))

#data1["Label"] = mlp.predict()