import pandas as pd
from featureExtraction import featureExtraction

data0 = pd.read_csv("Phishing_Mitre_Dataset_Summer_of_AI.csv", encoding="latin-1")
data0.head()
features = []
label = 0

for i in range(0, 100):
    url = data0["URL"][i]
    modified = url[:-3]
    print(modified)
    features.append(featureExtraction(modified, data0["Label"][i]))

feature_names = [
    "Domain",
    "Have_IP",
    "Have_At",
    "URL_Length",
    "URL_Depth",
    "Redirection",
    "https_Domain",
    "TinyURL",
    "Prefix/Suffix",
    "DNS_Record",
    "Web_Traffic",
    "Domain_Age",
    "Domain_End",
    "iFrame",
    "Mouse_Over",
    "Right_Click",
    "Web_Forwards",
    "Label",
]

featList = pd.DataFrame(features, columns=feature_names)
featList.head()
featList.to_csv("res.csv", index=False)
print(featList)
