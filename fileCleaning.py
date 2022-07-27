import pandas as pd
from featureExtraction import featureExtraction

data0 = pd.read_csv("Phishing_Mitre_Dataset_Summer_of_AI.csv", encoding="latin-1")
data0.head()
features = []
label = 0

for i in range(0, 4799):
    url = data0["URL"][i]
    modified = url[:-3]
    print(modified)
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
print(featList)
