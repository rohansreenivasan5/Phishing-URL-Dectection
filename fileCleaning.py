import pandas as pd
from featureExtraction import featureExtraction

data0 = pd.read_csv("Phishing_Mitre_Dataset_Summer_of_AI.csv", encoding="latin-1")
data0.head()
features = []
label = 0

for i in range(0, 350):
    url = data0["URL"][i]
    modified = url[:-3]
    print(modified)
    features.append(featureExtraction(modified, data0["Label"][i],data0["create_age(months)"][i],data0["expiry_age(months)"][i],data0["update_age(days)"][i]))
    
feature_names = [ 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                      'https_Domain', 'TinyURL', 'Prefix/Suffix',  
                      'Domain_Age', 'Domain_End','Update_Age', 'Exact Length', 'Zero Count', 'Zero Prop', 'Period Count', 'Period Prop', 'Special Count', 'Special Prop', 'Label']

featList = pd.DataFrame(features, columns=feature_names)
featList.head()
featList.to_csv("res.csv", index=False)
print(featList)
