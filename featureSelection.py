import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

df=pd.read_csv("res.csv")
df.head()

df.describe()

df.head()
cols = [ 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                      'https_Domain', 'TinyURL', 'Prefix/Suffix',  
                      'Domain_Age', 'Domain_End','Update_Age', 'Exact Length', 'Zero Count', 'Zero Prop', 'Period Count', 'Period Prop', 'Special Count', 'Special Prop', 'Label']




from sklearn.preprocessing import StandardScaler 
stdsc = StandardScaler() 
X_std = stdsc.fit_transform(df[cols].iloc[:,range(0,18)].values)
cov_mat =np.cov(X_std.T)
plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
hm = sns.heatmap(cov_mat,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 12},
                 cmap='coolwarm',                 
                 yticklabels=cols,
                 xticklabels=cols)
plt.title('Covariance matrix showing correlation coefficients', size =18)
plt.tight_layout()
plt.show(block = True)