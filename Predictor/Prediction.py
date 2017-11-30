import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from Processor import GetData
import warnings
warnings.filterwarnings('ignore')

df_train = GetData.getDataCSV(GetData.train_path)
SalePrice='SalePrice'
GrLivArea='GrLivArea'
TotalBsmtSF='TotalBsmtSF'
OverallQual='OverallQual'
YearBuilt='YearBuilt'
print df_train['SalePrice'].describe()

sns.distplot(df_train['SalePrice'])
#plt.show()

#scatter plot grlivarea/saleprice
data=pd.concat([df_train['SalePrice'],df_train['GrLivArea']],axis=1)
data.plot.scatter(x='GrLivArea',y='SalePrice')


#scatter plot totalbsmtsf/saleprice
data=pd.concat([df_train[SalePrice],df_train[TotalBsmtSF]],axis=1)
data.plot.scatter(x=TotalBsmtSF,y=SalePrice)

#box plot overallqual/saleprice
data=pd.concat([df_train[SalePrice],df_train[OverallQual]],axis=1)
plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=OverallQual,y=SalePrice,data=data)
fig.axis(ymin=0,ymax=800000)

#box plot YearBuilt/saleprice
data=pd.concat([df_train[SalePrice],df_train[YearBuilt]],axis=1)
plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=YearBuilt,y=SalePrice,data=data)
fig.axis(ymin=0,ymax=800000)

#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)

# plt.show()






