import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from Processor import GetData
import warnings
import datetime
from scipy.stats import skew

warnings.filterwarnings('ignore')


def data_preprocess(train, test):
    outlier_idx = [4, 11, 13, 20, 46, 66, 70, 167, 178, 185, 199, 224, 261, 309, 313, 318, 349, 412, 423, 440, 454, 477,
                   478, 523, 540, 581, 588, 595, 654, 688, 691, 774, 798, 875, 898, 926, 970, 987, 1027, 1109, 1169,
                   1182, 1239, 1256, 1298, 1324, 1353, 1359, 1405, 1442, 1447]
    train.drop(train.index[outlier_idx], inplace=True)

    all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                          test.loc[:, 'MSSubClass':'SaleCondition']))

    print all_data.head

    to_delete = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
    all_data = all_data.drop(to_delete, axis=1)

    train["SalePrice"] = np.log1p(train["SalePrice"])
    # log transform skewed numeric features
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice

    return X_train, X_test, y



def create_submission(prediction,score):
    now = datetime.datetime.now()
    sub_file = 'submission_'+str(score)+'_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv'
    print ('Creating submission: ', sub_file)
    pd.DataFrame({'Id': test['Id'].values, 'SalePrice': prediction}).to_csv(sub_file, index=False)

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

#missing data

total = df_train.isnull().sum().sort_values(ascending=False)
percentage = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missingData = pd.concat([total,percentage],axis=1,keys=["Total","Perc"])
#print missingData.head(20)

df_train = df_train.drop((missingData[missingData['Total']>1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
#print df_train.isnull().sum().max()

#Cleaning up

#Scale

scaled_sales_price=StandardScaler().fit_transform(df_train[SalePrice][:,np.newaxis])
low_range = scaled_sales_price[scaled_sales_price[:,0].argsort()][:10]
high_range = scaled_sales_price[scaled_sales_price[:,0].argsort()][-10:]

#deleting points
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

#Normality
df_train['SalePrice'] = np.log(df_train['SalePrice'])
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
hasBasement="hasBasement"
df_train[hasBasement] = pd.Series(len(df_train[TotalBsmtSF]),index=df_train.index)
df_train[hasBasement] = 0
df_train.loc[df_train[TotalBsmtSF]>0,hasBasement]=1

df_train.loc[df_train[hasBasement]==1,TotalBsmtSF] = np.log(df_train[TotalBsmtSF])

#Dummies
df_train=pd.get_dummies(df_train)



#Prediction

from sklearn import linear_model
from sklearn.model_selection import  train_test_split
df_train=pd.read_csv(GetData.train_path)
df_test=pd.read_csv(GetData.test_path)
test=df_test
X_train, X_test, y = data_preprocess(df_train,df_test)


clf = linear_model.LinearRegression()
clf.fit(X_train,y)


#CSV


submission=pd.DataFrame()






prediction= np.exp(clf.predict(X_test))

create_submission(prediction,"")










