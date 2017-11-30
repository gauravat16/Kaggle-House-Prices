from Processor import GetData
import  matplotlib.pyplot as plt
import pandas as pd
#GetData
data = GetData.getDataCSV(GetData.train_path)
# print data.head()
prices=data['SalePrice']
MSclass=GetData.downsize(data['MSSubClass'],1000)
LotArea=GetData.downsize(data['LotArea'],1000)




#SubClass
plt.bar(MSclass,prices)
plt.ylabel("prices")
plt.xlabel("MSSubClass")
# plt.show()

#LotArea
plt.bar(LotArea,prices)
plt.ylabel("prices")
plt.xlabel("LotArea")
plt.show()