 ############ Computer DATA #################
###  Predict sales of the computer ###################

import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# importing the data set  
Data = pd.read_csv('H:\\DATA SCIENCE\\Modules\\Module 7 Multipal linear Regression\\Computer_Data.csv\\Computer_Data.csv')
Data.describe() 
Data.head(11)
print(Data.columns)
sns.pairplot(Data.iloc[ :, :])
Data.corr()

####  Create Dummy variable ##### cd, multi, premium
Comp = pd.get_dummies(Data, columns=['cd','multi', 'premium'])

Comp.head(11)
Comp.corr()
print(Comp.columns)
# Rgression model

model = smf.ols('price ~ speed + hd+ram+screen+ cd_no+ cd_yes + multi_no+multi_yes + premium_no + premium_yes + ads + trend ', data= Comp).fit()
model.summary() #### summary  R^sqr = 0.776, AIC = 08.810^04

prediction =  model.predict()
prediction

modelspeed = smf.ols('price ~ speed', data= Comp).fit()
modelspeed.summary() ## P-value is <0.05 so speed is statistically significant

modelhd = smf.ols('price ~ hd', data=Comp).fit()
modelhd.summary() ## p-value is < 0.05 so speed is statistically significant

modelram = smf.ols('price ~ram', data = Comp).fit()
modelram.summary()

modelscreen = smf.ols('price ~ screen', data= Comp).fit()
modelscreen.summary()

modelcdn = smf.ols('price ~ cd_no', data = Comp).fit()
modelcdn.summary()

modelcdy = smf.ols('price ~ cd_yes', data = Comp).fit()
modelcdy.summary()

modelmultin = smf.ols('price ~ multi_no', data= Comp).fit()
modelmultin.summary()

modelmultiy = smf.ols('price ~ multi_yes', data= Comp).fit()
modelmultiy.summary()

modelpremiumn = smf.ols('price ~ premium_no', data = Comp).fit()
modelpremiumn.summary()

modelpremiumy = smf.ols('price ~ premium_yes', data = Comp).fit()
modelpremiumy.summary()

modelads = smf.ols('price ~ ads', data= Comp).fit()
modelads.summary()

modeltrend = smf.ols('price ~ trend', data = Comp). fit()

modeltrend.summary()

###### all the P values are significant

#Checking whether data has any influential values
#influence index plots

import statsmodels.formula.api as smf
sm.graphics.influence_plot(model)

#Added Variable Plot
sm.graphics.plot_partregress_grid(model)
#Added Variable plot for weight is not showing any significance
#Weight is the most straightest line which means it has very less influence in MPG

# final model
finalmodel = smf.ols('price ~ speed + hd+ram+screen+ cd_yes + multi_yes + premium_no + premium_yes + ads + trend ', data= Comp).fit()
finalmodel.summary() #### summary  R^sqr = 0.776

price_pred=finalmodel.predict(Comp[['speed','hd','ram','screen','cd_yes','multi_yes','premium_no','premium_yes','ads','trend']])
price_pred

er=Comp.price-price_pred
er
plt.boxplot(er)

sqerr=er*er
mse=np.mean(sqerr)
rmse=np.sqrt(mse)  ### type 1
rmse  #### rmse=  275.129

rmsee=np.sqrt(np.mean(er*er))  #### type 2
rmsee 

#Checking our LINE Assumptions

resid=pd.DataFrame(pd.Series(Comp['price']-price_pred))
resid
resid.mean()
resid.describe()
resid.rename(columns={"price":"Residuals"},inplace=True)


plt.scatter(x=price_pred, y=resid, color="red")
#Residuals Vs Fitted Values
#Errors are scattered
#ie., errors are independent

stdres=pd.DataFrame(finalmodel.resid_pearson) 
#standardized residual calculations
#Pearson Residual 

plt.scatter(x=price_pred, y=stdres, color="green")


#Plot between Actual Values of MPG and fitted values
plt.scatter(x=price_pred, y=Comp["price"], color="green")



### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
Comp_train,Comp_test  = train_test_split(Comp,test_size = 0.3) # 30% test data

# preparing the model on train data 
model_train = smf.ols("price ~ speed + hd+ram+screen+ cd_yes+ multi_yes+ premium_no + premium_yes+ ads + trend",data=Comp_train).fit()
model_train.summary() ###0.771
# train_data prediction
train_pred = model_train.predict(Comp_train)

# train residual values 
train_resid  = train_pred - Comp_train.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) # 279.771 
train_rmse
# prediction on test data set 
test_pred = model_train.predict(Comp_test)

# test residual values 
test_resid  = test_pred - Comp_test.price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) # 264.251
test_rmse

