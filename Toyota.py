# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:19:34 2020

@author: Deepak
"""
############### Toyota############
########## Consider only the below columns and prepare a prediction model for predicting Price.

import pandas as pd 
import numpy as np 
#mfrom sklearn.lineraRegression
import seaborn as sns
import statsmodels.formula.api as smf

## Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]
Df = pd.read_csv("H:\\DATA SCIENCE\\Modules\\Module 7 Multipal linear Regression\\ToyotaCorolla.csv\\ToyotaCorolla.csv",encoding='latin1')
Df.describe()
Df.head(11)
print(Df.columns)
sns.pairplot(Df.iloc[ :,:])
#Droping the columns
Df1 = Df.drop(['Model','Mfg_Month','Mfg_Year','Fuel_Type','Met_Color','Color','Automatic'], axis=1)
Df2 = Df1.drop(['Cylinders','Mfr_Guarantee','BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2','Airco'], axis=1)
Df3 = Df2.drop(['Automatic_airco', 'Boardcomputer', 'CD_Player','Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio','Mistlamps'],axis=1)
Df0 = Df3.drop (['Id','Sport_Model', 'Backseat_Divider', 'Metallic_Rim','Radio_cassette', 'Tow_Bar'], axis=1)
Df0

y =Df0 ['Price']
X = Df0[['Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears','Quarterly_Tax', 'Weight']]
y 
X    
Df0.corr()  # Corelation
# preparing model consadring all the variable
model = smf.ols('y ~ X', data= Df0).fit()
model.summary()

prediction = model.predict()
prediction

# Calculating VIF value of indepandant variable

Vif_Age = smf.ols('Age_08_04 ~ KM + HP+ cc+ Doors+ Gears + Quarterly_Tax + Weight', data= Df0).fit().rsquared
VIF_age = 1/ (1-Vif_Age)

Vif_KM = smf.ols(' KM~ Age_08_04 + HP+ cc+ Doors+ Gears + Quarterly_Tax + Weight', data= Df0).fit().rsquared
VIF_km = 1/ (1-Vif_KM)

Vif_HP = smf.ols('HP~ Age_08_04 +  KM + cc+ Doors+ Gears + Quarterly_Tax + Weight', data= Df0).fit().rsquared
VIF_HP = 1/ (1-Vif_HP)

Vif_cc = smf.ols('cc ~ Age_08_04 + KM + HP+ Doors+ Gears + Quarterly_Tax + Weight', data= Df0).fit().rsquared
VIF_cc = 1/ (1-Vif_cc)

Vif_Door= smf.ols('Doors ~Age_08_04 +KM + HP+ cc+ Gears + Quarterly_Tax + Weight', data= Df0).fit().rsquared
VIF_Door = 1/ (1-Vif_Door)

Vif_Gear = smf.ols('Gears ~ Age_08_04 + KM + HP+ cc+ Doors+ Quarterly_Tax + Weight', data= Df0).fit().rsquared
VIF_Gear = 1/ (1-Vif_Gear)

Vif_Tax = smf.ols('Quarterly_Tax ~ Age_08_04 + KM + HP+ cc+ Doors+ Gears + Weight', data= Df0).fit().rsquared
VIF_Tax = 1/ (1-Vif_Tax)

Vif_Wei = smf.ols('Weight ~ Age_08_04 + KM + HP+ cc+ Doors+ Gears + Quarterly_Tax', data= Df0).fit().rsquared
VIF_wei = 1/ (1-Vif_Wei)

# Storing VIf values in data frames

D1 ={'Variables' :['Age_08_04',' KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'], 'VIF':[VIF_age,VIF_km, VIF_HP, VIF_cc,VIF_Door, VIF_Gear,VIF_Tax, VIF_wei]}
Vif_frame = pd.DataFrame(D1)
Vif_frame

# Weight and Quarterly_Tax  has high VIF value so Remove it 
model2 = smf.ols('Price ~ Age_08_04 + KM+HP+cc+Doors+Gears+Quarterly_Tax', data=Df0).fit()
predict = model2.predict()
model2.summary() #r^2 =0.840

## Spliting the data into train and test 
from sklearn.model_selection import train_test_split
Df_train, Df_test = train_test_split(Df0, test_size=0.3)

# Preparing the model on train data
train_model=smf.ols('Price ~Age_08_04 + KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data= Df_train).fit() 
train_model.summary() # R^2 = 0.870
# Predict the train data
train_predict = train_model.predict(Df_train)
train_predict
# train residual values
train_resi = train_predict - Df_train.Price
train_resi
#RMSE value for train data
train_rmse = np.sqrt(np.mean(train_resi*train_resi))
train_rmse # =1304.5493

# test model
test_model=smf.ols('Price ~Age_08_04 + KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data= Df_test).fit()
test_model.summary() #R^2 = 0.853

# preparing the model on test data
test_model=smf.ols('Price ~Age_08_04 + KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data= Df_test).fit() 
test_model.summary()
# predit the test model
test_predict= test_model.predict(Df_test)
test_predict
# test residual values
test_resi = test_predict - Df_test.Price
test_resi

# RMSE value for test data
test_rmse = np.sqrt(np.mean(test_resi*test_resi))
test_rmse # = 1397.6655







