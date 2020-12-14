import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
##Step 1 - Import all the library functions along with the dataset/s

df = pd.read_csv("H:\\DATA SCIENCE\Modules\\Modal 7 Multipal linear Regression\\50_Startups.csv\\50_Startups.csv")
##Changing the names of the column of the main dataframe
df.columns = "RandD_Spend", "Admin", "Market_exp", "State", "Profit"

#since the state is in categorical ,we are converting it into numerical(0,1,2)
df['State'] = df['State'].astype('category')
df['State'] = df['State'].cat.codes
df['State']

#Y=Profit
#X1=RandD
#X2=Admin
#X3=Marketing_Spend

#Profit = b0 + b1RandD + b2Admin + b3Marketing_Spend + E

##Using Seaborn for pair plotting
sns.pairplot(df)

##Using Seaborn for heatmap
df.columns
sns.heatmap(df.corr(),linewidth = 0.2, vmax=1.0, square=True, linecolor='red',annot=True)
## here we have combined correlartion function with heatmap so as to observe the 
df.corr() # gives the same value as above

## building a model to check the significance of the variables
ml1 = smf.ols('Profit~RandD_Spend+Admin+State+Market_exp',data=df).fit()
ml1.summary()

#P-Values for Admin and Market_exp are more than 0.05 and hence statistically insignificant

import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
plt.show()

#As observed from influence index plot obs no 48 and 49 have are influencers we need to drop the same and check.
df_new=df.drop(df.index[[46,49]], axis = 0)


##Preparing the model after removing the outliers
ml2 = smf.ols('Profit~RandD_Spend+Admin+Market_exp',data=df_new).fit()
ml2.summary()

df_new = df.drop(["Admin"], axis=1, inplace=True)
## Lets check for the below model taking only RandD and Marketing into account
ml3 = smf.ols('Profit~RandD_Spend+Market_exp',data=df_new).fit()
ml3.summary()

 ## After removing 45th, 46th , 48th and 49th obs we found that Market_exp has a significance impact on the o/p,
## but Admin has a p value of 0.217 which is way higher than 0.05, hence the same can be omitted
# Hence going ahead with only RandD and Market_exp with the final model 
###
# Added variable plot to check the partial regression of features
sm.graphics.plot_partregress_grid(ml2)
sm.graphics.plot_partregress_grid(ml3)

''' Now we will be splitting the data into test and train and pass on the values for validation '''

features = df_new.iloc[:,0:-1].values
labels = df_new.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.2,random_state=0)

''' Creating the linear Regression model'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Step 7 - Interpreting the Coefficient and the Intercept
y_pred = regressor.predict(X_test)
y_pred

#Step 8 - Interpreting the Coefficient and the Intercept

print(regressor.coef_)
print(regressor.intercept_)

#Step 9 - Predict the Score (% Accuracy)
print('Train Score :', regressor.score(X_train,y_train))
print('Test Score:', regressor.score(X_test,y_test))

#Step 11- Calculate the MSE and RMSE
from sklearn import metrics
print('MSE :', metrics.mean_squared_error(y_test,y_pred))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

##### Testing the o/p
X1 = [[56466, 0, 366168,0]]
out1 = regressor.predict(X1)
print(out1)
