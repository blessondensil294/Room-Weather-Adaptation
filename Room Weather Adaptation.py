# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 20:12:21 2018

@author: BLAZIN
"""
#importing the packages
import pandas as pd
import numpy as np
#Data visualiztion
import matplotlib.pyplot as plt
import seaborn as sns

#loading the data
wed = pd.read_csv("E:\\Python\\Resume Projects\\Room weather adaptation\\Data\\Samples\\weatherHistoryBasic.csv")

#Present Working Directory
pwd

#Shape and head of the dataframe
print(wed.shape)
wed.head()

#Info of the dataframe
wed.info()

#Describe the dataframe (SUmmary)
wed.describe()

#To get the column names
wed.columns



#EDA

#To find the covariance of the variables
np.cov()

#To find the corelation of the variance
np.corrcoef()

#Pairplot of the data
sns.pairplot(wed)

#Distribution Plot
sns.distplot(wed['Apparent'])

#Correlation of the data
sns.heatmap(wed.corr())

#multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF = variance_inflation_factor()

transformer = ReduceVIF()


#Scatter Plot
sns.pairplot(wed, kind='reg')

#Encoding Categorical Data from string to continous categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lblenc = LabelEncoder()
X[] = lblenc.fit_transform(X[])
onehot = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoid variable dummy trap - encoder creates 2 columsn and to remove the columns
X=X[:,1:]

#Dependent Y Variable and Independent X Variable
X = wed[['Hour','Apparent']]
Y = wed['Temperature']

#Split into Train and test data
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#Model using Scikit learn
from sklearn.linear_model import LinearRegression
lmreg = LinearRegression()

#Fit the model
lmreg.fit(X_Train, Y_Train)
print(lmreg)

#Model Evaluation

#Coefficients
print(lmreg.coef_)
coeff_df = pd.DataFrame(lmreg.coef_,X.columns,columns=['Coefficient'])
coeff_df

#Intercepts
print(lmreg.intercept_)

#Predict the x test to get the y predict
Y_Pred = lmreg.predict(X_Test)
Y_Pred
print(Y_Pred)

#Calculate hte R2
from sklearn.metrics import r2_score as r2
r2(Y_Test, Y_Pred)

#Scatter plot of Prediction and test
plt.scatter(Y_Pred, Y_Test)

#Residual Histogram
sns.distplot((Y_Test - Y_Pred), bins = 50)

#Regression Evaluation Metrics

#Mean Absolute Error (MAE) (Average) is the mean of the absolute value of the errors: MAE is the easiest to understand, because it's the average error.
#Mean Squared Error (MSE) is the mean of the squared errors: MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
#Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors: RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.

#All of these are loss functions, because we want to minimize them.

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(Y_Test, Y_Pred))
print('MSE:', metrics.mean_squared_error(Y_Test, Y_Pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_Test, Y_Pred)))


#-------------------------------------------------------------------------------------------------
#Model using Stats Model
import statsmodels.formula.api as sm
lmregrs = sm.OLS(Y_Train, X_Train).fit()
lmregrs.summary()

#Error Calculation of the training model 
pred_train = lmregrs.predict(X_Train)
err_train = pred_train - Y_Train

#Error Calculation for the Test model
pred_test = lmregrs.predict(X_Test)
err_test = pred_test - Y_Test

#Plot the model for actual and predicted
plt.scatter(Y_Train, pred_train)
plt.xlabel('Y Train Data')
plt.ylabel('Predicted')
plt.title('Main')

#Plot the residuals and predicted value
plt.scatter(pred_train, err_train, c="b", s=40, alpha = 0.5)
plt.scatter(pred_test, err_test, c="g", s=40)
plt.hlines(y=0, xmin=0, xmax=40)
plt.title("Residuals Plot Train - Blue  Test - Green")
plt.ylabel("Residuals")
Text(0,0.5,"Residuals")
