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
#Shape and head of the dataframe
print(wed.shape)
wed.head()

#Dependent Y Variable and Independent X Variable
X = wed.iloc[:,0:2].values
Y = wed.iloc[:,-1].values

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

#Split into Train and test data
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#Model using Scikit learn
from sklearn.linear_model import LinearRegression
lmreg = LinearRegression()

#Fit the model
lmreg.fit(X_Train, Y_Train)
print(lmreg)

#Predict the x test to get the y predict
Y_Pred = lmreg.predict(X_Test)
print(Y_Pred)

#Coefficients
print(lmreg.coef_)

#Intercepts
print(lmreg.intercept_)

#Calculate hte R2
from sklearn.metrics import r2_score as r2
r2(Y_Test, Y_Pred)

#Model using Stats Model
import statsmodels.formula.api as sm
lmregrs = sm.OLS(Y_Train, X_Train).fit()
lmregrs.summary()

#Model using Scikit learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#cannot use rank1 matrix in scikit learn
X = X.reshape((m,1))
#creating model
lmreg = LinearRegression()
#Fitting Training Model
reg = reg.fit(X,Y)
#Y Prediction
Y_Pred = reg.predict(X)

#Calculating R2 Score
r2_score = reg.score(X,Y)
print(R2_score)