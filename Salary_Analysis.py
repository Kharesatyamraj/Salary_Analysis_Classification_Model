# -*- coding: utf-8 -*-
"""
@author: khare
"""
## Name- SATYAM RAJ KHARE


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler 
#Importing data
train = pd.read_csv(r"\SalaryData_Train.csv")
test = pd.read_csv(r"\SalaryData_Test.csv")


# data preprocessing


#traindata set preprocessing
train.head()
train1 = train.iloc[: ,[0,1,2,5,11,13]]# removing unnecessary features
train1.isna().sum()  # no missing values


#traindata set preprocessing
test.head()
test1 = test.iloc[:,[0,1,2,5,11,13]] # removing unnecessary features
test1.isna().sum()  # no missing values


#Converting string data into numerical data

dum1 =pd.get_dummies(train1)#dummy of train data 
dum2 = pd.get_dummies(test1) # dummy of test data


#Normalization 
Standardization = StandardScaler()
train_n = Standardization.fit_transform(dum1)
test_n = Standardization.fit_transform(dum2)


#NaiveBayes 
from sklearn.naive_bayes import BernoulliNB as MB
#from sklearn.naive_bayes import MultinomialNB as MB

classifier_mb = MB() # NaiveBayes object created
classifier_mb.fit(train_n , train["Salary"]) # fit in train data

#classifier applied on test data
test_pred_m = classifier_mb.predict(test_n)

#accuracy test on test-data
accuracy_test_m = np.mean(test_pred_m == test['Salary'])
print(accuracy_test_m )
# Evaluation-Matrix
print(pd.crosstab(test_pred_m, test.Salary))

#classifier applied on train data
train_pred_m = classifier_mb.predict(train_n)

#accuracy test on test-data
accuracy_train_m = np.mean(train_pred_m == train.Salary)
print(accuracy_train_m)

# Evaluation-Matrix
print(pd.crosstab(train_pred_m, train.Salary))
