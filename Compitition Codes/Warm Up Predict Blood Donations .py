# I am not sharing the optimization which i had used in the compitition. 
#I am just giving an over view of compitition data and code. 
# Importing necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,RandomizedSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,log_loss
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.svm import SVR,SVC
from sklearn.preprocessing import Imputer,scale,StandardScaler,normalize
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.neural_network import MLPClassifier

#Loading the data from UCI data repository which was used in blood donation compitition
train=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data')
train.columns=['Months since Last Donation','Number of Donations','Total Volume Donated (c.c.)',\
                          'Months since First Donation','Made Donation in March 2007']

test = pd.read_csv("https://s3.amazonaws.com/drivendata/data/2/public/5c9fa979-5a84-45d6-93b9-543d1a0efc41.csv",index_col=0)
# Data processing and cleaning
# Created new variables dif which is log difference
train['dif']=train['Months since First Donation']-train['Months since Last Donation']
train['dif']=train['dif'].replace(0,1)
train['dif']=np.log(train['dif'])
# created per month donation variable
train['perMonth']=train['Months since First Donation']/train['Number of Donations']
test['dif']=test['Months since First Donation']-test['Months since Last Donation']
test['dif']=test['dif'].replace(0,1)
test['dif']=np.log(test['dif'])
test['perMonth']=test['Months since First Donation']/test['Number of Donations']
# created log ratio variable
train['ratio']=train[['Months since First Donation','Months since Last Donation']].\
apply(lambda x:x['Months since First Donation']/1 if x['Months since Last Donation']==0 \
      else x['Months since First Donation']/x['Months since Last Donation'],axis=1)
train['ratio']=np.log(train['ratio'])
test['ratio']=test[['Months since First Donation','Months since Last Donation']].\
apply(lambda x:x['Months since First Donation']/1 if x['Months since Last Donation']==0 \
      else x['Months since First Donation']/x['Months since Last Donation'],axis=1)
test['ratio']=np.log(test['ratio'])
# Also generated volume variable 
train['volume']=np.log(train['perMonth']*train['Months since First Donation']/train['Total Volume Donated (c.c.)'])
test['volume']=np.log(test['perMonth']*test['Months since First Donation']/test['Total Volume Donated (c.c.)'])

# Split the data set in train and test and also sclaed the data set
X_train=train.drop(['Made Donation in March 2007'],axis=1).values
y_train=train['Made Donation in March 2007'].values
X_test=test.values
scale=StandardScaler()
scale.fit(X_train)
X_train=scale.transform(X_train)
X_test=scale.transform(X_test)
# fitted NN classifier and get the log loss value
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train,y_train)
log_loss(y_train,clf.predict_proba(X_train))
# Some further eveluations
confusion_matrix(y_train,clf.predict(X_train))
clf.score(X_train,y_train)
