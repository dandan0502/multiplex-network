# -*- coding: utf-8-*-
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

print("Loading Data ... ")

def classifier(data):
	X_data, Y_data = data.iloc[:,:-1], data['flag']
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_data, Y_data,\
	                                         test_size=0.1, random_state=0)
	clf = LogisticRegression()  
	clf.fit(X_train,y_train)
	predict = clf.predict_proba(X_test)
	fpr, tpr, thresholds = metrics.roc_curve(np.array(y_test), np.array(predict)[:,:1])
	print(metrics.auc(fpr, tpr))