#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 23:00:56 2022

@author: agastya
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")

dfx = pd.read_csv("file:///Users/agastya/Coding/M.L_DSE%20_317/Assignment%201/training_data.csv", header=None)
dfx

dfy = pd.read_csv("file:///Users/agastya/Coding/M.L_DSE%20_317/Assignment%201/training_data_class_labels.csv", header=None)
dfy

#Plotting the data
plt.figure(figsize=(8, 6))
ax = sns.scatterplot(data=dfx, x=0, y=1, hue=dfy[0])
ax.set(xlabel="Feature 0", ylabel="Feature 1", title="Scatter plot of data")
ax.legend(title="Class Labels")
plt.show()

#Splitting the data
x_train, x_test, y_train, y_test = train_test_split(dfx.values, dfy.values, test_size=0.2, random_state=5)
x_train

# Navie Bayes.

# Gaussian Navie Bayes

from sklearn.naive_bayes import GaussianNB, BernoulliNB
modelGNB = GaussianNB()
modelGNB.fit(x_train, y_train.ravel())

predGNB = modelGNB.predict(x_test)
print("Gaussian Navie Bayes Report")
print(classification_report(y_test, predGNB))

# Bernoulli Navie Bayes

modelBNB = BernoulliNB()
modelBNB.fit(x_train, y_train.ravel())

predBNB = modelBNB.predict(x_test)
print("Bernoulli Navie Bayes Report")
print(classification_report(y_test, predBNB))

# Logistic Regression

from sklearn.linear_model import LogisticRegression

modelLog = LogisticRegression()
log_parameters = {
    "C": np.logspace(-2, -1, 30),
    "penalty": ["l1", "l2", "elasticnet"],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
}
clfLog = GridSearchCV(
    modelLog, log_parameters, cv=10, verbose=0, n_jobs=11, scoring="f1_macro"
)
clfLog.fit(x_train, y_train.ravel())

clfLog.best_params_

predLog = clfLog.predict(x_test)
print("Logistic regression refort")
print(classification_report(y_test, predLog))

# Standerd Vector Machine

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
svm1 = svm.SVC(kernel='linear', C = 0.01)
svm1.fit(x_test,y_test)
SVC(C=0.01, kernel='linear')
y_train_pred = svm1.predict(x_train)
y_test_pred = svm1.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_test_pred)
print("Standered Vector Machine Report")
print(classification_report(y_test,y_test_pred))

# KNN

from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier()
params = {
    "n_neighbors": np.arange(2, 10, 1),
    "metric": [
        "cityblock",
        "cosine",
        "euclidean",
        "haversine",
        "l1",
        "l2",
        "manhattan",
        "nan_euclidean",
    ],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
}
clfKNN = GridSearchCV(modelKNN, params, cv=10, n_jobs=11)
clfKNN.fit(x_train, y_train.ravel())

clfKNN.best_params_

predKNN = clfKNN.predict(x_test)
print("KNN Report")
print(classification_report(y_test, predKNN))

from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test, predKNN)

## Final test data Pridiction

Final = pd.read_csv("file:///Users/agastya/Coding/M.L_DSE%20_317/Assignment%201/test_data.csv", header=None)
Final

# Using KNN as it has highest accuracy and f1 score among the classifiers.

predictions = clfKNN.predict(Final)
predictions

# Saving the predicted values as text file

np.savetxt("predicted_test_class_lables.txt", predictions, delimiter="\n")


