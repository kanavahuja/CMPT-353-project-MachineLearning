#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.metrics import f1_score


x = sys.argv[1]
df = pd.read_csv(x)


X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33)
# Resampling techniques — Undersample majority class.Source from https://towardsdatascience.com/methods-for-dealing-with-#imbalanced-data-5b761be45a18

X = pd.concat([X_train, y_train], axis=1)
not_fraud = X[X['Class']==0]
fraud = X[X['Class']==1]

not_fraud_downsampled = resample(not_fraud,
                                replace = False, 
                                n_samples = len(fraud), 
                                random_state = 32) 
downsampled = pd.concat([not_fraud_downsampled, fraud])
print("Checking our counts of Fraud and not_fraud after Undersampling majority class")
print(downsampled.Class.value_counts())

y_train = downsampled['Class']
X_train = downsampled.drop('Class', axis=1)

X_train, X_valid, y_train, y_valid  = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

## Bayesian Classifier
model_1 = make_pipeline(
    StandardScaler(),
    GaussianNB()
)
model_1.fit(X_train, y_train)
print("Model Score from Bayesian Classifier:")
print(model_1.score(X_valid, y_valid)) 
y_predicted = model_1.predict(X_valid)
print(classification_report(y_valid, y_predicted))


## KNN Classifier

print("Model Score from KNN Classifier:")
model_2 = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=13)
)
model_2.fit(X_train, y_train)
print(model_2.score(X_valid, y_valid))
y_predicted = model_2.predict(X_valid)
print(classification_report(y_valid, y_predicted))


## Decission Tree Classifier

print("Model Score from Decision Tree Classifier:")
model_3 = make_pipeline(
    StandardScaler(),
    DecisionTreeClassifier(max_depth=6)
)
model_3.fit(X_train, y_train)
print(model_3.score(X_valid, y_valid))
y_predicted = model_3.predict(X_valid)
print(classification_report(y_valid, y_predicted))


## Random Forest Classifier

print("Model Score from Random Forest Classifier:")
model_4 = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100,max_depth=5, min_samples_leaf=10)
)
model_4.fit(X_train, y_train)
print(model_4.score(X_valid, y_valid))
y_predicted = model_4.predict(X_valid)
print(classification_report(y_valid, y_predicted))


## Gradient Boosting Classifier

print("Model Score from Gradient Boosting Classifier:")
model_5 = make_pipeline(
    StandardScaler(),
    GradientBoostingClassifier(n_estimators=100,max_depth=5, min_samples_leaf=0.1)
)
model_5.fit(X_train, y_train)
print(model_5.score(X_valid, y_valid))
y_predicted = model_5.predict(X_valid)
print(classification_report(y_valid, y_predicted))



##Voting Classifier

print("Model score from Voting Classifier:")
model_6 = make_pipeline(
    StandardScaler(),
    VotingClassifier([
    ('nb', GaussianNB()),
    ('svm', SVC(kernel='linear', C=0.1)),
    ('tree1', DecisionTreeClassifier(max_depth=6)),
    ('tree2', DecisionTreeClassifier(min_samples_leaf=10))
])
)
model_6.fit(X_train, y_train)
print(model_6.score(X_valid, y_valid))
y_predicted = model_6.predict(X_valid)
print(classification_report(y_valid, y_predicted))




