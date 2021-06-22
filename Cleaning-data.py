#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler,MinMaxScaler

x = sys.argv[1]
o_file = sys.argv[2]
df = pd.read_csv(x)
print('The dataset has {0} rows and {1} columns.'.format(df.shape[0], df.shape[1]))
df.info()  ## for checking if there is any null value in the dataset or not. 

df.describe()
class_names = {0:'Not Fraud', 1:'Fraud'}
print(df.Class.value_counts().rename(index = class_names))

## We can see that Time feature is irrelevant in the context of model prediction so remove it. 
df = df.drop('Time', axis=1)

X = df.drop('Class', axis=1)

y = df['Class']

## Apart from the Amount column all column vaues are in certain range so lets modify the Amount column. 
sc = StandardScaler()
df['amount_new'] = sc.fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Amount'], axis=1)

## We also know that our data is not balanced since there are so less fraud cases in comparision to the Non-fraud cases so we first need to make our dataset Balanced. Balancing the dataset is done in fraud_not_fraud.py

df.to_csv(o_file)



