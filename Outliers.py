#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

data = pd.read_csv('creditcard.csv')

not_fraud = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]


count_f = fraud['Time'].count()
count_nf = not_fraud['Time'].count()
print("Number of Fraudulent cases = " + str(count_f))
print("Number of Non-Fraudulent cases = " + str(count_nf))

prob =  count_f/count_nf


prob_sample = 0
while(abs(prob-prob_sample) > 0.00005):
    data_sample = data.sample(n = 100000)
    not_fraud_sample = data_sample[data_sample['Class'] == 0]
    fraud_sample= data_sample[data_sample['Class'] == 1]

    count_fs = fraud_sample['Time'].count()
    count_nfs = not_fraud_sample['Time'].count()
    print("Number of Fraudulent sample cases = " + str(count_fs))
    print("Number of Non-Fraudulent sample cases = " + str      (count_nfs))
    
    prob_sample = count_fs/count_nfs

data_sample.to_csv("sample.csv")
sample = pd.read_csv("sample.csv")

## Adapted from: https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
iso = IsolationForest(contamination=0.1)
ret2 = iso.fit_predict(sample)

sample_copy1 = sample.copy()
sample_copy1 = sample_copy1.drop("Unnamed: 0", axis = 1)

sample_copy1['outlier'] = ret2
sample_copy1 = sample_copy1[sample_copy1['outlier'] != -1]

sample_copy1.to_csv("output_edited.csv")






