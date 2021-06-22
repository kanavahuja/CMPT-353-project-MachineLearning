#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm


# In[2]:


data1 = pd.read_csv("creditcard.csv")


# In[3]:


# Using the creditcard.csv for analysis of original dataset


# In[4]:


authentic = len(data1[data1['Class'] == 0])
fraudulent = len(data1[data1['Class'] == 1])
print("Authentic transactions:" + str(authentic))
print("Fraudulent transactions:" + str(fraudulent))


# In[40]:


a1 = data1['Class']
a2 = data1['Amount']
a3 = data1['Time']
fit=stats.linregress(a3,a2)
fit.slope,fit.intercept


# In[41]:


plt.xticks(rotation=25)
plt.xlabel("Time")
plt.ylabel("Amount")
plt.title("Credit Card")
plt.plot(a3, a2, 'b.' ,alpha=0.5)
plt.plot(a3, a2*fit.slope + fit.intercept, 'r-', linewidth=3)
plt.show()


# In[42]:


fit.pvalue


# In[43]:


residuals = a2 - (fit.slope*a3 + fit.intercept)
plt.hist(residuals)
plt.title("Plot of residuals")


# In[11]:


sns.scatterplot(data=data1, x='Amount', y='Class',hue='Class',palette='hot')


# In[12]:


sns.scatterplot(data=data1, x='Class', y='Amount',hue='Class',palette='hot')


# In[13]:


corrMatrix = data1.corr()
print (corrMatrix)


# In[14]:


sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[15]:


data=data1[['Class','Amount','Time']]
sns.pairplot(data,hue="Class")


# In[16]:


# Using cleaned.csv for analysis of cleaned dataset (without outliers)
df = pd.read_csv("cleaned.csv")


# In[17]:


authentic1 = len(df[df['Class'] == 0])
fraudulent1 = len(df[df['Class'] == 1])
print("Authentic transactions:" + str(authentic1))
print("Fraudulent transactions:" + str(fraudulent1))


# In[18]:


a4 = df['Class']
a5 = df['Amount']
a6 = df['Time']
fit=stats.linregress(a6,a5)
fit.slope,fit.intercept


# In[35]:


plt.xticks(rotation=25)
plt.xlabel("Time")
plt.ylabel("Amount")
plt.title("Credit Card")
plt.plot(a6, a5, 'b.' ,alpha=0.5)
plt.plot(a6, a5*fit.slope + fit.intercept, 'r-', linewidth=3)
plt.show()


# In[20]:


residuals = a5 - (fit.slope*a6 + fit.intercept)
plt.hist(residuals)
plt.title("Plot of residuals")


# In[21]:


sns.scatterplot(data=df, x='Amount', y='Class',hue='Class',palette='hot')


# In[22]:


sns.scatterplot(data=df, x='Class', y='Amount',hue='Class',palette='hot')


# In[23]:


# Performing statistical tests on the cleaned dataset i.e cleaned_edited.csv


# In[24]:


x1=df['Class']
x2=df['Amount']
z1=df['V1']
z2=df['V2']
z3=df['V3']
z4=df['V4']
z5=df['V5']
z6=df['V6']
z7=df['V7']
z8=df['V8']
z9=df['V9']
z10=df['V10']
z11=df['V11']
z12=df['V12']
z13=df['V13']
z14=df['V14']
z15=df['V15']
z16=df['V16']
z17=df['V17']
z18=df['V18']
z19=df['V19']
z20=df['V20']
z21=df['V21']
z22=df['V22']
z23=df['V23']
z24=df['V24']
z25=df['V25']
z26=df['V26']
z27=df['V27']
z28=df['V28']


# In[25]:


# T-test
# First Checking for normality
print(stats.normaltest(x1).pvalue)
print(stats.normaltest(z26).pvalue)


# In[26]:


# Checking Equal variance
print(stats.levene(x1, z1).pvalue)
print(stats.levene(z25, z26).pvalue)


# In[45]:


print(stats.ttest_ind(x1, z1, equal_var=False).pvalue)
print(stats.ttest_ind(x1, z2, equal_var=False).pvalue)
print(stats.ttest_ind(z25, z26, equal_var=False).pvalue)
print(stats.ttest_ind(x1, x2, equal_var=False).pvalue)


# In[28]:


# Anova test
anova = stats.f_oneway(x1,z1, z2, z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,z28)
print(anova)
print(anova.pvalue)


# In[29]:


# Post hoc analysis
x_data = pd.DataFrame({'X1':x1, 'V1':z1 , 'V2': z2, 'V3':z3, 'V4': z4, 'V5':z5, 'V6': z6, 'V7':z7, 'V8':z8, 'V9':z9, 'V10':z10,'V11':z11,'V12':z12, 'V13':z13, 'V14':z14, 'V15':z15, 'V16':z16, 'V17':z17, 'V18':z18,'V19':z19, 'V20':z20,'V21':z21, 'V22':z22, 'V23':z23, 'V24':z24, 'V25':z25, 'V26':z26,'V27':z27,'V28':z28})
x_melt = pd.melt(x_data)
posthoc = pairwise_tukeyhsd(
    x_melt['value'], x_melt['variable'],
    alpha=0.05)


# In[30]:


print(posthoc)


# In[31]:


fig = posthoc.plot_simultaneous()


# In[32]:


# Chi-square
contingency = [[authentic1],[fraudulent1]]
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(p)


# In[33]:


# OLS analysis
xa = df['Class']
xb = df['Amount'] 
xc = sm.add_constant(xa)
data_1 = sm.OLS(xb, xc)
data_2 = data_1.fit()
print(data_2.summary())


# In[ ]:




