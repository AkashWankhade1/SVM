#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC


# In[2]:


df_train = pd.read_csv('C:/Users/lenovo/Downloads/mobile_train.csv')
df_test = pd.read_csv('C:/Users/lenovo/Downloads/mobile_test.csv')


# In[3]:


df_train.head()


# In[4]:


df_test.head()


# In[5]:


df_train.shape


# In[6]:


df_test.shape


# In[7]:


corr = df_train.corr()
fig = plt.figure(figsize =(15,12))
r = sns.heatmap(corr, cmap='Purples')
r.set_title("Correlation ")


# In[8]:


corr.sort_values(by=["price_range"],ascending=False)


# In[9]:


df_train.describe()


# In[10]:


sns.countplot(df_train['price_range'])
plt.show()


# In[11]:


df_train['price_range'].unique()


# In[12]:


sns.boxplot(df_train['price_range'],df_train['talk_time'])


# In[13]:


labels = ["3G-supported",'Not supported']
values = df_train['three_g'].value_counts().values #0-1300, 1-700


# In[14]:


fig1, ax1 = plt.subplots()
colors = ['gold', 'lightskyblue']
ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90,colors=colors)
plt.show()


# In[15]:


labels = ["4G-supported",'Not supported']
values = df_train['four_g'].value_counts().values
fig1, ax1 = plt.subplots()
colors = ['gold', 'lightskyblue']
ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90,colors=colors)
plt.show()


# In[16]:


plt.figure(figsize=(10,6))
df_train['fc'].hist(alpha=0.5,color='blue',label='Front camera')
df_train['pc'].hist(alpha=0.5,color='red',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')


# In[17]:


scaler = StandardScaler()
x = df_train.drop('price_range',axis=1)
y = df_train['price_range']

scaler.fit(x)
x_transformed = scaler.transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_transformed,y,test_size=0.3)


# In[18]:


model = SVC() 
model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))
print("\nClassificationReport:\n%s"%classification_report(y_test_pred,y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




