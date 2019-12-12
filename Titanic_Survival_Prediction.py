#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


dataset=pd.read_csv(r'C:\Users\lenovo\Downloads\titanic-survival\titanic_data.csv')


# In[5]:


display(dataset)


# In[6]:


label =dataset['Survived']


# In[7]:


display(label)


# In[8]:


features=dataset.drop('Survived',axis=1)


# In[9]:


display(features)


# In[10]:


features =pd.get_dummies(features)


# In[12]:


features =features.fillna(0.0)


# In[13]:


display(features)


# In[14]:


X_train ,X_test, y_train, y_test=train_test_split(features,label,test_size=0.2,random_state=42)


# In[15]:


model=DecisionTreeClassifier().fit(X_train,y_train)


# In[16]:


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# In[17]:


from sklearn.metrics import accuracy_score
print('Training Accuracy:',accuracy_score(y_train,y_train_pred))
print('Testing Accuracy:',accuracy_score(y_test,y_test_pred))


# In[18]:


from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


# In[19]:


parameters= {'max_depth':[2,4,6,8,10],'min_samples_leaf':[2,4,6,8,10],'min_samples_split':[2,4,6,8,10]}


# In[20]:


scorer=make_scorer(f1_score)


# In[21]:


grid_obj=GridSearchCV(model,parameters,scoring=scorer)


# In[22]:


grid_fit=grid_obj.fit(X_train,y_train)


# In[23]:


best_model=grid_fit.best_estimator_


# In[24]:


best_model


# In[25]:


best_model.fit(X_train,y_train)


# In[27]:


y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
print('Training Accuracy:',accuracy_score(y_train,y_train_pred))
print('Testing Accuracy:',accuracy_score(y_test,y_test_pred))


# In[ ]:




