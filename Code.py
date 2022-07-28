#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


s1 = ["congrats. you have won a lottery of 1 million and you can get the lottery amount by calling the lottery number.",
     "give your bank details to get the lottery amount in your bank account",
     "lottery for sure if your bank account details are verified"]


# In[3]:


s1


# In[4]:


s1[0]


# In[5]:


s1[1]


# In[6]:


s1[2]


# In[7]:


# tokenization
s1[0].split()


# In[8]:


s1[1].split()


# In[9]:


s1[2].split()


# In[10]:


# count vectorization / Bag of words technique
from sklearn.feature_extraction.text import CountVectorizer


# In[11]:


vect = CountVectorizer(stop_words='english')


# In[12]:


op = vect.fit_transform(s1).toarray()
op


# In[13]:


df = pd.DataFrame(op,columns = vect.get_feature_names())
df


# In[14]:


spam = pd.read_table("https://raw.githubusercontent.com/arib168/data/main/spam.tsv")
spam


# In[15]:


spam.info()


# In[16]:


spam.describe


# In[17]:


spam.describe()


# In[18]:


spam.isnull().sum()


# In[19]:


spam.dtypes


# In[20]:


spam['message'][2]


# In[21]:


spam['label'][2]


# In[22]:


# check dataset is balanced or imbalanced
# balanced - metrics accuracy is suffuecient
# imbalanced - metrics(accuracy,recall,precision,f1-score)
spam['label'].value_counts()

#dataset is imbalanced


# In[23]:


# create a machine learning algorithmn to predict if mail is spam or not


# In[24]:


x = spam['message']
y = spam['label']


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)


# In[27]:


# apply countvectorizer then apply svm or naive bayes


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer


# In[29]:


vect = CountVectorizer()


# In[30]:


x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)


# In[31]:


from sklearn.svm import SVC


# In[32]:


model1 = SVC()


# In[33]:


model1.fit(x_train_vect,y_train)


# In[34]:


y_pred1 = model1.predict(x_test_vect)
y_pred1


# In[35]:


y_test


# In[36]:


from sklearn.metrics import accuracy_score


# In[37]:


accuracy_score(y_test,y_pred1)


# In[38]:


data = ['win free tickets to football game']
data = vect.transform(data)


# In[39]:


model1.predict(data)


# In[40]:


data1 = ["let's go out this weekend"]
data1 = vect.transform(data1)


# In[41]:


model1.predict(data1)


# In[42]:


# PIPELINE MODEL USING COUNT VECTORIZER AND SVC
# PIPELINE - the purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters
# Pipeline can be used to chain multiple estimators into one.


# In[43]:


#Pipeline removes the problem of having to apply the transformation(count vectorization) again and again everytime before predicting


# In[44]:


from sklearn.pipeline import make_pipeline


# In[45]:


model2 = make_pipeline(CountVectorizer(),SVC( ))


# In[46]:


model2.fit(x_train,y_train)


# In[47]:


y_pred2 = model2.predict(x_test)


# In[48]:


accuracy_score(y_test,y_pred2)


# In[49]:


from sklearn.naive_bayes import MultinomialNB


# In[50]:


model3 = MultinomialNB()


# In[51]:


model3.fit(x_train_vect,y_train)


# In[52]:


y_pred3 = model3.predict(x_test_vect)
y_pred3


# In[63]:


accuracy_score(y_test,y_pred3)


# In[54]:


data = ['win free tickets to football game']
data = vect.transform(data)


# In[55]:


model3.predict(data)


# In[56]:


data1 = ["let's go out this weekend"]
data1 = vect.transform(data1)


# In[57]:


model3.predict(data1)


# In[58]:


from sklearn.pipeline import make_pipeline


# In[59]:


model4 = make_pipeline(CountVectorizer(),MultinomialNB())


# In[60]:


model4.fit(x_train,y_train)


# In[61]:


y_pred4 = model4.predict(x_test)
y_pred4


# In[62]:


accuracy_score(y_test,y_pred4)


# In[70]:


# Using best model with highest accuracy to crate a joblib file
# joblib is used for pipeline models


# In[71]:


import joblib
joblib.dump(model4,'spam-ham')


# In[72]:


# reload the file
reloaded_model = joblib.load('spam-ham')
reloaded_model


# In[73]:


# prediction using reloaded-model
reloaded_model.predict(['Free tickets for a football match'])


# In[74]:


reloaded_model.predict(['Free entry to the mall'])


# In[ ]:




