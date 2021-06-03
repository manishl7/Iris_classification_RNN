#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


from sklearn.datasets import load_iris


# In[4]:


iris=load_iris()


# In[5]:


type(iris) 


# In[6]:





# In[7]:


#Let's grab feature and label information
X=iris.data #xfeatures already in the attribute in form of data 


# In[8]:





# In[8]:


y=iris.target #y features also orgainzed for us in for of target attribute
y


# In[9]:


from keras.utils import to_categorical


# In[10]:


y=to_categorical(y)


# In[11]:


y.shape #now it's 150 instances but has 3 values per label


# In[21]:


y #i.e hot encoding


# In[ ]:


# now we can split the data to training set and test set


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[15]:


from sklearn.preprocessing import MinMaxScaler


# In[16]:


scaler_object=MinMaxScaler()


# In[17]:


scaler_object.fit(X_train)


# In[18]:


scaled_X_train=scaler_object.transform(X_train)


# In[20]:


scaled_X_test=scaler_object.transform(X_test)


# In[35]:


#NOW WE"LL BUILD A NW USING KERAS


# In[21]:


from keras.models import Sequential 


# In[22]:


from keras.layers import Dense 


# In[23]:


model=Sequential() 
model.add(Dense(8,input_dim=4,activation='relu')) 


# In[24]:


#lets add 1 more dense layer
model.add(Dense(8,input_dim=4,activation='relu'))


# In[25]:


#for our output layer
model.add(Dense(3,activation='softmax'))


# In[26]:


#at the end we simply compile our model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[27]:


#if everything worked well, we can ask for summary
model.summary()


# In[47]:


model.fit(scaled_X_train,y_train,epochs=150,verbose=2)


# In[48]:


#For actual classes
model.predict_classes(scaled_X_test)
#It doesn't predict one hot encoded version,it just prints out classes


# In[ ]:


#To fix the results


# In[49]:


# let's compare this to y classes i.e
predictions=model.predict_classes(scaled_X_test)


# In[36]:


#nowwe'll transform our y_test
y_test.argmax(axis=1) #it'll give out original classes and not hot encoded


# In[37]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[50]:


print(confusion_matrix(y_test.argmax(axis=1),predictions))


# In[ ]:





# In[51]:


print(classification_report(y_test.argmax(axis=1),predictions)) 


# In[52]:


print(accuracy_score(y_test.argmax(axis=1),predictions)*100)


# In[53]:


#If we are running a large model say text generation we need to save our model
model.save('myfirstmodel.h5')


# In[55]:


# to load
from keras.models import load_model


# In[56]:


new_model=load_model('myfirstmodel.h5')


# In[57]:


new_model.predict_classes(scaled_X_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




