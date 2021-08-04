#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[2]:


dataset = pd.read_csv('spam.csv')
X = dataset.iloc[:,1].values
y = dataset.iloc[:,0].values


# In[3]:


print(X)


# ## Cleaning the texts

# In[4]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 5572):
    review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[:,1][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)


# In[5]:


print(corpus)


# ## Creating the Bag of Words model

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,0].values


# In[7]:


print(X)


# In[8]:


print(y)


# # Encoding the y

# In[9]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[10]:


print(y)


# ## Splitting the dataset into the Training set and Test set

# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 9)


# In[12]:


print(len(X_train))
print(X_train)


# In[13]:


print(len(X_test))
print(X_test)


# In[14]:


print(len(y_train))
print(y_train)


# In[17]:


print(len(y_test))
print(y_test)


# ## Training the Naive Bayes model on the Training set

# In[18]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# ## Predicting the Test set results

# In[19]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Making the Confusion Matrix

# In[20]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# # Applying k-Fold Cross Validation

# In[21]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

