
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv('Salary_Data.csv')
dataset



# In[21]:


# iloc integer location based [rows, columns] : means all rows :-1 all columns except last one
X = dataset.iloc[:, :-1].values

# In python indexes are started from 0 and R starts from 1
# 0 column for country; 1 - Age; 2 - Salary; 3 - Purchased
y = dataset.iloc[:, 1].values


# In[22]:


#Splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state =0)

#   Feature Scaling
# For multi-comment line use """ This will not be executed """ 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


# In[23]:


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)



# In[24]:


#   Predicting the Test set results
y_pred = regressor.predict(X_test)


# In[25]:



#   Visualizing the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[26]:



#   Visualizing the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# above line is same as plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

