#!/usr/bin/env python
# coding: utf-8

# In[3]:


#required libraries are imported
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#csv table is fetched here
dataset = pd.read_csv("E:\student_scores.csv")
#print("Dinabandhu Maiti RA1911030020032")

#csv table shape is given here. There are 30 rows and 2 columns
dataset.shape
(30, 2)

dataset.head()
dataset.describe()

#two variables are x and y
dataset.plot(x='Hours', y='Scores', style="*")
plt.title('Marks Prediction')
plt.xlabel('Hours')
plt.ylabel('Percentage Marks')
plt.show()

#dataset is predicted and value is splitted
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#dataset is split into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#linear regression is imported here
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

print(regressor.intercept_)

print(regressor.coef_)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual':Y_test,'Predicted':y_pred})
df
#here 3.062 is intercept and 9.70 is the slope
#this shows that if a student reads more than 2 hours he/she can expect 97% of marks


# In[ ]:




