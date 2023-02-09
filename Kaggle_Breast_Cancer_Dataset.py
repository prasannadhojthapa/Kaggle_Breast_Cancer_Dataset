#!/usr/bin/env python
# coding: utf-8

# #### Kaggle Breast Cancer Dataset

# In[1]:


import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

#Install Kaggle
get_ipython().system('pip install kaggle')


# In[26]:


#Note: The code below has been working because of the UTF-8. It has rectified the encoding.
#This loads the data in df
df  = pd.read_csv('C:\\Users\\prasa\\Downloads\\Kaggle Files\\Kaggle CSV Files\\breast-cancer.csv',encoding='UTF-8')
print()
print("data shape: ", df.shape)
print("data dimension: ", df.ndim)
print(df.diagnosis.value_counts())
print(len(df.axes[0])) #This shows the number of rows in the Data.
#print(df.describe())
print(df.columns)


# #### Using K-Nearest-Neigbors with decision boundaries on the Breast Cancer Dataset 

# In[90]:


import pandas as pd
# The file has no headers naming the columns, so we pass header=None
# and provide the column names explicitly in "names"
df = pd.read_csv(
    "C:\\Users\\prasa\\Downloads\\Kaggle Files\\Kaggle CSV Files\\Breast-cancer.csv", header=None,index_col=False, 
    names=['id','diagnosis','radius_mean', 'texture_mean', 'perimeter_mean',
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
           'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst',
           'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst'])
# IPython.display allows nice output formatting within the Jupyter notebook
display(df)


# In[91]:


# Drop first row using drop()
df.drop(index=df.index[0], axis=0, inplace=True)


# In[95]:


#Note: You have to run this ONCE!
print(df.diagnosis.value_counts())
print()
#assigning numbers for the species values:
assign_diagnosis = {'M':1,'B':2}
lst = [assign_diagnosis[k] for k in df.diagnosis]
df = df.values.tolist()

#To merge the two matrix

import sympy as sp

a = sp.Matrix(df).col_insert(-1, sp.Matrix(lst))
a_ = a.tolist()

#To pop the str species column

for row in a_:
    del row[-1]  # 0 for column 1, 1 for column 2, etc.

print(a_)

#Iterator comes here:
an_iterator = iter(a_)
print(an_iterator)


# In[ ]:





# In[93]:


import csv
from sklearn.utils import Bunch
from sklearn import preprocessing

def load_my_dataset():
    with open('C:\\Users\\prasa\\Downloads\\Kaggle Files\\Kaggle CSV Files\\breast-cancer.csv') as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = len(df[0]) #number of data rows, don't count header
        n_features = len(df[1])-1 #number of columns for features, don't count target column
        #n_features will be less than one because you are taking out the target column.
        feature_names = ['id','radius_mean', 'texture_mean', 'perimeter_mean',
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
           'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst',
           'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst'] #adjust accordingly
        target_names = ['diagnosis'] #adjust accordingly
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)
    
        for i, sample in enumerate(a_):
            data[i] = np.asarray(sample[:-1], dtype=np.float64)
            #print(target_names)    
            #Because the target is in strings we need to convert it into integers: Using factorize.
            #target_names = pd.factorize(target_names)[0]
            target[i] = np.asarray(sample[-1], dtype=np.int64)
            #print(data)

        return Bunch(data=data, target=target, feature_names = feature_names, target_names = target_names)
data = load_my_dataset()


# In[94]:


print(data)


# In[7]:


df = pd.DataFrame(df.data, columns=data.feature_names)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)

