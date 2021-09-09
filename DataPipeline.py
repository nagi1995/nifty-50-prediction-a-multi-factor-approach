#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')

get_ipython().system('ln -s /content/drive/MyDrive /mygdrive')


# In[2]:


get_ipython().system('ls /mygdrive')


# # importing libraries

# In[3]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import os
import datetime as dt
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import pickle
import random


# # loading models

# In[4]:


data_path = "/mygdrive/CS1/data/"

with open("/mygdrive/CS1/LR_L1_reg.pkl", "rb") as f: 
  model = pickle.load(f)

with open("/mygdrive/CS1/x_scaler.pkl", "rb") as f: 
  x_scaler = pickle.load(f)

with open("/mygdrive/CS1/y_scaler.pkl", "rb") as f: 
  y_scaler = pickle.load(f)


# # loading data

# In[10]:


df_test = pd.read_csv(data_path + "test_features_only.csv", parse_dates = True)
df_test["Date"] = pd.to_datetime(df_test["Date"], format = "%Y-%m-%d")
df_test.sort_values(by = ["Date"], inplace = True)
df_test.head(2)


# In[11]:


df_test.tail(2)


# # utility functions

# In[12]:


# Reference: https://github.com/Shagun-25/Nifty-Index-Prediction-Using-News-Sentiments/blob/master/Pipeline.ipynb

def single_day_prediction(date): 
  '''
  Given a date this function tries to predict the
  closing value of next day's NIFTY 50 index
  '''
  present_day = df_test[df_test['Date'] == date]
  if present_day.shape[0] == 1: 
    x = present_day.values[0, 1 : -1].reshape(1, -1)
    x_scaled = x_scaler.transform(x)
    
    y_pred_scaled = model.predict(x_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(1, -1))
    
    print("Actual closing index value for next working day is:", present_day.values[0, -1])
    print("Predicted closing index value for next working day is:", y_pred[0, 0])
  
  else:
    print("Please enter a non-stock market holiday date between \'2018-11-02\' and \'2021-08-10\' in \'YYYY-MM-DD format")

def multiple_days_prediction(): 
  
  # randomly selecting a row
  n = random.sample(range(df_test.shape[0] - 60), 1)[0]
  
  x = df_test.values[n : n+60, 1 : -1]
  x_scaled = x_scaler.transform(x)
  y_true = df_test.values[n : n+60, -1]
  
  y_pred_scaled = model.predict(x_scaled)
  
  y_pred = y_scaler.inverse_transform(y_pred_scaled)
  print("RMSE:", mean_squared_error(y_true, y_pred, squared = False))

  plt.figure(figsize = (10, 5))
  plt.plot(df_test.iloc[n : n+60, 0], y_true, "b.-", label = "original")
  plt.plot(df_test.iloc[n : n+60, 0], y_pred, "r.-", label = "predicted")
  plt.legend()
  plt.xlabel("Date")
  plt.ylabel("Index Value")
  plt.title("NIFTY 50 index prediction for 60 consecutive days")
  plt.grid()
  plt.show()


# # single day prediction

# In[13]:


get_ipython().run_cell_magic('time', '', 'single_day_prediction("2018-11-04")')


# In[14]:


get_ipython().run_cell_magic('time', '', 'single_day_prediction("2018-11-05")')


# # multiple days prediction

# In[22]:


get_ipython().run_cell_magic('time', '', 'multiple_days_prediction()')


# In[ ]:




