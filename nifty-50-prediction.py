#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().system('ln -s /content/drive/MyDrive /mygdrive')


# In[3]:


get_ipython().system('ls /mygdrive')


# # importing libraries

# In[7]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import datetime as dt
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, make_scorer
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from prettytable import PrettyTable
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
import cv2
from tensorflow.keras.utils import plot_model
from google.colab.patches import cv2_imshow
import pickle


# In[5]:


print(tf.__version__)


# In[6]:


data_path = "/mygdrive/CS1/data/"
os.listdir(data_path)


# # Performance metrics
# * In this Case Study, we are trying to predict NIFTY 50 index values based on previous data and other indices like S&P500, NASDAQ composite, and Gold prices, Oil prices.
# * This is a regression problem
# * So, MSE or RMSE are the metrics we should target
# * Units of MSE will be square of the units that we are predicting while units of RMSE will be same as the units that we are predicting

# # NIFTY 50

# In[ ]:


# loading NIFTY 50 data

nifty = pd.read_csv(data_path + "NIFTY 50.csv", parse_dates = True)
nifty["Date"] = pd.to_datetime(nifty["Date"], format = "%Y-%m-%d")
nifty.head()


# In[ ]:


nifty.tail()


# In[ ]:


nifty.info()


# In[ ]:


# reference: https://datatofish.com/rows-with-nan-pandas-dataframe/

def na_rows(df):
  '''
  This functions returns rows 
  where NaN values are present
  '''
  return df[df.isna().any(axis=1)]


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", nifty.duplicated().sum())
print("Number of rows with nan values:\n", nifty.isna().sum())


# In[ ]:


# displaying NaN rows of NIFTY 50

na_rows(nifty)


# #### Some of the rows are nan because of various reasons
# * Some of them are holidays Ex: 01 Jan 2013, 01 Jan 2014 etc
# * Due to terrorist attack on 26/11, 27 Nov 2008 is declared as trading hoilday
# * 13 Oct 2009 is poling day, so NSE is closed
# 
# #### Some of the data is missing, so the data from [NSE](https://www1.nseindia.com/products/content/equities/indices/historical_index_data.htm) is copied manually

# In[ ]:


# calculating absolute sum of difference between "Close" and "Adj Close" 

nifty["Close-Adj Close"] = abs(nifty["Close"] - nifty["Adj Close"])
print("column sum of Close-Adj Close:",np.sum(nifty["Close-Adj Close"]))
nifty.head()


# * Columns **Adj Close** and **Close** are same. So, removing **Adj Close** column.
# * dropping Volume column also as Volume information is available from 2013-01-18

# In[ ]:


def same_row_difference_feature(df, numerator, denominator, feature_name, absolute = False):
  '''
  This function calculates percentage change of features for particular row 
  '''
  df[feature_name] = ((df[numerator]/df[denominator]) - 1) * 100
  if absolute:
    df[feature_name] = df[feature_name].abs()
  return df


# In[ ]:


# loading the imputed NIFTY 50 data and calculating percentage 
# change between High and Low values for particular row

nifty_imputed = pd.read_csv(data_path + "NIFTY 50 imputed.csv", parse_dates = True)
nifty_imputed["Date"] = pd.to_datetime(nifty_imputed["Date"], format = "%Y-%m-%d")
nifty_imputed.drop(columns = ["Adj Close", "Volume"], inplace = True)
nifty_imputed = same_row_difference_feature(df = nifty_imputed, 
                                            numerator = "High", 
                                            denominator = "Low", 
                                            feature_name = "(H-L)*100/L")

nifty_imputed.head()


# In[ ]:


# displaying NaN rows of imputed NIFTY 50 data

na_rows(nifty_imputed)


# #### [NSE](https://www1.nseindia.com/products/content/equities/indices/historical_index_data.htm) does not have Adj Close and Volume.
# 

# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", nifty_imputed.duplicated().sum())
print("Number of rows with nan values:\n", nifty_imputed.isna().sum())


# In[ ]:


# splitting the data in 80:20 ratio

nifty_train, nifty_test = train_test_split(nifty_imputed, test_size = .2, shuffle = False)


# In[ ]:


nifty_train.head()


# In[ ]:


nifty_test.head()


# # Time based train test splitting

# ### We have data from 17 Sept, 2007 to 11 Aug, 2021. Of the total data, from 17 Sept, 2007 to 31 Oct, 2018 (approximately 80%) is taken as training data and remaining data is test data

# In[ ]:


# reference: https://stackoverflow.com/a/46230990

def time_based_train_test_split(df, start_date = "2007-09-17", split_date = "2018-10-31", end_date = "2021-08-11"):
  '''
  This function splits the given dataframe based on split_date
  '''
  train = df.loc[df["Date"] >= start_date]
  train = train.loc[train["Date"] <= split_date]
  test = df.loc[df["Date"] > split_date]
  test = test.loc[test["Date"] <= end_date]
  return train, test


# In[ ]:


# splitting NIFTY 50 based on time

nifty_train, nifty_test = time_based_train_test_split(nifty_imputed)
nifty_train = nifty_train.set_index('Date', drop = False)
nifty_test = nifty_test.set_index('Date', drop = False)
nifty_train.shape, nifty_test.shape


# In[ ]:


nifty_train.head()


# In[ ]:


nifty_test.head()


# #### Now that we have split the data, we will do analysis only on train data to prevent data leakage

# In[ ]:


# visualize data

plt.figure()
plt.plot(nifty_train["Date"], nifty_train["Close"], "g", label = "train data")
plt.plot(nifty_test["Date"], nifty_test["Close"], "b", label = "test data")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Index value")
plt.title("NIFTY 50 Index")
plt.grid()
plt.show()


# # NIFTY 50 PE

# In[ ]:


# loading NIFTY 50 PE data

nifty_pe = pd.read_csv(data_path + "NIFTY 50 PE.csv")
nifty_pe.head()


# In[ ]:


nifty_pe.info()


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", nifty_pe.duplicated().sum())
print("Number of rows with nan values:\n", nifty_pe.isna().sum())


# In[ ]:


# formatting the "Time" column and renaming it to "Date"

nifty_pe["Date"] = pd.to_datetime(pd.to_datetime(nifty_pe["Time"], format = "%d %b, %Y, %I:%M %p").dt.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
nifty_pe.drop(columns = ["Time"], inplace = True)
nifty_pe.head()


# In[ ]:


# splitting NIFTY 50 PE based on time

nifty_pe_train, nifty_pe_test = time_based_train_test_split(nifty_pe)
nifty_pe_train.shape, nifty_pe_test.shape


# In[ ]:


nifty_pe_train.head()


# In[ ]:


nifty_pe_test.head()


# In[ ]:


# reference: https://stackoverflow.com/a/45925049

def plot_nifty_wrt_others(other_train, other_test, other_title, col = "Close"):
  '''
  This function takes train and test dataframes of 
  other indices and other variables and plots them
  '''
  fig, ax1 = plt.subplots(figsize=(8,5)) 

  ax2 = ax1.twinx()

  ax1.set_xlabel("Date")
  ax1.set_ylabel("NIFTY 50 Index value")
  ax2.set_ylabel(other_title)

  p11, = ax1.plot(nifty_train["Date"], nifty_train["Close"], color = "b", label = "train NIFTY 50")
  p12, = ax1.plot(nifty_test["Date"], nifty_test["Close"],    color = "g", label = "test NIFTY 50")
  p21, = ax2.plot(other_train["Date"], other_train[col], color = "r", label="train " + other_title)
  p22, = ax2.plot(other_test["Date"], other_test[col], color = "y", label="test " + other_title)

  lns = [p11, p12, p21, p22]
  ax1.legend(handles = lns, loc='best')

  fig.tight_layout()
  plt.grid()
  plt.show()


# In[ ]:


# plotting NIFTY 50 wrt NIFTY 50 PE

plot_nifty_wrt_others(nifty_pe_train, nifty_pe_test, "NIFTY 50 PE", col = "PE_NIFTY50")


# #### NIFTY 50 PE ratio [Reference](https://www.samco.in/knowledge-center/articles/nifty-50-pe-ratio/)
# #### Price to Earnings (PE) ratio tells whether the market is undervalued, overvalued or fairly valued.
# <img src='https://imgur.com/FTmxotz.png'>
# 
# #### Buffett indicator [Reference](https://www.investopedia.com/terms/m/marketcapgdp.asp)
# #### Market capitalization to GDP ratio can also be used to tell if market is undervalued, overvalued or fairly valued.
# <img src='https://imgur.com/Az7RWAW.png'>
# 
# ### There are many other factors to say about market valuation. Here we are using one such indicator namely PE ratio
# ### From the above plot we can see that whenever PE went above 25, market corrected except in pandemic year after market fall sharpely in march 2020. From 01 April, 2021, consolidated earnings of companies were taken into account, so, there is drastic fall in PE.

# # Brent crude oil prices

# In[ ]:


# loading crude oil prices 

crude = pd.read_csv(data_path + "Brent crude oil.csv", parse_dates = True)
crude["Date"] = pd.to_datetime(crude["Date"], format = "%Y-%m-%d")
crude.head()


# In[ ]:


crude.info()


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", crude.duplicated().sum())
print("Number of rows with nan values:\n", crude.isna().sum())


# In[ ]:


# displaying NaN rows of crude oil dataframe

na_rows(crude)


# * Missing data is manually copied from [investing.com](https://in.investing.com/commodities/brent-oil-historical-data?end_date=1620239400&st_date=1195669800)

# In[ ]:


# calculating absolute sum of difference between "Close" and "Adj Close" 

crude["Close-Adj Close"] = abs(crude["Close"] - crude["Adj Close"])
print("column sum of Close-Adj Close:",np.sum(crude["Close-Adj Close"]))
crude.head()


# * Columns **Adj Close** and **Close** are same. So, removing **Adj Close** column

# In[ ]:


# loading the imputed crude oil data and calculating percentage 
# change between High and Low values for particular row

crude_imputed = pd.read_csv(data_path + "Brent crude oil imputed.csv", parse_dates = True)
crude_imputed["Date"] = pd.to_datetime(crude_imputed["Date"], format = "%Y-%m-%d")

crude_imputed = same_row_difference_feature(df = crude_imputed, 
                                            numerator = "High", 
                                            denominator = "Low", 
                                            feature_name = "crude (H-L)*100/L")
# renaming columns
crude_imputed.rename(columns = {"Open" : "crude open", 
                                "High" : "crude high", 
                                "Low" : "crude low",
                                "Close" : "crude close"},
                     inplace = True)

crude_imputed.drop(columns = ["Adj Close", "Volume"], inplace = True)
crude_imputed.head()


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", crude_imputed.duplicated().sum())
print("Number of rows with nan values:\n", crude_imputed.isna().sum())


# * Adj Close and volume information is not present in [investing.com](https://in.investing.com/commodities/brent-oil-historical-data?end_date=1620239400&st_date=1195669800)

# In[ ]:


# splitting crude oil prices dataframe based on time

crude_train, crude_test = time_based_train_test_split(crude_imputed)
crude_train.shape, crude_test.shape


# In[ ]:


crude_train.head()


# In[ ]:


crude_test.head()


# In[ ]:


plot_nifty_wrt_others(crude_train, crude_test, "Brent Crude Oil Prices", col = "crude close")


# ### How movement in crude oil price impacts economy and stock market [moneycontrol.com](https://www.moneycontrol.com/news/business/markets/how-movement-in-crude-oil-price-impacts-economy-and-stock-market-3576051.html)
# * India imports 86 percent of its annual crude oil requirement. Since the payments are made in the US dollars, India’s deficit will depend on crude price as well as on the USD/INR exchange rates.
# * Crude price impact on Indian economy: Higher crude price will have a negative impact on the fiscal and current account deficits of the economy. Increase in these deficits will lead to higher inflation and also impact monetary policy, consumption, and investment behaviour in the economy. A 10 percent increase in oil price will increase the trade deficit by $7 billion, that is, trade deficit will widen by 560bps.
# * Impact on Indian financial markets: Energy stocks have 12.5 percent weightage in the Nifty50 and 15.2 percent in the Sensex. Hence, the Nifty and the Sensex are sensitive to oil price movements. Higher crude prices adversely affect tyre manufacturers, footwear, lubricants, paints, and airline companies.
# 
# ### Observation:
# * Moderate or fair oil prices impacts the index positively or does not have an impact whereas has too high prices impacts index negatively

# # Gold

# In[ ]:


# loading gold prices dataframe

gold = pd.read_csv(data_path + "Gold.csv", parse_dates = True)
gold["Date"] = pd.to_datetime(gold["Date"], format = "%Y-%m-%d")
gold.head()


# In[ ]:


gold.info()


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", gold.duplicated().sum())
print("Number of rows with nan values:\n", gold.isna().sum())


# In[ ]:


# displaying NaN rows of gold prices dataframe

na_rows(gold)


# * Missing data is manually copied from [investing.com](https://in.investing.com/commodities/gold-historical-data?end_date=1629743400&st_date=1189881000)

# In[ ]:


# calculating absolute sum of difference between "Close" and "Adj Close" 

gold["Close-Adj Close"] = abs(gold["Close"] - gold["Adj Close"])
print("column sum of Close-Adj Close:", np.sum(gold["Close-Adj Close"]))
gold.head()


# * Columns **Adj Close** and **Close** are same. So, removing **Adj Close** column

# In[ ]:


# loading the imputed gold prices data and calculating percentage 
# change between High and Low values for particular row

gold_imputed = pd.read_csv(data_path + "Gold imputed.csv", parse_dates = True)
gold_imputed["Date"] = pd.to_datetime(gold_imputed["Date"], format = "%Y-%m-%d")

gold_imputed = same_row_difference_feature(df = gold_imputed, 
                                           numerator = "High", 
                                           denominator = "Low", 
                                           feature_name = "gold (H-L)*100/L")
# renaming columns
gold_imputed.rename(columns = {"Open" : "gold open", 
                               "High" : "gold high", 
                               "Low" : "gold low",
                               "Close" : "gold close"}, inplace = True)

gold_imputed.drop(columns = ["Adj Close", "Volume"], inplace = True)
gold_imputed.head()


# In[ ]:


gold_imputed.tail()


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", gold_imputed.duplicated().sum())
print("Number of rows with nan values:\n", gold_imputed.isna().sum())


# * Adj Close and Volume information is not present in [investing.com](https://in.investing.com/commodities/gold-historical-data?end_date=1629743400&st_date=1189881000)

# In[ ]:


# splitting gold prices dataframe based on time

gold_train, gold_test = time_based_train_test_split(gold_imputed)
gold_train.shape, gold_test.shape


# In[ ]:


gold_train.head()


# In[ ]:


gold_test.head()


# In[ ]:


# plotting NIFTY 50 wrt gold closing prices
plot_nifty_wrt_others(gold_train, gold_test, "Gold Prices", col = "gold close")


# #### Gold has traditionally been considered a safe haven and a hedge to equity markets ([Reference](https://www.moneycontrol.com/news/business/personal-finance/how-much-of-gold-and-global-equity-do-you-need-in-your-portfolio-5553681.html)).
# #### In times of market crash, generally Gold holds or increases its value. 
# #### In asset allocation, it is adviced to have around 5-10% holdings in Gold as hedging.
# ### Observation
# * In 2008, when market crashed, gold has hold its value and remained almost constant around 900 levels.
# * When market crashed in 2020, Gold prices increased from around 1600 level to 2000 level. 

# # Euronext 100

# In[ ]:


# loading euronext 100 index values

euronext = pd.read_csv(data_path + "Euronext100.csv", parse_dates = True)
euronext["Date"] = pd.to_datetime(euronext["Date"], format = "%Y-%m-%d")
euronext.head()


# In[ ]:


euronext.info()


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", euronext.duplicated().sum())
print("Number of rows with nan values:\n", euronext.isna().sum())


# In[ ]:


# displaying NaN rows of euronext 100 index

na_rows(euronext)


# * 01 May 2014 is trading hoilday
# * 25 Dec 2019 is Christmas
# * 02 Jan 2012 data is copied manually from [investing.com](https://in.investing.com/indices/euronext-100-historical-data)

# In[ ]:


# calculating absolute sum of difference between "Close" and "Adj Close" 

euronext["Close-Adj Close"] = abs(euronext["Close"] - euronext["Adj Close"])
print("column sum of Close-Adj Close:", np.sum(euronext["Close-Adj Close"]))
euronext.head()


# * Columns **Adj Close** and **Close** are same. So, removing **Adj Close** column

# In[ ]:


# loading the imputed euronext 100 data and calculating percentage 
# change between High and Low values for particular row

euronext_imputed = pd.read_csv(data_path + "Euronext100 imputed.csv", parse_dates = True)
euronext_imputed["Date"] = pd.to_datetime(euronext_imputed["Date"], format = "%Y-%m-%d")

euronext_imputed = same_row_difference_feature(df = euronext_imputed, 
                                           numerator = "High", 
                                           denominator = "Low", 
                                           feature_name = "euronext (H-L)*100/L")
# renaming columns
euronext_imputed.rename(columns = {"Open" : "euronext open", 
                                   "High" : "euronext high", 
                                   "Low" : "euronext low",
                                   "Close" : "euronext close"}, inplace = True)

euronext_imputed.drop(columns = ["Adj Close", "Volume"], inplace = True)
euronext_imputed.head()


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", euronext_imputed.duplicated().sum())
print("Number of rows with nan values:\n", euronext_imputed.isna().sum())


# #### Adj Close information is not available in [investing.com](https://in.investing.com/indices/euronext-100-historical-data)

# In[ ]:


# splitting euronext 100 index dataframebased on time
euronext_train, euronext_test = time_based_train_test_split(euronext_imputed)
euronext_train.shape, euronext_test.shape


# In[ ]:


euronext_train.head()


# In[ ]:


euronext_test.head()


# In[ ]:


# plotting NIFTY 50 wrt euronext 100 index
plot_nifty_wrt_others(euronext_train, euronext_test, "Euronext 100 Index value", col = "euronext close")


# #### The Euronext 100 Index is the blue chip index of the pan-European exchange, Euronext NV ([Reference](https://en.wikipedia.org/wiki/Euronext_100)).
# ### Observation:
# * When the time range is divided into intervals, it can be observed that both indices move in same direction.
# * Example: Till around 2009, both indices corrected.
# * Between 2016 and 2018 both indices alomst have an uptrend.
# * In march 2020, all the stocks around the Globe crashed.

# # NASDAQ composite

# In[ ]:


# loading nasdaq composite index data

nasdaq = pd.read_csv(data_path + "NASDAQ composite.csv", parse_dates = True)
nasdaq["Date"] = pd.to_datetime(nasdaq["Date"], format = "%Y-%m-%d")
nasdaq.head()


# In[ ]:


nasdaq.info()


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", nasdaq.duplicated().sum())
print("Number of rows with nan values:\n", nasdaq.isna().sum())


# In[ ]:


# calculating absolute sum of difference between "Close" and "Adj Close" 

nasdaq["Close-Adj Close"] = abs(nasdaq["Close"] - nasdaq["Adj Close"])
print("column sum of Close-Adj Close:",np.sum(nasdaq["Close-Adj Close"]))
nasdaq.head()


# * Columns **Adj Close** and **Close** are same. So, removing **Adj Close** column.

# In[ ]:


# calculating percentage change between High and Low values for particular row

nasdaq = same_row_difference_feature(df = nasdaq, 
                                           numerator = "High", 
                                           denominator = "Low", 
                                           feature_name = "nasdaq (H-L)*100/L")

# renaming columns
nasdaq.rename(columns = {"Open" : "nasdaq open", 
                         "High" : "nasdaq high", 
                         "Low" : "nasdaq low", 
                         "Close" : "nasdaq close"}, inplace = True)

nasdaq.drop(columns = ["Adj Close",  "Close-Adj Close", "Volume"], inplace = True)
nasdaq.head()


# In[ ]:


# splitting nasdaq composite data based on time
nasdaq_train, nasdaq_test = time_based_train_test_split(nasdaq)
nasdaq_train.shape, nasdaq_test.shape


# In[ ]:


nasdaq_train.head()


# In[ ]:


nasdaq_test.head()


# In[ ]:


# plotting NIFTY 50 wrt nasdaq composite index closing values
plot_nifty_wrt_others(nasdaq_train, nasdaq_test, "NASDAQ Composite Index value", col = "nasdaq close")


# ### Observation:
# * When the time range is divided into intervals, it can be observed that both indices move in same direction.
# * Example: Till around 2009, both indices corrected.
# * Between 2016 and 2018 both indices almost have an uptrend.

# # S&P500

# In[ ]:


# loading S&P 500 index data

sp500 = pd.read_csv(data_path + "S&P 500.csv", parse_dates = True)
sp500["Date"] = pd.to_datetime(sp500["Date"], format = "%Y-%m-%d")
sp500.head()


# In[ ]:


sp500.info()


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", sp500.duplicated().sum())
print("Number of rows with nan values:\n", sp500.isna().sum())


# In[ ]:


# calculating absolute sum of difference between "Close" and "Adj Close" 

sp500["Close-Adj Close"] = abs(sp500["Close"] - sp500["Adj Close"])
print("column sum of Close-Adj Close:",np.sum(sp500["Close-Adj Close"]))
sp500.head()


# * Columns **Adj Close** and **Close** are same. So, removing **Adj Close** column.

# In[ ]:


# calculating percentage change between High and Low values for particular row

sp500 = same_row_difference_feature(df = sp500, 
                                    numerator = "High", 
                                    denominator = "Low", 
                                    feature_name = "sp500 (H-L)*100/L")

# renaming columns
sp500.rename(columns = {"Open" : "sp500 open", 
                        "High" : "sp500 high", 
                        "Low" : "sp500 low", 
                        "Close" : "sp500 close"}, inplace = True)

sp500.drop(columns = ["Adj Close", "Close-Adj Close", "Volume"], inplace = True)
sp500.head()


# In[ ]:


# splitting S&P 500 based on time
sp500_train, sp500_test = time_based_train_test_split(sp500)
sp500_train.shape, sp500_test.shape


# In[ ]:


sp500_train.head()


# In[ ]:


sp500_test.head()


# In[ ]:


#plotting NIFTY 50 wrt S&P 500 index closing values
plot_nifty_wrt_others(sp500_train, sp500_test, "S&P 500 Index value", col = "sp500 close")


# ### Observation:
# * Both the indices almost overlap or remained in range for most of the time.

# # US 10 year treasury yield

# In[ ]:


# loading treasury yield data

treasury = pd.read_csv(data_path + "Treasury Yield 10 Years.csv", parse_dates = True)
treasury["Date"] = pd.to_datetime(treasury["Date"], format = "%Y-%m-%d")
treasury.head()


# In[ ]:


treasury.info()


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", treasury.duplicated().sum())
print("Number of rows with nan values:\n", treasury.isna().sum())


# In[ ]:


# displaying NaN rows

na_rows(treasury)


# * Most of the missing rows belong to weekends.
# * Some of the missing rows are copied from [investing.com](https://in.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data)

# In[ ]:


# calculating absolute sum of difference between "Close" and "Adj Close" 

treasury["Close-Adj Close"] = abs(treasury["Close"] - treasury["Adj Close"])
print("column sum of Close-Adj Close:", np.sum(treasury["Close-Adj Close"]))
treasury.head()


# * Columns **Adj Close** and **Close** are same. So, removing **Adj Close** column.

# In[ ]:


# loading the imputed treasury yields data and calculating percentage 
# change between High and Low values for particular row

treasury_imputed = pd.read_csv(data_path + "Treasury Yield 10 Years imputed.csv", parse_dates = True)
treasury_imputed["Date"] = pd.to_datetime(treasury_imputed["Date"], format = "%Y-%m-%d")

treasury_imputed = same_row_difference_feature(df = treasury_imputed, 
                                           numerator = "High", 
                                           denominator = "Low", 
                                           feature_name = "treasury (H-L)*100/L")

# renaming columns
treasury_imputed.rename(columns = {"Open" : "treasury open", 
                                  "High" : "treasury high", 
                                  "Low" : "treasury low",
                                  "Close" : "treasury close"}, inplace = True)

treasury_imputed.drop(columns = ["Adj Close", "Volume"], inplace = True)
treasury_imputed.head()


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", treasury_imputed.duplicated().sum())
print("Number of rows with nan values:\n", treasury_imputed.isna().sum())


# * Adj Close and Volume are not present in [investing.com](https://in.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data)

# In[ ]:


# splitting treasury yields data based on time
treasury_train, treasury_test = time_based_train_test_split(treasury_imputed)
treasury_train.shape, treasury_test.shape


# In[ ]:


treasury_train.head()


# In[ ]:


treasury_test.head()


# In[ ]:


# plotting NIFTY 50 wrt treasury yields
plot_nifty_wrt_others(treasury_train, treasury_test, "US 10 year treasury yield", col = "treasury close")


# ### Why the 10-Year U.S. Treasury Yield Matters ([Reference](https://www.investopedia.com/articles/investing/100814/why-10-year-us-treasury-rates-matter.asp))
# <img src='https://imgur.com/IFUFVoy.png'>
# 
# ### Observation:
# * Abnormal change in treasury yields impacts index 
# * Treasury yields increased from 2.5% to 3.75% around 2011 at the same time index went down from 6000 levels to 5000 levels
# * Treasury yields increased from 1.5% to 2.5% around 2017 at the same time index went down from 9000 levels to 8000 levels
# 

# # USD-INR conversion

# In[ ]:


# loading USD-INR exchange rate data

usd_inr = pd.read_csv(data_path + "USD-INR conversion.csv", parse_dates = True)
usd_inr["Date"] = pd.to_datetime(usd_inr["Date"], format = "%Y-%m-%d")
usd_inr.head()


# In[ ]:


usd_inr.info()


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", usd_inr.duplicated().sum())
print("Number of rows with nan values:\n", usd_inr.isna().sum())


# In[ ]:


# displaying NaN rows

na_rows(usd_inr)


# #### data is copied manually from [investing.com](https://in.investing.com/currencies/usd-inr-historical-data)

# In[ ]:


# calculating absolute sum of difference between "Close" and "Adj Close" 

usd_inr["Close-Adj Close"] = abs(usd_inr["Close"] - usd_inr["Adj Close"])
print("column sum of Close-Adj Close:",np.sum(usd_inr["Close-Adj Close"]))
usd_inr.head()


# * Columns **Adj Close** and **Close** are same. So, removing **Adj Close** column.

# In[ ]:


# loading the imputed USD-INR exchange rate data and calculating percentage 
# change between High and Low values for particular row

usd_inr_imputed = pd.read_csv(data_path + "USD-INR conversion imputed.csv", parse_dates = True)
usd_inr_imputed["Date"] = pd.to_datetime(usd_inr_imputed["Date"], format = "%Y-%m-%d")

usd_inr_imputed = same_row_difference_feature(df = usd_inr_imputed, 
                                              numerator = "High", 
                                              denominator = "Low", 
                                              feature_name = "usd_inr (H-L)*100/L")

# renaming columns
usd_inr_imputed.rename(columns = {"Open" : "usd_inr open", 
                                  "High" : "usd_inr high", 
                                  "Low" : "usd_inr low",
                                  "Close" : "usd_inr close"}, inplace = True)

usd_inr_imputed.drop(columns = ["Adj Close", "Volume"], inplace = True)
usd_inr_imputed.head()


# In[ ]:


# Checking for duplicate values and nuumber of rows with NaN values

print("Number of duplicate rows:", usd_inr_imputed.duplicated().sum())
print("Number of rows with nan values:\n", usd_inr_imputed.isna().sum())


# #### Adj Close information is not available in [investing.com](https://in.investing.com/currencies/usd-inr-historical-data)

# In[ ]:


# splitting USD-INR exchange data based on time
usd_inr_train, usd_inr_test = time_based_train_test_split(usd_inr_imputed)
usd_inr_train.shape, usd_inr_test.shape


# In[ ]:


usd_inr_train.head()


# In[ ]:


usd_inr_test.head()


# In[ ]:


# plotting NIFTY 50 wrt USD-INR exchange rate 
plot_nifty_wrt_others(usd_inr_train, usd_inr_test, "USD INR Conversion", col = "usd_inr close")


# ### Impact of Exchange Rate Fluctuation ([Reference](https://www.equityfriend.com/articles/38-effect-of-rupee-movement-on-stock-prices.html))
# 
# #### Indian companies can be divided into two groups based on the impact of currency fluctuation on their stock price and profitability:
# 
# 1. Net Exporters – These companies sell product to outside world and receive payment in foreign currency (be it dollar, pound, euro etc). Whenever rupee appreciates as compared to these currencies, companies are exposed to translation loss as they can buy fewer rupees with same amount of foreign currency. This translation loss hurts their profitability since the raw material cost is in terms of rupees. Similarly, company’s profitability increases in case of rupee depreciation.
# 2. Net Importers – These companies buy product from outside world and make payment in foreign currency. Whenever rupee appreciates they are able to buy more foreign currency for payment resulting in overall translation gain. Profitability of companies increases in this case and similarly, profitability decreases when rupee depreciates.
# 
# #### [Blog](https://www.motilaloswal.com/blog-details/Does-the-Nifty-performance-correlate-to-the-INR-USD-movement/1321) by Motilal Oswal says that the Nifty is not overly dependent on the INR/USD currency over the longer term.
# 
# ### Observation:
# * Abnormal change in USD-INR conversion impacts index value
# * Example: Around 2009, USD-INR conversion increased steeply from around 45 to around 50 and index value decreased steeply from 4500 levels to 2500 levels
# 
#  

# # Testing for stationarity in NIFTY 50 

# In[ ]:


#reference: https://www.kaggle.com/kanncaa1/time-series-prediction-tutorial-with-eda?scriptVersionId=24709907&cellId=35

# check_adfuller
def check_adfuller(ts):
  
  print("NULL HYPOTHESIS: time series is NOT stationary")
  print("ALTERNATE HYPOTHESIS: time series is stationary")

  # Dickey-Fuller test
  result = adfuller(ts, autolag = "AIC")

  print('Test statistic:', result[0])
  print('p-value:', result[1])
  print('Critical Values:' ,result[4])

  if (result[0] > result[4]["5%"]):
    print("Test statistic is greater than 5% critical value, ACCEPT NULL HYPOTHESIS")
  else:
    print("Test statistic is less than 5% critical value, REJECT NULL HYPOTHESIS")
  
  if result[1] < 0.05:
    print("p-value is less than 0.05, REJECT NULL HYPOTHESIS")
  else:
    print("p-value is greater than 0.05, ACCEPT NULL HYPOTHESIS")

# check_mean_std
def check_mean_std(ts, date, window = 6):
  #Rolling statistics
  rolmean = ts.rolling(window = window).mean()
  rolstd = ts.rolling(window = window).std()
  
  plt.figure(figsize = (15, 10))   
  plt.plot(date, ts, color = 'red', label = 'Original')
  plt.plot(date, rolmean, color = 'black', label = 'Rolling Mean')
  plt.plot(date, rolstd, color = 'green', label = 'Rolling Std')
  plt.xlabel("Date")
  plt.ylabel("NIFTY 50 index value")
  plt.title('Rolling Mean & Standard Deviation of NIFTY 50 with window = '+str(window))
  plt.legend()
  plt.grid()
  plt.show()


# In[ ]:


# check stationary: mean, variance and adfuller test

date = nifty_train["Date"]
check_mean_std(nifty_train["Close"], date)
check_adfuller(nifty_train["Close"])


# In[ ]:


check_mean_std(nifty_train["Close"], date, window = 30)


# In[ ]:


check_mean_std(nifty_train["Close"], date, window = 60)


# # Conditions for time series to be stationary: ([Reference](https://cran.r-project.org/web/packages/TSTutorial/vignettes/Stationary.pdf))
# * constant mean for all t
# * constant standard deviation for all t
# * autocovariance c(k,l) depends only on difference (k-l)
# 
# # Dickey Fuller Test:
# If test statistic is greater than critical value, then accept NULL HYPOTHESIS, else REJECT NULL HYPOTHESIS

# In[ ]:


# plotting distribution of closing NIFTY 50 index values

sns.displot(data = nifty_train, x = "Close", kde = True)
plt.xlabel("Closing value")
plt.title("Distribution plot of NIFTY 50 index closing values")
plt.show()


# #### Observation
# * The market was almost flat from 2000 to 2014 around 5000 levels, so a peak is observed at 5000 levels
# * similar peaks were observed at 8000 and 11000 levels

# In[ ]:


# Reference: https://www.kaggle.com/kanncaa1/time-series-prediction-tutorial-with-eda?scriptVersionId=24709907&cellId=41

periods = 1
ts = nifty_train["Close"]
date = nifty_train["Date"]

# differencing method
ts_diff = ts - ts.shift(periods = periods)

plt.figure()
plt.plot(date, ts_diff)
plt.title("Differencing method") 
plt.xlabel("Date")
plt.ylabel("Differencing NIFTY 50 closing values")
plt.grid()
plt.show()


# In[ ]:


ts_diff.dropna(inplace = True)

# check stationary 

check_mean_std(ts_diff, date.values[periods:])
check_adfuller(ts_diff)


# # Baseline Model

# In[ ]:


'''
mean of train data will be the baseline prediction for training set and 
mean of test data will be the baseline prediction for testing set
'''

baseline_train_prediction = [nifty_train["Close"].mean()] * nifty_train["Close"].shape[0]
baseline_test_prediction = [nifty_test["Close"].mean()] * nifty_test["Close"].shape[0]

# calculating RMSE
baseline_train_rmse = mean_squared_error(nifty_train["Close"], baseline_train_prediction, squared = False)
baseline_test_rmse = mean_squared_error(nifty_test["Close"], baseline_test_prediction, squared = False)

print("###################################################################")
print("Baseline model")
print("train rmse:", baseline_train_rmse)
print("test rmse:", baseline_test_rmse)
print("###################################################################")

# visualization

plt.figure()
plt.plot(nifty_train["Date"], nifty_train["Close"], label = "train original")
plt.plot(nifty_test["Date"], nifty_test["Close"], label = "test original")
plt.plot(nifty_train["Date"], baseline_train_prediction, label = "train prediction")
plt.plot(nifty_test["Date"], baseline_test_prediction, label = "test prediction")
plt.xlabel("Date")
plt.ylabel("NIFTY 50 index value")
plt.title("Baseline prediction")
plt.legend()
plt.grid()
plt.show()


# # Simple Moving Average Model

# In[ ]:


def SMA(ts, window):
  """
  simple moving average of the previous i days will 
  be predicted as (i+1)th day prediction.
  After predictions RMSE is calculated. 
  """
  rmse = []
  for i in tqdm(window):
    rolling_mean = ts.rolling(i).mean().shift(1) # shifting by 1 because mean of previous i values will be predicted as (i+1)th prediction
    rmse.append(mean_squared_error(ts.values[i:], rolling_mean.dropna(), squared = False))
    
  plt.figure()
  plt.plot(window, rmse, "bo-", label = "train rmse")
  plt.legend()
  plt.xlabel("window")
  plt.ylabel("RMSE")
  plt.title("train RMSE wrt window size")
  plt.grid()
  plt.show()
SMA(nifty_train["Close"], range(1, 50))


# * From above plot, we can that train RMSE is lowest when we predict yesterday's closing price as today's close price.

# In[ ]:


# calculating RMSE
train_rmse = mean_squared_error(nifty_train["Close"].values[1:], nifty_train["Close"].shift(1).dropna(), squared = False)
test_rmse = mean_squared_error(nifty_test["Close"].values[1:], nifty_test["Close"].shift(1).dropna(), squared = False)

print("###################################################################")
print("Simple Moving Average Model")
print("train rmse:", train_rmse)
print("test rmse:", test_rmse)
print("###################################################################")

# visualization
plt.figure(figsize = (7.5, 5))
plt.plot(nifty_train["Date"], nifty_train["Close"], "yo-", label = "train original")
plt.plot(nifty_test["Date"], nifty_test["Close"], "go-", label = "test original")
plt.plot(nifty_train["Date"], nifty_train["Close"].shift(1), "r", label = "train prediction")
plt.plot(nifty_test["Date"], nifty_test["Close"].shift(1), "pink", label = "test prediction")
plt.legend()
plt.xlabel("Date")
plt.ylabel("NIFTY 50 index value")
plt.title("Simple Moving Average Prediction")
plt.grid()
plt.show()


# # Weighted Moving Average

# In[ ]:


def WMA(ts, window):
  """
  Weighted Moving Average of the previous i days will 
  be predicted as (i+1)th day prediction.
  After predictions RMSE is calculated. 
  """
  rmse = []
  for i in tqdm(window):
    # Reference: https://stackoverflow.com/a/66293592
    wma = ts.rolling(i).apply(lambda x: x[::-1].cumsum().sum() * 2 / i / (i + 1)).shift(1) # shifting by 1 because mean of previous i values will be predicted as (i+1)th prediction
    rmse.append(mean_squared_error(ts.values[i:], wma.dropna(), squared = False))
    
  plt.figure()
  plt.plot(window, rmse, "bo-", label = "train rmse")
  plt.legend()
  plt.xlabel("window")
  plt.ylabel("RMSE")
  plt.title("train RMSE wrt window size")
  plt.grid()
  plt.show()
WMA(nifty_train["Close"], range(1, 50))


# * From above plot, we can that train RMSE is lowest when Weighted Moving Average window size = 1 i.e., we predict yesterday's closing price as today's close price.
# * WMA with window size = 1 is same as SMA with window size = 1. 

# # Exponential Moving Average Model

# In[ ]:


def EMA(ts, window, alpha):
  """
  Exponential Weighted Average of the last i days will 
  be predicted as (i+1)th day prediction.
  After predictions RMSE is calculated. 
  """
  count = 1
  plt.figure(figsize = (25, 10))
  for i, a in enumerate(alpha):
    rmse = []
    for j in window:
      ema = ts.ewm(alpha = a, min_periods = j, adjust = False).mean().shift(1) # shifting by 1 because mean of previous i values will be predicted as (i+1)th prediction
      rmse.append(mean_squared_error(ts.values[j:], ema.dropna(), squared = False))
    
    plt.subplot(2, 5, count)
    plt.plot(window, rmse, "bo-")
    plt.xlabel("window")
    plt.ylabel("RMSE")
    plt.title("alpha = "+str(a))
    plt.grid()
    count += 1
  plt.suptitle("train RMSE")
  plt.show()
    

alpha = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
EMA(nifty_train["Close"], range(1, 1000), alpha)


# * alpha = 1 (without shift = 1) implies predicted value will be equal to today's index value. But we are shifting by 1 as we are making prediction for next day. In short we are predicting yesterday's closing price as today's closing price. So, we won't consider alpha = 1.
# 
# * When we observe alpha = 1 plot, RMSE decreases till window size = 450 (approx). This is because, first window size predictions will be NaN. For Example, if our window size is 10, then our prediction will start from 11th day. In this case we are missing values of first 10 days in RMSE calculation.
# 
# * Keeping alpha as constant, the RMSE first decreases, reaches a minimum and then increases.
# 
# * As alpha increases, the minimum RMSE value for particular plot is decreasing.

# In[ ]:


EMA(nifty_train["Close"], range(410, 500), [.9])


# * choosing alpha = 0.9 and window size = 473

# In[ ]:


alpha = 0.9
window = 473

# prediction
train_ema = nifty_train["Close"].ewm(alpha = alpha, min_periods = window, adjust = False).mean().shift(1)
test_ema = nifty_test["Close"].ewm(alpha = alpha, min_periods = window, adjust = False).mean().shift(1)

# calculating RMSE
train_rmse = mean_squared_error(nifty_train["Close"].values[window:], train_ema.dropna(), squared = False)
test_rmse = mean_squared_error(nifty_test["Close"].values[window:], test_ema.dropna(), squared = False)


print("###################################################################")
print("Exponential Moving Average Model")
print("train rmse:", train_rmse)
print("test rmse:", test_rmse)
print("###################################################################")

# visualization
plt.figure(figsize = (7.5, 5))
plt.plot(nifty_train["Date"], nifty_train["Close"], "yo-", label = "train original")
plt.plot(nifty_test["Date"], nifty_test["Close"], "go-", label = "test original")
plt.plot(nifty_train["Date"], train_ema, "r", label = "train prediction")
plt.plot(nifty_test["Date"], test_ema, "pink", label = "test prediction")
plt.legend()
plt.xlabel("Date")
plt.ylabel("NIFTY 50 index value")
plt.title("Exponential Moving Average Prediction")
plt.grid()
plt.show()


# In[ ]:


nifty_test.shape


# * There is a marginal decrease in test RMSE when compared to SMA. But the important thing to be noted is that, there are less number of data points in calculation of RMSE in EMA when compared to SMA. 
# 
# * Number of test predictions in SMA = 682 (nifty_test.shape[0]-1)
# * Number of test predictions in EMA = 210 (nifty_test.shape[0]-473)

# # ARIMA 

# In[ ]:


# Reference: https://github.com/Shagun-25/Nifty-Index-Prediction-Using-News-Sentiments/blob/master/Stock_Prediction.ipynb


lag_acf = acf(ts_diff, nlags = 40)
lag_pacf = pacf(ts_diff, nlags = 40, method = "ols")

# visualization
plt.figure(figsize = (10, 10))
# ACF
plt.subplot(211) 
plt.plot(lag_acf, "b*-")
plt.axhline(y = 0, linestyle = "--", color = "gray")
plt.axhline(y = -1.96/np.sqrt(len(ts_diff)), linestyle = "--", color = "gray")
plt.axhline(y = 1.96/np.sqrt(len(ts_diff)), linestyle = "--", color = "gray")
plt.title("Autocorrelation Function")
plt.grid()
# PACF
plt.subplot(212)
plt.plot(lag_pacf, "bo-")
plt.axhline(y = 0, linestyle = "--", color = "gray")
plt.axhline(y = -1.96/np.sqrt(len(ts_diff)), linestyle = "--", color = "gray")
plt.axhline(y = 1.96/np.sqrt(len(ts_diff)), linestyle = "--", color = "gray")
plt.title("Partial Autocorrelation Function")
plt.grid()
plt.show()


# In[ ]:


# fit model
model = ARIMA(ts, order=(1,0,1)) # (ARMA) = (1,0,1)
model_fit = model.fit(disp=0)

# predict
start_index = 2736 
end_index = 3418 
forecast = model_fit.predict(start=start_index, end=end_index)

# visualization
plt.figure(figsize=(7.5, 5))
plt.plot(nifty_train["Date"], nifty_train["Close"], label = "train original")
plt.plot(nifty_test["Date"], nifty_test["Close"], label = "test original")
plt.plot(nifty_test["Date"], forecast,label = "test predicted")
plt.title("ARIMA(1, 0, 1) prediction")
plt.xlabel("Date")
plt.ylabel("NIFTY 50 index value")
plt.legend()
plt.grid()
plt.show()


# * From above it is clear that, ARMA model for this case is worst. 
# * The trend in time series is positive but the predicted ARMA model has negative trend.

# In[ ]:


# fit model
model = ARIMA(ts, order = (2, 1, 2))  
results_ARIMA = model.fit(disp = -1)

# predict 
yhat_train = results_ARIMA.predict(1, 2735)
predictions_ARIMA_diff_cumsum_train = yhat_train.cumsum() + nifty_train["Close"][0]

# nifty_train.shape[0] = 2735, nifty_test.shape[0] = 683
yhat_test = results_ARIMA.predict(2735, 2735 + 682)
predictions_ARIMA_diff_cumsum_test = yhat_test.cumsum() + nifty_test["Close"][0]
predictions_test = pd.DataFrame(predictions_ARIMA_diff_cumsum_test.values)
predictions_test.set_index(nifty_test.index, inplace = True)


# In[ ]:


# calculating RMSE 
train_rmse = mean_squared_error(nifty_train["Close"].values[1:], predictions_ARIMA_diff_cumsum_train.values, squared = False)
test_rmse = mean_squared_error(nifty_test["Close"].values, predictions_ARIMA_diff_cumsum_test.values, squared = False)


print("###################################################################")
print("ARIMA(2, 1, 2) Model")
print("train rmse:", train_rmse)
print("test rmse:", test_rmse)
print("###################################################################")

# visualization
plt.figure(figsize=(7.5, 5))
plt.plot(nifty_train["Close"], label = "train original")
plt.plot(nifty_test["Close"], label = "test original")
plt.plot(predictions_ARIMA_diff_cumsum_train, label = "train predicted")
plt.plot(predictions_test, label = "test predicted")
plt.title("ARIMA(2, 1, 2) Prediction")
plt.xlabel("Date")
plt.ylabel("NIFTY 50 index value")
plt.legend()
plt.grid()
plt.show()


# * Till now we considered only NIFTY 50 index values for prediction.
# * Now let us consider other variables like Gold prices, Crude oil prices, USD-INR exchange rate and other indices like NASDAQ composite, S&P 500 etc

# # Joining all csv files to form a single csv file

# In[ ]:


df = pd.merge(nifty_imputed, nifty_pe, on = "Date", how = "left")
df = pd.merge(df, crude_imputed, on = "Date", how = "outer")
df = pd.merge(df, gold_imputed, on = "Date", how = "outer")
df = pd.merge(df, euronext_imputed, on = "Date", how = "outer")
df = pd.merge(df, nasdaq, on = "Date", how = "outer")
df = pd.merge(df, sp500, on = "Date", how = "outer")
df = pd.merge(df, treasury_imputed, on = "Date", how = "outer")
df = pd.merge(df, usd_inr_imputed, on = "Date", how = "outer")
df.sort_values(by = ["Date"], inplace = True)

# adding indicator variable
df["is_nifty_pe_imputed"] = 0
df["crude is_holiday"] = 0
df["gold is_holiday"] = 0
df["euronext is_holiday"] = 0
df["nasdaq is_holiday"] = 0
df["sp500 is_holiday"] = 0
df["treasury is_holiday"] = 0
df["usd_inr is_holiday"] = 0


df.head()


# In[ ]:


df.tail()


# In[ ]:


# displaying NaN rows
na_rows(df)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


# splitting data into train, cross-validation and test
train_cv, test = time_based_train_test_split(df)
train, cv = time_based_train_test_split(train_cv, start_date = "2007-09-18", split_date = "2016-10-30", end_date = "2018-10-31")

train.shape, cv.shape, test.shape


# In[ ]:


train.tail()


# In[ ]:


test.head()


# In[ ]:


# We will not use raw "High", "Low" columns, so let's remove them.

train_cv.drop(columns = ["High", "Low", 
                         "crude high", "crude low", 
                         "gold high", "gold low", 
                         "euronext high", "euronext low", 
                         "nasdaq high", "nasdaq low", 
                         "sp500 high", "sp500 low", 
                         "treasury high", "treasury low", 
                         "usd_inr high", "usd_inr low"], 
              inplace = True)

train.drop(columns = ["High", "Low", 
                      "crude high", "crude low", 
                      "gold high", "gold low", 
                      "euronext high", "euronext low", 
                      "nasdaq high", "nasdaq low", 
                      "sp500 high", "sp500 low", 
                      "treasury high", "treasury low", 
                      "usd_inr high", "usd_inr low"], 
           inplace = True)

cv.drop(columns = ["High", "Low", 
                   "crude high", "crude low", 
                   "gold high", "gold low", 
                   "euronext high", "euronext low", 
                   "nasdaq high", "nasdaq low", 
                   "sp500 high", "sp500 low", 
                   "treasury high", "treasury low", 
                   "usd_inr high", "usd_inr low"], 
        inplace = True)

test.drop(columns = ["High", "Low", 
                     "crude high", "crude low", 
                     "gold high", "gold low", 
                     "euronext high", "euronext low", 
                     "nasdaq high", "nasdaq low", 
                     "sp500 high", "sp500 low", 
                     "treasury high", "treasury low", 
                     "usd_inr high", "usd_inr low"], 
          inplace = True)


# In[ ]:


train.head()


# In[ ]:


train_cv.sort_values(by = ["Date"], inplace = True)
cv.sort_values(by = ["Date"], inplace = True)
train.sort_values(by = ["Date"], inplace = True)
test.sort_values(by = ["Date"], inplace = True)

# displaying NaN rows
na_rows(train)


# * Our task is to predict NIFTY 50 index value based on other indicators.
# * On some days some stock market (other than NIFTY 50) will have a hoilday. Generally, previous day's market conditions of S&P 500 effect today's NIFTY 50 prices. Example: As today is holiday for S&P 500, tomorrow's NIFTY 50 prices doesn't have any effect. So, we will replace **Open**, **Close** values of S&P 500 with yesterday's **Open** values and **(H-L)*100/L** will be 0. We will update the **is_holiday** indicator value to 1.
# * On some days NIFTY 50 market is closed but other markets are open. Example, 01 May 2020 is holiday for NIFTY 50 and not for S&P 500. So, we will replace **Close** value of S&P 500 on 30 Apr 2020 with 01 May 2020 and also replace **(H-L)*100/L** value of 30 Apr 2020 with mean of 30 Apr 2020 and 01 May 2020. The reason for doing this is S&P 500 on 30 Apr 2020 opens after NIFTY 50 is closed and next NIFTY 50 opens on 02 May 2020 after S&P 500 closes on 01 May 2020.
# * Below two code cells is to test our assumption

# In[ ]:


data = {"Date": [dt.date(2007, 9, 17), dt.date(2007, 9, 18), dt.date(2007, 9, 19), dt.date(2007, 9, 20), dt.date(2007, 9, 21), dt.date(2007, 9, 24), dt.date(2007, 9, 25), dt.date(2007, 9, 26), dt.date(2007, 9, 27), dt.date(2007, 9, 28), dt.date(2007, 10, 1), dt.date(2007, 10, 2)], 
        "nifty" : [1,4,np.nan,9,12,15,16,19,np.nan,24,np.nan,27], 
        "sp500 open": [2,5,7,10,13,np.nan,17,20,22,np.nan,25,np.nan], 
        "sp500 close": [3,6,8,11,14,np.nan,18,21,23,np.nan,26,np.nan]}
d = pd.DataFrame(data, columns = ["Date", "nifty", "sp500 open", "sp500 close"])
d["Date"] = pd.to_datetime(d["Date"], format = "%Y-%m-%d")
d


# In[ ]:


d_array = np.array(d)
for i in range(d_array.shape[0]):
  if not np.isnan(d_array[i,1]) and np.isnan(d_array[i,2]): 
    d_array[i, 2] = d_array[i-1, 2]
    d_array[i, 3] = d_array[i-1, 2]
  elif np.isnan(d_array[i,1]) and not np.isnan(d_array[i,2]): 
    d_array[i-1, 3] = d_array[i, 3]
  else:
    continue
my_df = pd.DataFrame(d_array)
my_df.dropna(inplace = True)
my_df


# In[ ]:


def recursive_look_back(arr, row, col):
  '''
  This function looks back recursively in 
  a column and returns non NaN value
  '''
  if np.isnan(arr[row-1, col]):
     return recursive_look_back(arr, row-1, col)
  else:
    return arr[row-1, col]


def handle_nan(df):
  '''
  1. On some days some stock market (other than NIFTY 50) will have a hoilday. 
  Generally, previous day's market conditions of S&P 500 effect today's NIFTY 50 prices. 
  Example: As today is holiday for S&P 500, tomorrow's NIFTY 50 prices doesn't have any effect. 
  So, we will replace 'Open', 'Close' values of S&P 500 with yesterday's 'Open' values 
  and '(H-L)*100/L' will be 0. We will update the 'is_holiday' indicator value to 1.

  2. On some days NIFTY 50 market is closed but other markets are open. Example, 01 May 2020 
  is holiday for NIFTY 50 and not for S&P 500. So, we will replace 'Close' value of S&P 500 
  on 30 Apr 2020 with 01 May 2020 and also replace '(H-L)*100/L' value of 30 Apr 2020 with 
  mean of 30 Apr 2020 and 01 May 2020. The reason for doing this is S&P 500 on 30 Apr 2020 opens 
  after NIFTY 50 is closed and next NIFTY 50 opens on 02 May 2020 after S&P 500 closes on 01 May 2020.

  3. This function return values imputed with above two criteria. If NIFTY 50 data is not available for 
  a particular day then this will return nan values for NIFTY 50 values. Other columns may or may not be nan
  for this aparticular day. 

  4. Strict condition is that when NIFTY 50 data is available on particular day, then other values should not be nan


  0: Date	    1: Open	     2: Close	   3: (H-L)*100/L	
  4: PE_NIFTY50	  24: is_nifty_pe_imputed	
  5: crude open	  6: crude close	7: crude (H-L)*100/L	  26: crude is_holiday
  8: gold open  9: gold close   10: gold (H-L)*100/L	  27: gold is_holiday	
  11: euronext open	  12: euronext close  13: euronext (H-L)*100/L	  28: euronext is_holiday	
  14: nasdaq open	  15: nasdaq close	  16: nasdaq (H-L)*100/L	  29: nasdaq is_holiday	
  17: sp500 open	18: sp500 close	19: sp500 (H-L)*100/L	    30: sp500 is_holiday
  20: treasury open   21: treasury close	22: treasury (H-L)*100/L	31: treasury is_holiday
  23: usd_inr open	24: usd_inr close	25: usd_inr (H-L)*100/L   32: usd_inr is_holiday
  '''
  columns = list(df.columns)
  arr = np.array(df)
  for i in range(arr.shape[0]):
    
    # NIFTY 50 is open
    if not np.isnan(arr[i, 1]): 
      # NIFTY_PE values are nan
      if np.isnan(arr[i, 4]):
        arr[i, 4] = arr[i-1, 4]
        arr[i, 24] = 1
      # crude oil prices are nan
      if np.isnan(arr[i, 5]):
        previous_crude_open = recursive_look_back(arr, i, 5)
        arr[i, 5] = previous_crude_open
        arr[i, 6] = previous_crude_open
        arr[i, 7] = 0
        arr[i, 26] = 1
      # gold prices are nan
      if np.isnan(arr[i, 8]): 
        previous_gold_open = recursive_look_back(arr, i, 8)
        arr[i, 8] = previous_gold_open
        arr[i, 9] = previous_gold_open
        arr[i, 10] = 0
        arr[i, 27] = 1
      # euronext values are nan
      if np.isnan(arr[i, 11]): 
        previous_euronext_open = recursive_look_back(arr, i, 11)
        arr[i, 11] = previous_euronext_open
        arr[i, 12] = previous_euronext_open
        arr[i, 13] = 0
        arr[i, 28] = 1
      # nasdaq values are nan
      if np.isnan(arr[i, 14]): 
        previous_nasdaq_open = recursive_look_back(arr, i, 14)
        arr[i, 14] = previous_nasdaq_open
        arr[i, 15] = previous_nasdaq_open
        arr[i, 16] = 0
        arr[i, 29] = 1
      # sp500 values are nan
      if np.isnan(arr[i, 17]): 
        previous_sp500_open = recursive_look_back(arr, i, 17)
        arr[i, 17] = previous_sp500_open
        arr[i, 18] = previous_sp500_open
        arr[i, 19] = 0
        arr[i, 30] = 1
      # treasury values are nan
      if np.isnan(arr[i, 20]): 
        previous_treasury_open = recursive_look_back(arr, i, 20)
        arr[i, 20] = previous_treasury_open
        arr[i, 21] = previous_treasury_open
        arr[i, 22] = 0
        arr[i, 31] = 1
      # usd_inr values are nan
      if np.isnan(arr[i, 23]): 
        previous_usd_inr_open = recursive_look_back(arr, i, 23)
        arr[i, 23] = previous_usd_inr_open
        arr[i, 24] = previous_usd_inr_open
        arr[i, 25] = 0
        arr[i, 32] = 1
    
    # NIFTY 50 is holiday
    else: 
      # crude oil prices are not nan
      if not np.isnan(arr[i, 5]):
        arr[i-1, 6] = arr[i, 6]
        arr[i-1, 7] = (arr[i, 7] + arr[i-1, 7]) / 2
      # gold prices are not nan
      if not np.isnan(arr[i, 8]):
        arr[i-1, 9] = arr[i, 9]
        arr[i-1, 10] = (arr[i, 10] + arr[i-1, 10]) / 2
      # euronext values are not nan
      if not np.isnan(arr[i, 11]):
        arr[i-1, 11] = arr[i, 12]
        arr[i-1, 13] = (arr[i, 13] + arr[i-1, 13]) / 2
      # nasdaq values are not nan
      if not np.isnan(arr[i, 14]):
        arr[i-1, 15] = arr[i, 15]
        arr[i-1, 16] = (arr[i, 16] + arr[i-1, 16]) / 2
      # sp500 values are not nan
      if not np.isnan(arr[i, 17]):
        arr[i-1, 18] = arr[i, 18]
        arr[i-1, 19] = (arr[i, 19] + arr[i-1, 19]) / 2
      # treasury values are not nan
      if not np.isnan(arr[i, 20]):
        arr[i-1, 21] = arr[i, 21]
        arr[i-1, 22] = (arr[i, 22] + arr[i-1, 22]) / 2
      # usd_inr values are not nan
      if not np.isnan(arr[i, 23]):
        arr[i-1, 24] = arr[i, 24]
        arr[i-1, 25] = (arr[i, 25] + arr[i-1, 25]) / 2
  
  my_df = pd.DataFrame(arr, columns = columns)
  my_df.sort_values(by = ["Date"], inplace = True)
  return my_df


# In[ ]:


# handling NaN values in train, cv and test sets

imputed_train_cv = handle_nan(train_cv)
imputed_train = handle_nan(train)
imputed_cv = handle_nan(cv)
imputed_test = handle_nan(test)

imputed_train_cv.shape, imputed_train.shape, cv.shape, imputed_test.shape


# In[ ]:


# displaying top 50 NaN rows 

na_rows(imputed_train).head(50)


# In[ ]:


# Let us drop the rows where we don't have information of NIFTY 50

imputed_train.dropna(inplace = True)
imputed_test.dropna(inplace = True)
imputed_train_cv.dropna(inplace = True)
imputed_cv.dropna(inplace = True)


# * Common sense tells us that High value shoudl be greater than or equal to Low value. Let's test that condition

# In[ ]:


def high_low_test(s, text): 
  if (s < 0).any(): 
    print("High Value is less than Low Value. INCORRECT DATA for", text)
  else: 
    print("High value is greater than low value. CORRECR DATA for", text)


def high_low_test_util(df, df_name):
  high_low_test(df["(H-L)*100/L"], "nifty " + df_name)
  high_low_test(df["crude (H-L)*100/L"], "crude " + df_name)
  high_low_test(df["gold (H-L)*100/L"], "gold " + df_name)
  high_low_test(df["euronext (H-L)*100/L"], "euronext " + df_name)
  high_low_test(df["nasdaq (H-L)*100/L"], "nasdaq " + df_name)
  high_low_test(df["sp500 (H-L)*100/L"], "sp500 " + df_name)
  high_low_test(df["treasury (H-L)*100/L"], "treasury " + df_name)
  high_low_test(df["usd_inr (H-L)*100/L"], "usd_inr " + df_name)
  print()


# In[ ]:


# checking whether high value is greater than or equal to low value 
high_low_test_util(train_cv, "train_cv")
high_low_test_util(train, "train")
high_low_test_util(cv, "cv")
high_low_test_util(test, "test")


# * From above, it is clear that all High-Low data except crude train_cv gold train_cv, crude train and gold train are incorrect. So, we will replace the negative values with mean value of **(H-L)*100/L** column

# In[ ]:


# Reference: https://pythonexamples.org/pandas-dataframe-replace-values-in-column-based-on-condition/

imputed_train.loc[imputed_train["crude (H-L)*100/L"] < 0, "crude (H-L)*100/L"] = imputed_train["crude (H-L)*100/L"].mean()
imputed_train.loc[imputed_train["gold (H-L)*100/L"] < 0, "gold (H-L)*100/L"] = imputed_train["gold (H-L)*100/L"].mean()

imputed_train_cv.loc[imputed_train_cv["crude (H-L)*100/L"] < 0, "crude (H-L)*100/L"] = imputed_train_cv["crude (H-L)*100/L"].mean()
imputed_train_cv.loc[imputed_train_cv["gold (H-L)*100/L"] < 0, "gold (H-L)*100/L"] = imputed_train_cv["gold (H-L)*100/L"].mean()

high_low_test(imputed_train_cv["crude (H-L)*100/L"], "crude train_cv")
high_low_test(imputed_train_cv["gold (H-L)*100/L"], "gold train_cv")

high_low_test(imputed_train["crude (H-L)*100/L"], "crude train")
high_low_test(imputed_train["gold (H-L)*100/L"], "gold train")


# In[ ]:


imputed_train.head()


# * Let us compute other features namely **(C-O)*100/O** and **($X_{2}$-$X_{1}$)*100/$X_{1}$**
# 
# where 
# 
# $X_{2}$ = today's value
# 
# $X_{2}$ = yesterday's value
# 
# $X_{i}$ can be **Open** or **Close** values

# In[ ]:


def different_row_difference_feature(df, numerator, denominator, feature_name):
  '''
  This function computes percentage change in values of feature(s)
  '''
  df[feature_name] = ((df[numerator]/df[denominator].shift(1)) - 1) * 100
  return df

def add_features(df):
  '''
  (C-O)*100/O
  '''

  df = same_row_difference_feature(df = df, numerator = "Close", denominator = "Open", feature_name = "(C-O)*100/O")
  df = same_row_difference_feature(df = df, numerator = "crude close", denominator = "crude open", feature_name = "crude (C-O)*100/O")
  df = same_row_difference_feature(df = df, numerator = "gold close", denominator = "gold open", feature_name = "gold (C-O)*100/O")
  df = same_row_difference_feature(df = df, numerator = "euronext close", denominator = "euronext open", feature_name = "euronext (C-O)*100/O")
  df = same_row_difference_feature(df = df, numerator = "nasdaq close", denominator = "nasdaq open", feature_name = "nasdaq (C-O)*100/O")
  df = same_row_difference_feature(df = df, numerator = "sp500 close", denominator = "sp500 open", feature_name = "sp500 (C-O)*100/O")
  df = same_row_difference_feature(df = df, numerator = "treasury close", denominator = "treasury open", feature_name = "treasury (C-O)*100/O")
  df = same_row_difference_feature(df = df, numerator = "usd_inr close", denominator = "usd_inr open", feature_name = "usd_inr (C-O)*100/O")

  '''
  ( X2 - X1 )*100/ X1
  '''

  # nifty
  df = different_row_difference_feature(df = df, numerator = "Open", denominator = "Open", feature_name = "(O2-O1)*100/O1")
  df = different_row_difference_feature(df = df, numerator = "Close", denominator = "Close", feature_name = "(C2-C1)*100/C1")
  df = different_row_difference_feature(df = df, numerator = "Open", denominator = "Close", feature_name = "(O2-C1)*100/C1")

  # crude
  imputed_train = different_row_difference_feature(df = df, numerator = "crude open", denominator = "crude open", feature_name = "crude (O2-O1)*100/O1")
  imputed_train = different_row_difference_feature(df = df, numerator = "crude close", denominator = "crude close", feature_name = "crude (C2-C1)*100/C1")
  imputed_train = different_row_difference_feature(df = df, numerator = "crude open", denominator = "crude close", feature_name = "crude (O2-C1)*100/C1")

  # gold
  df = different_row_difference_feature(df = df, numerator = "gold open", denominator = "gold open", feature_name = "gold (O2-O1)*100/O1")
  imputed_train = different_row_difference_feature(df = df, numerator = "gold close", denominator = "gold close", feature_name = "gold (C2-C1)*100/C1")
  imputed_train = different_row_difference_feature(df = df, numerator = "gold open", denominator = "gold close", feature_name = "gold (O2-C1)*100/C1")

  # euronext
  imputed_train = different_row_difference_feature(df = df, numerator = "euronext open", denominator = "euronext open", feature_name = "euronext (O2-O1)*100/O1")
  df = different_row_difference_feature(df = df, numerator = "euronext close", denominator = "euronext close", feature_name = "euronext (C2-C1)*100/C1")
  df = different_row_difference_feature(df = df, numerator = "euronext open", denominator = "euronext close", feature_name = "euronext (O2-C1)*100/C1")

  # nasdaq
  df = different_row_difference_feature(df = df, numerator = "nasdaq open", denominator = "nasdaq open", feature_name = "nasdaq (O2-O1)*100/O1")
  df = different_row_difference_feature(df = df, numerator = "nasdaq close", denominator = "nasdaq close", feature_name = "nasdaq (C2-C1)*100/C1")
  df = different_row_difference_feature(df = df, numerator = "nasdaq open", denominator = "nasdaq close", feature_name = "nasdaq (O2-C1)*100/C1")

  # sp500
  df = different_row_difference_feature(df = df, numerator = "sp500 open", denominator = "sp500 open", feature_name = "sp500 (O2-O1)*100/O1")
  df = different_row_difference_feature(df = df, numerator = "sp500 close", denominator = "sp500 close", feature_name = "sp500 (C2-C1)*100/C1")
  df = different_row_difference_feature(df = df, numerator = "sp500 open", denominator = "sp500 close", feature_name = "sp500 (O2-C1)*100/C1")

  # treasury
  df = different_row_difference_feature(df = df, numerator = "treasury open", denominator = "treasury open", feature_name = "treasury (O2-O1)*100/O1")
  df = different_row_difference_feature(df = df, numerator = "treasury close", denominator = "treasury close", feature_name = "treasury (C2-C1)*100/C1")
  df = different_row_difference_feature(df = df, numerator = "treasury open", denominator = "treasury close", feature_name = "treasury (O2-C1)*100/C1")

  # usd_inr
  df = different_row_difference_feature(df = df, numerator = "usd_inr open", denominator = "usd_inr open", feature_name = "usd_inr (O2-O1)*100/O1")
  df = different_row_difference_feature(df = df, numerator = "usd_inr close", denominator = "usd_inr close", feature_name = "usd_inr (C2-C1)*100/C1")
  df = different_row_difference_feature(df = df, numerator = "usd_inr open", denominator = "usd_inr close", feature_name = "usd_inr (O2-C1)*100/C1")

  return df


# In[ ]:


# adding new features

imputed_train_cv = add_features(imputed_train_cv)
imputed_train = add_features(imputed_train)
imputed_cv = add_features(imputed_cv)
imputed_test = add_features(imputed_test)

imputed_train_cv.shape, imputed_train.shape, imputed_cv.shape, imputed_test.shape


# In[ ]:


imputed_train.head()


# In[ ]:


imputed_test.head()


# * If we look at 1st row, some cell values are **NaN** because we are taking difference by shifting. So, removing 1st row and saving the csv files
# * One important thing to note is that the features computed for a particular row corresponds to change in value for that particular day or features computed wrt previous day.

# In[ ]:


# displaying unique values in indicator variables

print(imputed_train_cv["is_nifty_pe_imputed"].value_counts(), "\n")
print(imputed_train_cv["crude is_holiday"].value_counts(), "\n")
print(imputed_train_cv["gold is_holiday"].value_counts(), "\n")
print(imputed_train_cv["euronext is_holiday"].value_counts(), "\n")
print(imputed_train_cv["nasdaq is_holiday"].value_counts(), "\n")
print(imputed_train_cv["sp500 is_holiday"].value_counts(), "\n")
print(imputed_train_cv["treasury is_holiday"].value_counts(), "\n")
print(imputed_train_cv["usd_inr is_holiday"].value_counts(), "\n")


# In[ ]:


# displaying unique values in indicator variables

print(imputed_test["is_nifty_pe_imputed"].value_counts(), "\n")
print(imputed_test["crude is_holiday"].value_counts(), "\n")
print(imputed_test["gold is_holiday"].value_counts(), "\n")
print(imputed_test["euronext is_holiday"].value_counts(), "\n")
print(imputed_test["nasdaq is_holiday"].value_counts(), "\n")
print(imputed_test["sp500 is_holiday"].value_counts(), "\n")
print(imputed_test["treasury is_holiday"].value_counts(), "\n")
print(imputed_test["usd_inr is_holiday"].value_counts(), "\n")


# In[ ]:


# usd_inr never had a holiday. So, removing it

imputed_train_cv.drop(columns = ["usd_inr is_holiday"], inplace = True)
imputed_train.drop(columns = ["usd_inr is_holiday"], inplace = True)
imputed_cv.drop(columns = ["usd_inr is_holiday"], inplace = True)
imputed_test.drop(columns = ["usd_inr is_holiday"], inplace = True)
imputed_train_cv.dropna(inplace = True)
imputed_train.dropna(inplace = True)
imputed_cv.dropna(inplace = True)
imputed_test.dropna(inplace = True)

imputed_train.head()


# In[ ]:


imputed_train_cv.sort_values(by = ["Date"], inplace = True)
imputed_cv.sort_values(by = ["Date"], inplace = True)
imputed_train.sort_values(by = ["Date"], inplace = True)
imputed_test.sort_values(by = ["Date"], inplace = True)

# saving the data
imputed_train_cv.to_csv(data_path + "final_train_cv_features.csv", index = False)
imputed_cv.to_csv(data_path + "final_cv_features.csv", index = False)
imputed_train.to_csv(data_path + "final_train_features.csv", index = False)
imputed_test.to_csv(data_path + "final_test_features.csv", index = False)


# # Multi-factor estimation of stock index movement: A case analysis of NIFTY 50, National Stock Exchange of India ([paper](https://ro.uow.edu.au/cgi/viewcontent.cgi?article=1009&context=dubaiwp))
# 
# ### The author in this paper used a linear relationship between NIFTY 50 and other factors (market cues) like crude oil, currency, bond market, Japanese and US stock indices movement to predict NIFTY 50 index value.
# 
# ### Below image shows the linear relationship the author assumed
# <img src='https://imgur.com/ug6c3UZ.png'>
# 
# ### In the similar way, let us try to predict present day's closing value given previous day's features

# In[9]:


# loading the data and adding output column
# output here will be next day's NIFTY 50 closing value

df_train_cv = pd.read_csv(data_path + "final_train_cv_features.csv", parse_dates = True)
df_train_cv["Date"] = pd.to_datetime(df_train_cv["Date"], format = "%Y-%m-%d")
df_train_cv["output"] = df_train_cv["Close"].shift(-1)
df_train_cv.sort_values(by = ["Date"], inplace = True)


df_cv = pd.read_csv(data_path + "final_cv_features.csv", parse_dates = True)
df_cv["Date"] = pd.to_datetime(df_cv["Date"], format = "%Y-%m-%d")
df_cv["output"] = df_cv["Close"].shift(-1)
df_cv.sort_values(by = ["Date"], inplace = True)


df_train = pd.read_csv(data_path + "final_train_features.csv", parse_dates = True)
df_train["Date"] = pd.to_datetime(df_train["Date"], format = "%Y-%m-%d")
df_train["output"] = df_train["Close"].shift(-1)
df_train.sort_values(by = ["Date"], inplace = True)


df_test = pd.read_csv(data_path + "final_test_features.csv", parse_dates = True)
df_test["Date"] = pd.to_datetime(df_test["Date"], format = "%Y-%m-%d")
df_test["output"] = df_test["Close"].shift(-1)
df_test.sort_values(by = ["Date"], inplace = True)

df_train.tail(2)


# In[10]:


# visualization

plt.figure()
plt.plot(df_train["Date"], df_train["Close"], label = "train")
plt.plot(df_cv["Date"], df_cv["Close"], label = "cv")
plt.plot(df_test["Date"], df_test["Close"], label = "test")
plt.legend()
plt.xlabel("Date")
plt.ylabel("index values")
plt.title("NIFTY 50")
plt.grid()
plt.show()


# In[ ]:


# saving only features needed for ML models

train_cv_features = df_train_cv.dropna().drop(columns = ["Open", "crude open", "gold open", "euronext open", "nasdaq open", "sp500 open", "treasury open", "usd_inr open", "is_nifty_pe_imputed",	"crude is_holiday",	"gold is_holiday",	"euronext is_holiday",	"nasdaq is_holiday",	"sp500 is_holiday",	"treasury is_holiday"])
train_features = df_train.dropna().drop(columns = ["Open", "crude open", "gold open", "euronext open", "nasdaq open", "sp500 open", "treasury open", "usd_inr open", "is_nifty_pe_imputed",	"crude is_holiday",	"gold is_holiday",	"euronext is_holiday",	"nasdaq is_holiday",	"sp500 is_holiday",	"treasury is_holiday"])
cv_features = df_cv.dropna().drop(columns = ["Open", "crude open", "gold open", "euronext open", "nasdaq open", "sp500 open", "treasury open", "usd_inr open", "is_nifty_pe_imputed",	"crude is_holiday",	"gold is_holiday",	"euronext is_holiday",	"nasdaq is_holiday",	"sp500 is_holiday",	"treasury is_holiday"])
test_features = df_test.dropna().drop(columns = ["Open", "crude open", "gold open", "euronext open", "nasdaq open", "sp500 open", "treasury open", "usd_inr open", "is_nifty_pe_imputed",	"crude is_holiday",	"gold is_holiday",	"euronext is_holiday",	"nasdaq is_holiday",	"sp500 is_holiday",	"treasury is_holiday"])


train_cv_features.to_csv("/mygdrive/CS1/data/train_cv_features_only.csv", index = False)
train_features.to_csv("/mygdrive/CS1/data/train_features_only.csv", index = False)
cv_features.to_csv("/mygdrive/CS1/data/cv_features_only.csv", index = False)
test_features.to_csv("/mygdrive/CS1/data/test_features_only.csv", index = False)


# In[11]:


# Let us remove "Open" columns and indicator variables 
# like is_nifty_pe_imputed, crude is_holiday etc

y_train_cv = np.array(df_train_cv["output"].dropna())
y_cv = np.array(df_cv["output"].dropna())
y_train = np.array(df_train["output"].dropna())
y_test = np.array(df_test["output"].dropna())

x_train_cv = np.array(df_train_cv.drop(columns = ["Date", "Open", "crude open", "gold open", "euronext open", "nasdaq open", "sp500 open", "treasury open", "usd_inr open", "is_nifty_pe_imputed",	"crude is_holiday",	"gold is_holiday",	"euronext is_holiday",	"nasdaq is_holiday",	"sp500 is_holiday",	"treasury is_holiday", "output"]))
x_cv = np.array(df_cv.drop(columns = ["Date", "Open", "crude open", "gold open", "euronext open", "nasdaq open", "sp500 open", "treasury open", "usd_inr open", "is_nifty_pe_imputed",	"crude is_holiday",	"gold is_holiday",	"euronext is_holiday",	"nasdaq is_holiday",	"sp500 is_holiday",	"treasury is_holiday",	"output"]))
x_train = np.array(df_train.drop(columns = ["Date", "Open", "crude open", "gold open", "euronext open", "nasdaq open", "sp500 open", "treasury open", "usd_inr open", "is_nifty_pe_imputed",	"crude is_holiday",	"gold is_holiday",	"euronext is_holiday",	"nasdaq is_holiday",	"sp500 is_holiday",	"treasury is_holiday",	"output"]))
x_test = np.array(df_test.drop(columns = ["Date", "Open", "crude open", "gold open", "euronext open", "nasdaq open", "sp500 open", "treasury open", "usd_inr open", "is_nifty_pe_imputed",	"crude is_holiday",	"gold is_holiday",	"euronext is_holiday",	"nasdaq is_holiday",	"sp500 is_holiday",	"treasury is_holiday",	"output"]))

# removing last row as last row does not have future value
x_train_cv = x_train_cv[:-1]
x_cv = x_cv[:-1]
x_train = x_train[:-1]
x_test = x_test[:-1]

x_train_cv.shape, x_cv.shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape, 


# In[12]:


# scaling the data

x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_scaler.fit(x_train)
x_train_cv_scaled = x_scaler.transform(x_train_cv)
x_train_scaled = x_scaler.transform(x_train)
x_cv_scaled = x_scaler.transform(x_cv)
x_test_scaled = x_scaler.transform(x_test)

y_scaler.fit(y_train.reshape(-1, 1))
y_train_cv_scaled = y_scaler.transform(y_train_cv.reshape(-1, 1))
y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1))
y_cv_scaled = y_scaler.transform(y_cv.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

x_train_cv_scaled.shape, x_cv_scaled.shape, x_train_scaled.shape, y_train_scaled.shape, x_test_scaled.shape, y_test_scaled.shape


# In[ ]:


with open("/mygdrive/CS1/x_scaler.pkl", "wb") as f: 
  pickle.dump(x_scaler, f)

with open("/mygdrive/CS1/y_scaler.pkl", "wb") as f: 
  pickle.dump(y_scaler, f)


# ### Linear Regression

# In[13]:


# fit the model
model = LinearRegression()
model.fit(x_train_cv_scaled, y_train_cv_scaled)

# prediction
y_train_cv_pred = model.predict(x_train_cv_scaled)
y_test_pred = model.predict(x_test_scaled)

# calculating RMSE
train_rmse = mean_squared_error(y_train_cv, y_scaler.inverse_transform(y_train_cv_pred), squared = False)
test_rmse = mean_squared_error(y_test, y_scaler.inverse_transform(y_test_pred), squared = False)


print("###################################################################")
print("Linear Regression Model")
print("train rmse:", train_rmse)
print("test rmse:", test_rmse)
print("###################################################################")

# visualization
plt.figure()
plt.plot(df_train_cv.iloc[:-1, 0], y_train_cv, "yo-", label = "train original")
plt.plot(df_test.iloc[:-1, 0], y_test, "go-", label = "test original")
plt.plot(df_train_cv.iloc[:-1, 0], y_scaler.inverse_transform(y_train_cv_pred), "r", label = "train original")
plt.plot(df_test.iloc[:-1, 0], y_scaler.inverse_transform(y_test_pred), "pink", label = "test original")
plt.legend()
plt.xlabel("Date")
plt.ylabel("NIFTY 50 index value")
plt.title("Linear Regression Model ")
plt.grid()
plt.show()


# ### Linear Regression with L1 regularization

# In[ ]:



params = {"alpha" : [.0001, .00033, .001, .0033, .01, .033, .1, .33, 1, 3.3, 10, 33, 100]}

train_rmse, cv_rmse = [], []
# hyperparameter tuning
for i in params["alpha"]:
  model = Lasso(alpha = i, max_iter = 2000)
  model.fit(x_train_scaled, y_train_scaled)
  y_train_pred = model.predict(x_train_scaled)
  y_cv_pred = model.predict(x_cv_scaled)
  train_rmse.append(mean_squared_error(y_train, y_scaler.inverse_transform(y_train_pred), squared = False))
  cv_rmse.append(mean_squared_error(y_cv, y_scaler.inverse_transform(y_cv_pred), squared = False))

# visualization
plt.figure()
plt.plot(params["alpha"], train_rmse, "bo-", label = "train")
plt.plot(params["alpha"], cv_rmse, "ro-", label = "cv")
plt.legend()
plt.xscale("log")
plt.xlabel("Hyperparameter: alpha")
plt.ylabel("RMSE")
plt.title("RMSE vs alpha")
plt.grid()
plt.show()


# In[ ]:


alpha = 0.001


# In[ ]:


# fitting the model
model = Lasso(alpha = alpha, max_iter = 2000)
model.fit(x_train_cv_scaled, y_train_cv_scaled)

# prediction
y_train_cv_pred = model.predict(x_train_cv_scaled)
y_test_pred = model.predict(x_test_scaled)

# calculating RMSE
train_rmse = mean_squared_error(y_train_cv, y_scaler.inverse_transform(y_train_cv_pred), squared = False)
test_rmse = mean_squared_error(y_test, y_scaler.inverse_transform(y_test_pred), squared = False)


print("###################################################################")
print("Linear Regression Model with L1 Regularization")
print("train rmse:", train_rmse)
print("test rmse:", test_rmse)
print("###################################################################")

# visualization
plt.figure()
plt.plot(df_train_cv.iloc[:-1, 0], y_train_cv, "yo-", label = "train original")
plt.plot(df_test.iloc[:-1, 0], y_test, "go-", label = "test original")
plt.plot(df_train_cv.iloc[:-1, 0], y_scaler.inverse_transform(y_train_cv_pred), "r", label = "train original")
plt.plot(df_test.iloc[:-1, 0], y_scaler.inverse_transform(y_test_pred), "pink", label = "test original")
plt.legend()
plt.xlabel("Date")
plt.ylabel("NIFTY 50 index value")
plt.title("Linear Regression Model with L1 Regularization")
plt.grid()
plt.show()


# In[ ]:


with open("/mygdrive/CS1/LR_L1_reg.pkl", "wb") as f:  
    pickle.dump(model, f)


# ### Linear Regression with L2 regularization

# In[ ]:


params = {"alpha" : [.00001, .000033, .0001, .00033, .001, .0033, .01, .033, .1, .33, 1, 3.3, 10, 33, 100]}

train_rmse, cv_rmse = [], []

# hyperparameter tuning
for i in params["alpha"]:
  model = Ridge(alpha = i)
  model.fit(x_train_scaled, y_train_scaled)
  y_train_pred = model.predict(x_train_scaled)
  y_cv_pred = model.predict(x_cv_scaled)
  train_rmse.append(mean_squared_error(y_train, y_scaler.inverse_transform(y_train_pred), squared = False))
  cv_rmse.append(mean_squared_error(y_cv, y_scaler.inverse_transform(y_cv_pred), squared = False))

# visualization
plt.figure()
plt.plot(params["alpha"], train_rmse, "bo-", label = "train")
plt.plot(params["alpha"], cv_rmse, "ro-", label = "cv")
plt.legend()
plt.xscale("log")
plt.xlabel("Hyperparameter: alpha")
plt.ylabel("RMSE")
plt.title("RMSE vs alpha")
plt.grid()
plt.show()


# In[ ]:


alpha = 0.033


# In[ ]:


# fitting model
model = Ridge(alpha = alpha)
model.fit(x_train_cv_scaled, y_train_cv_scaled)

# prediction
y_train_cv_pred = model.predict(x_train_cv_scaled)
y_test_pred = model.predict(x_test_scaled)

# calculating RMSR
train_rmse = mean_squared_error(y_train_cv, y_scaler.inverse_transform(y_train_cv_pred), squared = False)
test_rmse = mean_squared_error(y_test, y_scaler.inverse_transform(y_test_pred), squared = False)


print("###################################################################")
print("Linear Regression Model with L2 Regularization")
print("train rmse:", train_rmse)
print("test rmse:", test_rmse)
print("###################################################################")

# visualization
plt.figure()
plt.plot(df_train_cv.iloc[:-1, 0], y_train_cv, "yo-", label = "train original")
plt.plot(df_test.iloc[:-1, 0], y_test, "go-", label = "test original")
plt.plot(df_train_cv.iloc[:-1, 0], y_scaler.inverse_transform(y_train_cv_pred), "r", label = "train original")
plt.plot(df_test.iloc[:-1, 0], y_scaler.inverse_transform(y_test_pred), "pink", label = "test original")
plt.legend()
plt.xlabel("Date")
plt.ylabel("NIFTY 50 index value")
plt.title("Linear Regression Model with L2 Regularization")
plt.grid()
plt.show()


# ### ElasticNet

# In[26]:


params = {"alpha" : [.0001, .00033, .001, .0033, .01, .033, .1, .33, 1, 3.3, 10, 33, 100, 330, 1000], 
          "l1_ratio" : [.1, .2, .3, .4, .5, .6, .7, .8, .9]}

lol = []

# hyperparameter tuning
for c in tqdm(params["alpha"]):
  for k in params["l1_ratio"]:
    model = ElasticNet(alpha = c, l1_ratio = k)
    model.fit(x_train_scaled, y_train_scaled)
    y_train_pred = model.predict(x_train_scaled)
    y_cv_pred = model.predict(x_cv_scaled)
    lol.append([c, 
                k, 
                mean_squared_error(y_train, y_scaler.inverse_transform(y_train_pred), squared = False), 
                mean_squared_error(y_cv, y_scaler.inverse_transform(y_cv_pred), squared = False)])


rmse = pd.DataFrame(lol, columns = ["alpha", "l1_ratio", "train_rmse", "cv_rmse"])
rmse.head()


# In[27]:


# visualization
hmap = rmse.pivot("alpha", "l1_ratio", "train_rmse")
plt.figure(figsize = (20, 10))
sns.heatmap(hmap, linewidth = 1, annot = True, fmt="f")
plt.ylabel("alpha")
plt.xlabel("l1_ratio")
plt.title("train rmse in heatmap")
plt.show()


# In[28]:


# visualization
hmap = rmse.pivot("alpha", "l1_ratio", "cv_rmse")
plt.figure(figsize = (20, 10))
sns.heatmap(hmap, linewidth = 1, annot = True, fmt="f")
plt.ylabel("alpha")
plt.xlabel("l1_ratio")
plt.title("cv rmse in heatmap")
plt.show()


# In[61]:


alpha = .001
l1_ratio = .9


# In[63]:


# fitting model
model = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)
model.fit(x_train_cv_scaled, y_train_cv_scaled)

# prediction
y_train_cv_pred = model.predict(x_train_cv_scaled)
y_test_pred = model.predict(x_test_scaled)

# calcaulating RMSE
train_rmse = mean_squared_error(y_train_cv, y_scaler.inverse_transform(y_train_cv_pred), squared = False)
test_rmse = mean_squared_error(y_test, y_scaler.inverse_transform(y_test_pred), squared = False)


print("###################################################################")
print("Linear Regression Model with L1 and L2 Regularization")
print("train rmse:", train_rmse)
print("test rmse:", test_rmse)
print("###################################################################")

# visualization
plt.figure()
plt.plot(df_train_cv.iloc[:-1, 0], y_train_cv, "yo-", label = "train original")
plt.plot(df_test.iloc[:-1, 0], y_test, "go-", label = "test original")
plt.plot(df_train_cv.iloc[:-1, 0], y_scaler.inverse_transform(y_train_cv_pred), "r", label = "train original")
plt.plot(df_test.iloc[:-1, 0], y_scaler.inverse_transform(y_test_pred), "pink", label = "test original")
plt.legend()
plt.xlabel("Date")
plt.ylabel("NIFTY 50 index value")
plt.title("Linear Regression Model with L1 and L2 Regularization")
plt.grid()
plt.show()


# ### Support Vector Regressor

# In[ ]:


model = SVR()

params = {"C" : [.00001, .000033, .0001, .00033, .001, .0033, .01, .033, .1, .33, 1, 3.3, 10, 33, 100, 330, 1000], 
          "kernel" : ["linear", "poly", "rbf", "sigmoid"]}

lol = []

# hyperparameter tuning
for c in tqdm(params["C"]):
  for k in params["kernel"]:
    model = SVR(C = c, kernel = k)
    model.fit(x_train_scaled, y_train_scaled.ravel())
    y_train_pred = model.predict(x_train_scaled)
    y_cv_pred = model.predict(x_cv_scaled)
    lol.append([c, 
                k, 
                mean_squared_error(y_train, y_scaler.inverse_transform(y_train_pred), squared = False), 
                mean_squared_error(y_cv, y_scaler.inverse_transform(y_cv_pred), squared = False)])


rmse = pd.DataFrame(lol, columns = ["C", "kernel", "train_rmse", "cv_rmse"])
rmse.head()


# In[ ]:


# visualization
hmap = rmse.pivot("C", "kernel", "train_rmse")
plt.figure(figsize = (10, 10))
sns.heatmap(hmap, linewidth = 1, annot = True, fmt="f")
plt.ylabel("C")
plt.xlabel("kernel")
plt.title("train rmse in heatmap")
plt.show()


# In[ ]:


# visualization
hmap = rmse.pivot("C", "kernel", "cv_rmse")
plt.figure(figsize = (10, 10))
sns.heatmap(hmap, linewidth = 1, annot = True, fmt="f")
plt.ylabel("C")
plt.xlabel("kernel")
plt.title("cv rmse in heatmap")
plt.show()


# In[ ]:


C = 3.3
kernel = "linear"


# In[ ]:


# fitting model
model = SVR(C = C, kernel = kernel)
model.fit(x_train_cv_scaled, y_train_cv_scaled.ravel())

# prediction
y_train_cv_pred = model.predict(x_train_cv_scaled)
y_test_pred = model.predict(x_test_scaled)

# calcaulating RMSE
train_rmse = mean_squared_error(y_train_cv, y_scaler.inverse_transform(y_train_cv_pred), squared = False)
test_rmse = mean_squared_error(y_test, y_scaler.inverse_transform(y_test_pred), squared = False)


print("###################################################################")
print("Support Vector Regression Model")
print("train rmse:", train_rmse)
print("test rmse:", test_rmse)
print("###################################################################")

# visualization
plt.figure()
plt.plot(df_train_cv.iloc[:-1, 0], y_train_cv, "yo-", label = "train original")
plt.plot(df_test.iloc[:-1, 0], y_test, "go-", label = "test original")
plt.plot(df_train_cv.iloc[:-1, 0], y_scaler.inverse_transform(y_train_cv_pred), "r", label = "train original")
plt.plot(df_test.iloc[:-1, 0], y_scaler.inverse_transform(y_test_pred), "pink", label = "test original")
plt.legend()
plt.xlabel("Date")
plt.ylabel("NIFTY 50 index value")
plt.title("Support Vector Regression Model")
plt.grid()
plt.show()


# # LSTM 

# In[ ]:


# Reference: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

def prepare_data(sequence, steps):
  '''
  This function takes a sequence and 
  prepapres data to feed into lstm
  '''
  x, y = [], []
  for i in range(len(sequence)):
    end_index = i + steps
    if end_index > len(sequence) - 1:
      break
    seq_x, seq_y = sequence[i:end_index], sequence[end_index]
    x.append(seq_x)
    y.append(seq_y)
  return np.array(x), np.array(y)


# In[ ]:


# loading data

df_train = pd.read_csv(data_path + "final_train_features.csv", parse_dates = True)
df_train["Date"] = pd.to_datetime(df_train["Date"], format = "%Y-%m-%d")
df_train.sort_values(by = ["Date"], inplace = True)

df_cv = pd.read_csv(data_path + "final_cv_features.csv", parse_dates = True)
df_cv["Date"] = pd.to_datetime(df_cv["Date"], format = "%Y-%m-%d")
df_cv.sort_values(by = ["Date"], inplace = True)

df_test = pd.read_csv(data_path + "final_test_features.csv", parse_dates = True)
df_test["Date"] = pd.to_datetime(df_test["Date"], format = "%Y-%m-%d")
df_test.sort_values(by = ["Date"], inplace = True)

df_train.shape, df_cv.shape, df_test.shape


# In[ ]:


scaler = StandardScaler()

# scaling data
scaler.fit(df_train["Close"].values.reshape(-1, 1))
train_scaled = scaler.transform(df_train["Close"].values.reshape(-1, 1))
cv_scaled = scaler.transform(df_cv["Close"].values.reshape(-1, 1))
test_scaled = scaler.transform(df_test["Close"].values.reshape(-1, 1))

train_scaled.shape, cv_scaled.shape, test_scaled.shape


# In[ ]:


steps = 180

# preparing the data
x_train, y_train = prepare_data(train_scaled, steps)
x_cv, y_cv = prepare_data(cv_scaled, steps)
x_test, y_test = prepare_data(test_scaled, steps)

x_train.shape, y_train.shape, x_cv.shape, y_cv.shape, x_test.shape, y_test.shape


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


def checkpoint_path(i): 
  '''
  This functions returns the path to store model weights
  '''
  return "/mygdrive/CS1/model_" + str(i) + "/weights_steps_" + str(steps) + ".hdf5"

def log_dir(i): 
  '''
  This function return the path to store tensorboard logs
  '''
  return "/mygdrive/CS1/logs/fit" + str(i) + "/" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

# callbacks 
earlystop = EarlyStopping(monitor = "val_loss", 
                          patience = 15, 
                          verbose = 1, 
                          mode = "min", 
                          restore_best_weights = True)

reduce_lr = ReduceLROnPlateau(monitor = "val_loss", 
                              factor = .4642, 
                              patience = 3, 
                              verbose = 1, 
                              min_lr = 1.0e-5, 
                              mode = "min")


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir /mygdrive/CS1/logs/fit1')


# * epoch loss
# 
# <img src='https://imgur.com/ZszQ8I1.png'>

# In[ ]:


# creating LSTM model
tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(1024, activation = "tanh", kernel_initializer = "glorot_uniform", input_shape = (None, 1)))
model.add(Dense(1024))
model.add(Dense(1))
model.compile(optimizer = "adam", loss = "mse")
model.summary()


# In[ ]:


model_number = 1

# callbacks
checkpoint = ModelCheckpoint(filepath = checkpoint_path(model_number), 
                             monitor = "val_loss", 
                             verbose = 1, 
                             save_best_only = True, 
                             mode = "min")

tensorboard_callback = TensorBoard(log_dir = log_dir(model_number))

callbacks_list = [tensorboard_callback, checkpoint, earlystop, reduce_lr]

# fitting the model
history = model.fit(x_train, y_train, 
                    epochs = 50, 
                    batch_size = 16, 
                    validation_data = (x_cv, y_cv), 
                    callbacks = callbacks_list)



# prediction
y_test_pred_scaled = model.predict(x_test)
y_test_pred = scaler.inverse_transform(y_test_pred_scaled)

# calculating RMSE
test_rmse = mean_squared_error(scaler.inverse_transform(y_test), y_test_pred, squared = False)

print("###################################################################")
print("LSTM Model")
print("test rmse:", test_rmse)
print("###################################################################")

# visualization
plt.figure()
plt.plot(df_train["Date"], df_train["Close"], label = "train")
plt.plot(df_cv["Date"], df_cv["Close"], label = "cv")
plt.plot(df_test["Date"], df_test["Close"], label = "test")
plt.plot(df_test.iloc[steps : ]["Date"], y_test_pred.reshape(y_test_pred.shape[0]), label = "prediction")
plt.legend()
plt.xlabel("Date")
plt.ylabel("NIFTY 50 index value")
plt.title("LSTM model")
plt.show()


# In[ ]:


def prepare_df(df, steps): 
  '''
  This function takes a dataframe and 
  prepares it to feed to LSTM model with extra features
  '''
  df.rename(columns = {"Close" : "t"}, inplace = True)
  original_cols = list(df.columns)
  cols_order = original_cols[:2]
  for i in range(steps-1, 0, -1):
    col_name = "t-" + str(i)
    cols_order.append(col_name)
    df[col_name] = df["t"].shift(i)
  df["t+1"] = df["t"].shift(-1)
  cols_order.extend(original_cols[2:])
  cols_order.append("t+1")
  df = df[cols_order]
  df.dropna(inplace = True)
  return df.drop(columns= ["Open", "crude open", "crude close", "gold open", "gold close", "euronext open", "euronext close", "nasdaq open", "nasdaq close", "sp500 open", "sp500 close", "treasury open", "treasury close", "usd_inr open", "usd_inr close"])


# In[ ]:


# loading data

train = pd.read_csv(data_path + "final_train_features.csv", parse_dates = True)
train["Date"] = pd.to_datetime(train["Date"], format = "%Y-%m-%d")
train.sort_values(by = ["Date"], inplace = True)

cv = pd.read_csv(data_path + "final_cv_features.csv", parse_dates = True)
cv["Date"] = pd.to_datetime(cv["Date"], format = "%Y-%m-%d")
cv.sort_values(by = ["Date"], inplace = True)

test = pd.read_csv(data_path + "final_test_features.csv", parse_dates = True)
test["Date"] = pd.to_datetime(test["Date"], format = "%Y-%m-%d")
test.sort_values(by = ["Date"], inplace = True)

train.shape, cv.shape, test.shape


# In[ ]:


scaler1 = StandardScaler()

# scaling the sequence data
scaler1.fit(train["Close"].values.reshape(-1, 1))
train["Close"] = scaler1.transform(train["Close"].values.reshape(-1, 1))
cv["Close"] = scaler1.transform(cv["Close"].values.reshape(-1, 1))
test["Close"] = scaler1.transform(test["Close"].values.reshape(-1, 1))

steps = 180

# preparing the dataframe
df_train = prepare_df(train.copy(deep = True), steps = steps)
df_cv = prepare_df(cv.copy(deep = True), steps = steps)
df_test = prepare_df(test.copy(deep = True), steps = steps)

df_train.shape, df_cv.shape, df_test.shape


# In[ ]:


df_train.head()


# In[ ]:


train.head()


# In[ ]:


# extracting sequence data from dataframe
x_train_sequence = df_train.values[:, 1:steps+1]
x_train_sequence = np.asarray(np.reshape(x_train_sequence, (x_train_sequence.shape[0], x_train_sequence.shape[1], 1))).astype(np.float32)
x_cv_sequence = df_cv.values[:, 1:steps+1]
x_cv_sequence = np.asarray(np.reshape(x_cv_sequence, (x_cv_sequence.shape[0], x_cv_sequence.shape[1], 1))).astype(np.float32)
x_test_sequence = df_test.values[:, 1:steps+1]
x_test_sequence = np.asarray(np.reshape(x_test_sequence, (x_test_sequence.shape[0], x_test_sequence.shape[1], 1))).astype(np.float32)

# scaling features other than sequence data
# as sequence data is already scaled
scaler2 = StandardScaler()
scaler2.fit(df_train.values[:, steps+1 : -1])
x_train_extracted_features = np.asarray(scaler2.transform(df_train.values[:, steps+1 : -1])).astype(np.float32)
x_cv_extracted_features = np.asarray(scaler2.transform(df_cv.values[:, steps+1 : -1])).astype(np.float32)
x_test_extracted_features = np.asarray(scaler2.transform(df_test.values[:, steps+1 : -1])).astype(np.float32)

# extracting output variable from dataframe
y_train = np.asarray(df_train.values[:, -1].reshape(-1, 1)).astype(np.float32)
y_cv = np.asarray(df_cv.values[:, -1].reshape(-1, 1)).astype(np.float32)
y_test = np.asarray(df_test.values[:, -1].reshape(-1, 1)).astype(np.float32)

x_train_sequence.shape, x_train_extracted_features.shape, y_train.shape, x_cv_sequence.shape, x_cv_extracted_features.shape, y_cv.shape, x_test_sequence.shape, x_test_extracted_features.shape, y_test.shape


# In[ ]:


def model_2():
  tf.keras.backend.clear_session()
  sequence_input = Input(shape = (steps, 1), name = "sequence")
  # LSTM layer
  lstm = LSTM(1024, activation = "tanh", kernel_initializer = "glorot_uniform")(sequence_input)
  flatten1 = Flatten()(lstm)
  features_input = Input(shape = (x_train_extracted_features.shape[1], ), name = "features")
  # dense layer for features other than sequence
  dense1 = Dense(32, activation = "relu", kernel_initializer = "he_uniform")(features_input)
  # concatenation
  concat = Concatenate()([flatten1, dense1])
  dense2 = Dense(1024)(concat)
  # output
  output = Dense(1)(dense2)
  model = Model(inputs = [sequence_input, features_input], outputs = output)
  # compile the model
  model.compile(optimizer = "adam", loss = "mse")
  # saving the model diagram
  plot_model(model, to_file = "./model.png", show_shapes = True)
  return model


# In[ ]:


# getting the model and displaying summary

model2 = model_2()
model2.summary()


# In[ ]:


# displaying model architecture with shapes

cv2_imshow(cv2.imread("./model.png"))


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir /mygdrive/CS1/logs/fit2')


# In[ ]:


model_number = 2

# callbacks
checkpoint = ModelCheckpoint(filepath = checkpoint_path(model_number), 
                             monitor = "val_loss", 
                             verbose = 1, 
                             save_best_only = True, 
                             mode = "min")

tensorboard_callback = TensorBoard(log_dir = log_dir(model_number))

callbacks_list = [tensorboard_callback, checkpoint, earlystop, reduce_lr]

# data to be fed into multi-input model 
x_train = {"sequence" : x_train_sequence, "features" : x_train_extracted_features}
x_cv = {"sequence" : x_cv_sequence, "features" : x_cv_extracted_features}
x_test = {"sequence" : x_test_sequence, "features" : x_test_extracted_features}

# fitting model
history = model2.fit(x_train, y_train, 
                     epochs = 50, 
                     validation_data = (x_cv, y_cv), 
                     callbacks = callbacks_list)

# prediction
y_test_pred_scaled = model2.predict(x_test)
y_test_pred = scaler1.inverse_transform(y_test_pred_scaled)

# calculating RMSE
test_rmse = mean_squared_error(scaler1.inverse_transform(y_test), y_test_pred, squared = False)

print("###################################################################")
print("LSTM Model with other features")
print("test rmse:", test_rmse)
print("###################################################################")

# visualization
plt.figure()
plt.plot(train["Date"], scaler1.inverse_transform(train["Close"]), label = "train")
plt.plot(cv["Date"], scaler1.inverse_transform(cv["Close"]), label = "cv")
plt.plot(test["Date"], scaler1.inverse_transform(test["Close"]), label = "test")
plt.plot(df_test["Date"], y_test_pred.reshape(y_test_pred.shape[0]), label = "prediction")
plt.legend()
plt.xlabel("Date")
plt.ylabel("NIFTY 50 index value")
plt.title("LSTM model with other features")
plt.grid()
plt.show()


# * epoch loss
# 
# <img src='https://imgur.com/KU6Nf6i.png'>

# # Results

# In[5]:


# tabulating results
from prettytable import PrettyTable
p = PrettyTable(["Model", "test RMSE"])
p.add_row(["Baseline Model", "1863.719"])
p.add_row(["Simple Moving Average", "150.375"])
p.add_row(["Exponential Moving Average Model", "149.114"])
p.add_row(["ARIMA(2, 1, 2) Model", "1875.698"])
p.add_row(["Linear Regression Model", "149.809"])
p.add_row(["Linear Regression Model with L1 Regularization", "144.816"])
p.add_row(["Linear Regression Model with L2 Regularization", "149.603"])
p.add_row(["Linear Regression Model with L1 & L2 Regularization", "144.851"])
p.add_row(["Support Vector Regressor (linear)", "186.138"])
p.add_row(["LSTM", "168.314"])
p.add_row(["LSTM with other features", "166.954"])

print(p)


# # Error Analysis

# In[ ]:


df_test = pd.read_csv(data_path + "final_test_features.csv", parse_dates = True)
df_test["Date"] = pd.to_datetime(df_test["Date"], format = "%Y-%m-%d")
df_test["output"] = df_test["Close"].shift(-1)
df_test.dropna(inplace = True)
df_test.drop(columns = ["Open", "crude open", "gold open", "euronext open", "nasdaq open", "sp500 open", "treasury open", "usd_inr open", "is_nifty_pe_imputed",	"crude is_holiday",	"gold is_holiday",	"euronext is_holiday",	"nasdaq is_holiday",	"sp500 is_holiday",	"treasury is_holiday"], inplace = True)
df_test.sort_values(by = ["Date"], inplace = True)

df_test.head(2)


# In[ ]:


with open("/mygdrive/CS1/LR_L1_reg.pkl", "rb") as f: 
  model = pickle.load(f)

with open("/mygdrive/CS1/x_scaler.pkl", "rb") as f: 
  x_scaler = pickle.load(f)

with open("/mygdrive/CS1/y_scaler.pkl", "rb") as f: 
  y_scaler = pickle.load(f)


# In[ ]:


x_test = x_scaler.transform(df_test.values[:, 1 : -1])
y_pred = y_scaler.inverse_transform(model.predict(x_test))

df_test["prediction"] = y_pred

df_test.head(2)


# In[ ]:


df_test = same_row_difference_feature(df_test, "prediction", "output", "error", absolute = True)
df_test.head(2)


# In[ ]:


# reference: https://stackoverflow.com/a/45925049

fig, ax1 = plt.subplots(figsize = (15, 7.5))

ax2 = ax1.twinx()

ax1.set_xlabel("Date")
ax1.set_ylabel("NIFTY 50 Index value")
ax2.set_ylabel("absolute percentage error")

p11, = ax1.plot(df_test["Date"], df_test["output"], "m", label = "original")
p12, = ax1.plot(df_test["Date"], df_test["prediction"], "r", label = "prediction")
p21, = ax2.plot(df_test["Date"], df_test["error"], "y", label = "error")

lns = [p11, p12, p21]
ax1.legend(handles = lns, loc = "best")
plt.grid()
plt.show()


# * From above plot we can see that whenever there is steep rise or fall in index value, then absolute percentage errors is more. 
# * Example: We can see that absolute percentage error shot upto 14% around March 2020 due to Covid uncertanity.

# In[ ]:


denominator = df_test.shape[0]
for i in [2, 4, 6, 8, 10, 12]: 
  print("Percentage of predictions with absolute percentage error greater than {} is {}".format(i, np.round(100 * df_test[df_test["error"] > i].shape[0] / denominator, 2)))


# # The best model, i.e., linear regression with L1 regularization is deployed on the Heroku platform. [Link](https://nifty-50-prediction.herokuapp.com/) to web-app. Video [link](https://vimeo.com/manage/videos/600210825) for sample predictions.

# # Conclusion

# Stock prices are extremely volatile and highly non-linear. Accurate prediction of stock prediction is an extremely difficult task. As India is an emerging economy, global events impact the Indian stock market. In this Case Study, we tried to incorporate global event factors to predict NIFTY 50 closing value. Experiments are done with different algorithms. Linear Regression with L1 regularization gave the lowest RMSE. One important thing to note is in the Simple Moving Average Model, we did the prediction that yesterday's closing value of NIFTY 50 is today's closing value. With this, we got an RMSE of 150.375 but using Linear Regression with L1 regularization we got an RMSE of 144. 816. This proves that predicting stock prices is an extremely difficult task. We did not consider all the variables that impact NIFTY 50, we did consider only a few and we could improve from the SMA model. With the multi-factor approach, we could slightly decrease the RMSE of the LSTM model from 168.314 to 166.954. This also proves that other indices affect Indian indices.

# # Future Work

# * Other indices like [Nikkei 225](https://quote.jpx.co.jp/jpx/template/quote.cgi?F=tmp/e_real_index2&QCODE=101) of [Tokyo Stock Exchange](https://www.jpx.co.jp/english/), [Straits Times Index](https://www.sgx.com/indices/products/sti) of [Singapore Stock Exchange](https://www.sgx.com/), etc can be used to decrease RMSE.
# * In short term, the market is influenced by news. Polarity of news sentiment can be incorporated as a feature to predict index value and decrease RMSE.
# * Deep learning models are data-hungry. More data should be collected so as to train the complex DL model.

# In[ ]:





# In[ ]:





# In[ ]:



