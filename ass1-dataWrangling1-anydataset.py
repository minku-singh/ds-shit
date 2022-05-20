'''
Statement - 
data wrangling - 1
perform the following operations on any open source dataset
import python libraries
locate open source dataset from web 
load dataset into pandas data frame
data preprocessing
data formatting and normalization
turn categorical var into quantitative var'''

import pandas as pd
import numpy as np

df = pd.read_csv("weatherdata.csv")
df

df.head()

df.tail()

df.describe()

df.replace(" ", np.nan)

#check if the value is null or not
missingdata = df.isnull()
missingdata

df.isnull().sum()

df.dropna()

df.dropna(axis = 1)

# drop rows only if all element are nan
df.dropna(how='all')

# used to remove missing value 
df.dropna(subset=['event'])

new_df = df.fillna({'temperature': 0.0, 'windspeed': 0.0, 'event':'Rain'})
new_df

df.dtypes

df[['duration']] = df[['duration']].astype('float')

anotherdf = df.copy()
anotherdf

# importing dataset using URL ------------------
import pandas as pd
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = pd.read_csv(URL, header=None)
print(iris)

