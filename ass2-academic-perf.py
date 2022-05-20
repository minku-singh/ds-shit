'''
data wrangling 2
create academic performance dataset and perform follwing operation 
missing value 
outlier 
data transformation '''

import pandas as pd
import numpy as np

df = pd.read_csv("E:\\all-docs\\TE-assignments\\DS-BDA\\academic-performance.csv")
df

df.isnull()

series = pd.isnull(df["math score"])
df[series]

df.notnull()

series = pd.notnull(df["math score"])
df[series]

missing_values = ["Na", "na"]
df = pd.read_csv("E:\\all-docs\\TE-assignments\\DS-BDA\\academic-performance.csv")
df

ndf = df
ndf.fillna(0)

m_v = df["math score"].mean()
df["math score"].fillna(value=m_v, inplace=True)
df

ndf.replace(to_replace = np.nan, value = -99)

ndf.dropna(how="all")

ndf.dropna(axis = 1)

new_data = ndf.dropna(axis = 0, how = 'any')
new_data

#Identification and handling of outliers-----------------------------------------------
import pandas as pd
import numpy as np

df1 = pd.read_csv("E:\\all-docs\\TE-assignments\\DS-BDA\\heights.csv")
df1.head()

df1

df1.shape

df1["name"]

df1["height"]

df1["height"].quantile(0.95)

# detect outlier using percentile -------------------------------------------------------
max_threshold = df1["height"].quantile(0.95)
max_threshold

df1[df1["height"] > max_threshold]

min_threshold = df1["height"].quantile(0.05)
min_threshold

df1[df1["height"] < min_threshold]

# remove outliers ------------------------------------------------------------------
df1[(df1["height"] < max_threshold) & (df1["height"] > min_threshold)]

df2 = df1[(df1["height"] < max_threshold) & (df1["height"] > min_threshold)]
df2.shape

df2.describe()

df1.shape

df1.describe()

import pandas as pd
import numpy as np

df2 = pd.read_csv("E:\\all-docs\\TE-assignments\\DS-BDA\\academic-performance-outlier.csv")
df2

max_threshold = df2["math score"].quantile(0.90)
max_threshold

df2[df2["math score"] > max_threshold]

min_threshold = df2["math score"].quantile(0.50)
min_threshold

df2[df2["math score"] < min_threshold]

df3 = df2[(df2["math score"] > max_threshold) & (df2["math score"] < min_threshold)]
df3

df3.shape

# outliers visualization ----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("E:\\all-docs\\TE-assignments\\DS-BDA\\academic-performance-outlier.csv")
df

df.describe()

col = ["math score", 'reading score', 'writing score', 'placement score']
df.boxplot(col)

print(np.where(df['math score']>90))
print(np.where(df['math score']<25))
print(np.where(df['math score']<30))

df.shape

# detecting outlier using IQR ---------------------------------------------------
import pandas as pd
df = pd.read_csv("E:\\all-docs\\TE-assignments\\DS-BDA\\heights.csv")
df

df.describe()

Q1 = df.height.quantile(0.25)
Q3 = df.height.quantile(0.75)
Q1, Q3

IQR = Q3-Q1
IQR

lower_limit = Q1-1.5*IQR
upper_limit = Q3+1.5*IQR
lower_limit, upper_limit

# found outliers are 
df[(df.height<lower_limit) | (df.height>upper_limit)]

# trimming and removing the outliers ------------------------------------------------
df_no_outlier = df[(df.height>lower_limit)&(df.height<upper_limit)]
df_no_outlier

df.shape

df_no_outlier.shape

# detection of outliers on academic performance dataset----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("E:\\all-docs\\TE-assignments\\DS-BDA\\academic-performance-outlier.csv")
df

new_df = df
col = ["math score"]
new_df.boxplot(col)

# detecting outlier with IQR -----------------------------------------------------------
q1 = np.percentile(df["math score"], 25)
q3 = np.percentile(df["math score"], 75)
print(q1, q3)

IQR = q3-q1
lwr_bound = q1 - (1.5*IQR)
upr_bound = q3 + (1.5*IQR)
print(lwr_bound, upr_bound)

index_outliers = np.where((df[col] < lwr_bound) | (df[col] > upr_bound))
index_outliers

df

sample_outliers = df[col][(df[col] < lwr_bound) | (df[col] > upr_bound)]
sample_outliers

sample_outiers = df[col][(df[col] < lwr_bound) | (df[col] > upr_bound)]
sample_outliers

# handling of outliers - Qunatile based flooring and capping -----------------------------
df1 = df
df[col] = np.where(df[col] < lwr_bound, lwr_bound, df[col])
df[col] = np.where(df[col] > upr_bound, upr_bound, df[col])
df

ninetieth_percentile = np.percentile(df1["math score"], 90)
ninetieth_percentile

df1[col] = np.where(df1[col] > upr_bound, ninetieth_percentile,df1[col])
df1

# replacing outlier with median value --------------------------------------------------
median = np.median(new_df[col])
median

for i in index_outliers:
    new_df.at[i, col]=median
new_df

# detecting outlier with z-score --------------------------------------------------
import numpy as np
from scipy import stats
df3 = pd.read_csv("E:\\all-docs\\TE-assignments\\DS-BDA\\academic-performance-outlier.csv")
df3

df3.shape

z = np.abs(stats.zscore(df3["math score"]))
z

threshold = 0.18

sample_outliers = np.where(z < threshold)
sample_outliers

upperthreshold = 1.4
lowerthreshold = 0.18

index_outliers = np.where((z<lowerthreshold) | (z > upperthreshold))
index_outliers

sample_outliers = np.where((z<lowerthreshold) | (z > upperthreshold))
sample_outliers

# trimming and removing the outlier ---------------------------------------------------------
new_df1 = df3
for i in sample_outliers:
    new_df1.drop(i, inplace = True)
new_df1

#module 2 --------------------------------------------------------------------
# NORMALIZATION --------------------------------------------------------------
import pandas as pd
import numpy as mp
import matplotlib.pyplot as plt

df4 = pd.read_csv("E:\\all-docs\\TE-assignments\\DS-BDA\\normalization.csv")
df4

df4.plot(kind = "bar")

df_max_scaled = df4.copy()
for column in df_max_scaled.columns:
    df_max_scaled[column] = df_max_scaled[column] / df_max_scaled[column].abs().max()
df_max_scaled

df_max_scaled.plot(kind="bar")

df_min_max_scaled = df4.copy()

for column in df_min_max_scaled.columns:
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
print(df_min_max_scaled)

df_min_max_scaled.plot(kind = 'bar')

# using z-score method --------------------------------------------------------------
df_z_scaled = df4.copy()
for column in df_z_scaled.columns:
    df_z_scaled[column] = (df_z_scaled[column] - df_z_scaled[column].mean())/df_z_scaled[column].std()
    
print(df_z_scaled)

import matplotlib.pyplot as plt
df_z_scaled.plot(kind = "bar")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df5 = pd.read_csv("E:\\all-docs\\TE-assignments\\DS-BDA\\academic-performance-outlier-z.csv")
df5

df5.plot(kind = "bar")

df5["math score"].plot(kind = "bar")

#min max normalization
df_min_max_scaled = df5.copy()
for column in df_min_max_scaled.columns:
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
print(df_min_max_scaled)

df_min_max_scaled.plot(kind = "bar")

df_min_max_scaled.skew()

# Hence, we can check which variable is normally distributed and which is not
# skewness > 1 => highly positively skewed
# skewness < -1 => highly negatively skewed
# 0.5 < skewness < 1 => moderately positively skewed
# -0.5 < skewness < -1 => moderately negatively skewed
# -0.5 < skewness < 0.5 => symmetric(normally distributed)
