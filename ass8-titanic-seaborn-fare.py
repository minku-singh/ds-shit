# Assignment_8 : Data Visualization I
# 1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows
# and contains information about the passengers who boarded the
# unfortunate Titanic ship. Use the Seaborn library to see if we can
# find any patterns in the data.
# 2. Write a code to check how the price of the ticket (column
# name: 'fare') for each passenger is distributed by plotting a
# histogram

import pandas as pd
import matplotlib.pyplot as plt
from pyrsistent import v
%matplotlib inline
import math
import numpy as np
import seaborn as sns #library for statistical plotting
df = sns.load_dataset('titanic')

df.head()

df.info()

sns.distplot(df['fare'])

sns.distplot(df["fare"],kde=False)

sns.distplot(df["fare"],kde=True,bins=15)

sns.distplot(df["fare"],bins=30,kde=False,color='Red')

sns.jointplot(x='fare',y='age',data=df)

sns.jointplot(x = df['age'], y = df['fare'], kind = 'scatter')

sns.jointplot(x = df['age'], y = df['fare'], kind = 'hex')

sns.jointplot(x='fare',y='age',data=df,kind='kde')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv("E:\\all-docs\\TE-assignments\\DS-BDA\\practical\\test.csv")
dataset

sns.pairplot(dataset)

sns.pairplot(dataset,hue="Sex")

sns.pairplot(dataset, hue='Sex', diag_kind="hist", kind="scatter", palette="husl")

sns.rugplot(dataset['Fare'])

sns.barplot(x='Sex', y='Age', data=dataset)

ax = sns.barplot(x="Age", y="Fare", hue="Sex", data=dataset)

sns.countplot(x ='sex', data = df)

sns.set_theme(style="darkgrid")
ax = sns.countplot(x="class", data=df)

ax = sns.countplot(x="class", hue="who", data=df)

ax = sns.countplot(y="class", hue="who", data=df)

ax = sns.countplot(x="who", data=df, palette="Set3")

fig,axes=plt.subplots(1,3,figsize=(15,8))
plt.suptitle(" Number of Survivors Based On Sex,Pclass and Embarked",fontsize=20)
sns.countplot(x="sex",hue="survived",data=df,ax=axes[0],palette="Paired")
sns.countplot(x="pclass",hue="survived",data=df,ax=axes[1],palette="Paired")
sns.countplot(x="embarked",hue="survived",data=df,ax=axes[2],palette="Paired")

sns.boxplot(x='sex', y='age', data=df)

sns.violinplot(x='sex', y='age', data=df)

ax = sns.violinplot(x="age", hue="sex",
 data=df, palette="Set3", split=True,
 scale="count")

ax = sns.violinplot(x="age",y="fare", hue="sex",
 data=df, palette="Set3", split=True,
 scale="count")

sns.violinplot(x ="sex", y ="age", hue ="survived",
data = df, split = True)

sns.stripplot(y = dataset['Age'], x = dataset['Pclass'])

sns.swarmplot(y = dataset['Age'], x = dataset['Pclass'])

plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), cmap = "YlGnBu", annot=True, fmt=".2f")



