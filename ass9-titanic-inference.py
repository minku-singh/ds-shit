# Assignment_9 : Data Visualization II
# 1. Use the inbuilt dataset 'titanic' as used in the above problem. Plot a box plot for distribution of age with respect to each gender along
# with the information about whether they survived or not. (Column names : 'sex' and 'age')
# 2. Write observations on the inference from the above statistics

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import math
import numpy as np
import seaborn as sns #library for statistical plotting
df = sns.load_dataset('titanic')

df

df.head()

sns.boxplot(x= 'sex', y='age', data=df)

sns.boxplot(x='sex', y='age', data=df, hue="survived")