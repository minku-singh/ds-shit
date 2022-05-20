import numpy as np
import pandas as pd
# Dataset link from the UCI repository
csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(csv_url, header=None)
col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']
iris = pd.read_csv(csv_url,names=col_names)
print(iris)

iris.head()

iris.info()

np.unique(iris['Species'])

# Creating a histogram for each feature in the dataset to illustrate the feature distributions
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

fig, axes = plt.subplots(2,2, figsize=(16,8))
axes[0,0].set_title('Distribution of first column')
axes[0,0].hist(iris['Sepal_Length'])
axes[0,1].set_title('Distribution of second column')
axes[0,1].hist(iris['Sepal_Width'])
axes[1,0].set_title('Distribution of third column')
axes[1,0].hist(iris['Petal_Length'])
axes[1,1].set_title('Distribution of fourth column')
axes[1,1].hist(iris['Petal_Width'])

data_to_plot = [iris['Sepal_Length'],iris['Sepal_Width'],iris['Petal_Length'],iris['Petal_Width']]
sns.set_style('whitegrid')
# Creating a figure instance
fig = plt.figure(1,figsize=(12,8))
# Creating an axes instance
ax = fig.add_subplot(111)
# Creating the boxplot
bp = ax.boxplot(data_to_plot);

