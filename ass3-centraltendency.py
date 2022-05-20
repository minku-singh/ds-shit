'''
descriptive statistics - Measures of central tendency and variability
provide mean median mode 
standard deviation
iris setosa
'''

import pandas as pd
import numpy as np

dataset = pd.read_csv("E:\\all-docs\\TE-assignments\\DS-BDA\\Mall_Customers.csv")
dataset

dataset.shape
dataset.info()

dataset.mean()
dataset.loc[:, 'Age'].mean()

dataset.loc[:"Annual Income (k$)"].mean()

dataset.mean(axis = 1)[0:5]
dataset.median()
dataset.loc[:,"Age"].median()
dataset.mode(axis = 0)
dataset.mode()
dataset.std()

dataset.loc[:,"Age"].std()
dataset.std(axis = 1)[0:5]
dataset.var()
from scipy.stats import iqr
iqr(dataset['Age'])

dataset.skew()

dataset.describe()

dataset.describe(include = "all")

grouped = dataset.groupby("Age")
grouped

grouped.groups

grouped.size()

grouped["Age"]
grouped["Age"].size()

print(dataset.groupby(["Gender"]).count().reset_index())

print(dataset.groupby(["Gender"]).mean().reset_index())


#################################### part 2 ###################################

import pandas as pd
data = pd.read_csv("E:\\all-docs\\TE-assignments\\DS-BDA\\Iris.csv")
print("Iris-setosa")
setosa = data["Species"] == "Iris-setosa"
print(data[setosa].describe())
print('\nIris-versicolor')
setosa = data["Species"] == 'Iris-versicolor'
print(data[setosa].describe())
print("\nIris-virginica")
setosa = data["Species"] == "Iris-virginica"
print(data[setosa].describe())

import pandas as pd
import numpy as np

csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = pd.read_csv(csv_url, header=None)
col_names = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'Species']
iris = pd.read_csv(csv_url, names = col_names)
print(iris)

irisSet = (iris['Species'] == "Iris-setosa")
print("Iris-setosa")
print(iris[irisSet].describe())

irisVer = (iris["Species"] == "Iris-versicolor")
print("Iris-versicolor")
print(iris[irisVer].describe())

irisVir = (iris["Species"] == "Iris-virginica")
print("Iris-virginica")
print(iris[irisVir].describe())

