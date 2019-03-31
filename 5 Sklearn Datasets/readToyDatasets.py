
# read sklearn toy datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

dataIris = load_iris()
print(dir(dataIris))
# print(dataIris['data'][0])
# print(dataIris['feature_names'])
# print(dataIris['target'][0])
# print(dataIris['target_names'][0])

# create dataframe
dfIris = pd.DataFrame(
    dataIris['data'],
    columns = ['sepalL', 'sepalW', 'petalL', 'petalW']
)
dfIris['target'] = dataIris['target']
dfIris['spesies'] = dfIris['target'].apply(
    lambda x: dataIris['target_names'][x]
)
# print(dfIris[dfIris['target'] == 2])

# separate dataframe by spesies
dfSetosa = dfIris[dfIris['target'] == 0]
dfVersicolor = dfIris[dfIris['target'] == 1]
dfVirginica = dfIris[dfIris['target'] == 2]

# plot by sepal length vs sepal width
fig = plt.figure('Data Iris', figsize = (12,6))

# plot sepal
plt.subplot(121)
plt.scatter(
    dfSetosa['sepalL'],
    dfSetosa['sepalW'],
    marker = 'o',
    color = 'r'
)
plt.scatter(
    dfVersicolor['sepalL'],
    dfVersicolor['sepalW'],
    marker = 'o',
    color = 'g'
)
plt.scatter(
    dfVirginica['sepalL'],
    dfVirginica['sepalW'],
    marker = 'o',
    color = 'y'
)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal length vs sepal width')
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.grid(True)

# plot petal
plt.subplot(122)
plt.scatter(
    dfSetosa['petalL'],
    dfSetosa['petalW'],
    marker = 'o',
    color = 'r'
)
plt.scatter(
    dfVersicolor['petalL'],
    dfVersicolor['petalW'],
    marker = 'o',
    color = 'g'
)
plt.scatter(
    dfVirginica['petalL'],
    dfVirginica['petalW'],
    marker = 'o',
    color = 'y'
)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal length vs petal width')
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.grid(True)

plt.show()