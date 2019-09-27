import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

dataIris = load_iris()
# print(dir(dataIris))

# create dataframe
dfIris = pd.DataFrame(
    dataIris['data'],
    columns = ['sepalL', 'sepalW', 'petalL', 'petalW']
)
dfIris['target'] = dataIris['target']
dfIris['spesies'] = dfIris['target'].apply(
    lambda x: dataIris['target_names'][x]
)

# separate dataframe by spesies
dfSetosa = dfIris[dfIris['target'] == 0]
dfVersicolor = dfIris[dfIris['target'] == 1]
dfVirginica = dfIris[dfIris['target'] == 2]

# split dataset test 10% & training 90%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    dfIris[['sepalL', 'sepalW', 'petalL', 'petalW']],
    dfIris['target'],
    test_size = .1
)

# menentukan nilai k (n_neighbors)
# k = sqrt(jumlah data) => ganjil
def nilai_k():
    k = round((len(x_train) + len(x_test)) ** .5)
    if (k % 2 == 0):
        return k+1
    else:
        return k
print('Nilai k =', nilai_k())

# KNN model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(
    n_neighbors = nilai_k()
)

# training
model.fit(x_train, y_train)

# accuracy
print(model.score(x_test, y_test) * 100, '%')

# predict
print(model.predict([[6.3, 2.5, 5.0, 1.9]]))
print(dfIris.tail())