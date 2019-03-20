import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
myData = pd.read_excel('data.xlsx')

# split datasets: 90% training data & 10% test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    myData['ukuran'],
    myData['harga'],
    test_size = .1
)

# linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# training
model.fit(myData[['ukuran']], myData['harga'])

# save model: pickle
import pickle

with open('1_modelPickle', 'wb') as modelku:
    pickle.dump(model, modelku)
