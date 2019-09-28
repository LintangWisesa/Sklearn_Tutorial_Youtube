import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load sklearn's iris datasets
from sklearn.datasets import load_iris

dataIris = load_iris()
df = pd.DataFrame(
    dataIris['data'],
    columns = ['SL', 'SW', 'PL', 'PW']
)
df['target'] = dataIris['target']
df['spesies'] = df['target'].apply(
    lambda row: dataIris['target_names'][row]
)
print(df.head())