import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    digits['data'], 
    digits['target'], 
    test_size=.1
)

# import models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

modellr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)
modelsv = SVC(gamma='auto', probability=True)
modelrf = RandomForestClassifier(n_estimators=10)

# ==================================================
# 1. manual scoring

modellr.fit(x_train, y_train)
modelsv.fit(x_train, y_train)
modelrf.fit(x_train, y_train)

print('Score LR:', modellr.score(x_test, y_test))
print('Score SV:', modelsv.score(x_test, y_test))
print('Score RF:', modelrf.score(x_test, y_test))
