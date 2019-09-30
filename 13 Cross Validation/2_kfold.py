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
# 2. K-Fold Cross Validation

from sklearn.model_selection import KFold
kf = KFold(n_splits=3)

# for i, j in kf.split([1,2,3,4,5,6,7,8,9]):
#     print(i, j)

def get_score(model, xtr, xts, ytr, yts):
    model.fit(xtr, ytr)
    return model.score(xts, yts)

score_lr = []
score_sv = []
score_rf = []

for train_i, test_i in kf.split(digits['data']):
    xtr = digits['data'][train_i]
    xts = digits['data'][test_i]
    ytr = digits['target'][train_i]
    yts = digits['target'][test_i]

    score_lr.append(get_score(modellr, xtr, xts, ytr, yts))
    score_sv.append(get_score(modelsv, xtr, xts, ytr, yts))
    score_rf.append(get_score(modelrf, xtr, xts, ytr, yts))

# print(score_lr)
# print(score_sv)
# print(score_rf)

print('Score LR:', np.mean(score_lr))
print('Score SV:', np.mean(score_sv))
print('Score RF:', np.mean(score_rf))