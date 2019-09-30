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
# 3. Sklearn's Cross Validation Score

from sklearn.model_selection import cross_val_score

score_lr = cross_val_score(modellr, digits['data'], digits['target'], cv=5)
score_sv = cross_val_score(modelsv, digits['data'], digits['target'], cv=5)
score_rf = cross_val_score(modelrf, digits['data'], digits['target'], cv=5)

print('Score LR:', np.mean(score_lr))
print('Score SV:', np.mean(score_sv))
print('Score RF:', np.mean(score_rf))