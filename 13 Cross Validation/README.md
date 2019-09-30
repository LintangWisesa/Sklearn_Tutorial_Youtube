![simplinnovation](https://4.bp.blogspot.com/-f7YxPyqHAzY/WJ6VnkvE0SI/AAAAAAAADTQ/0tDQPTrVrtMAFT-q-1-3ktUQT5Il9FGdQCLcB/s350/simpLINnovation1a.png)

# Basic Machine Learning Using Sklearn Tutorial

### __13. Cross Validation__ (📂[_click to go to its repo_](https://github.com/LintangWisesa/Sklearn_Tutorial_Youtube/tree/master/13%20Cross%20Validation))

[![lintang](https://img.youtube.com/vi/umueLxxRNaU/0.jpg)](https://www.youtube.com/watch?v=umueLxxRNaU)

#

### 1⃣ Scoring Manually

```python
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
```

#

### 2⃣ K-Fold Cross Validation

```python
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
```

#

### 3⃣ Sklearn ```cross_val_score()```

```python
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
```

#

#### Lintang Wisesa :love_letter: _lintangwisesa@ymail.com_

[Facebook](https://www.facebook.com/lintangbagus) | 
[Twitter](https://twitter.com/Lintang_Wisesa) |
[Google+](https://plus.google.com/u/0/+LintangWisesa1) |
[Youtube](https://www.youtube.com/user/lintangbagus) | 
:octocat: [GitHub](https://github.com/LintangWisesa) |
[Hackster](https://www.hackster.io/lintangwisesa)