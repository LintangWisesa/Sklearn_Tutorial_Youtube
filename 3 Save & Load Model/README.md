![simplinnovation](https://4.bp.blogspot.com/-f7YxPyqHAzY/WJ6VnkvE0SI/AAAAAAAADTQ/0tDQPTrVrtMAFT-q-1-3ktUQT5Il9FGdQCLcB/s350/simpLINnovation1a.png)

# Basic Machine Learning Using Sklearn Tutorial

### __3. Save & Load Model__ (📂[_click to go to its repo_](https://github.com/LintangWisesa/Sklearn_Tutorial_Youtube/tree/master/3%20Save%20%26%20Load%20Model))

[![lintang](https://img.youtube.com/vi/fmsRpDM3Kvk/0.jpg)](https://www.youtube.com/watch?v=fmsRpDM3Kvk)

#

#### Save Model with Pickle:

```python
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
```

#### Load Model with Pickle:

```python

# load model: pickle
import pickle

with open('1_modelPickle', 'rb') as modelku:
    model = pickle.load(modelku)

# prediction: harga terbaik untuk rumah ukuran: 300m2, 450m2, 1200m2?
print(model.predict([[ 300 ]]))
print(model.predict([[ 450 ]]))
print(model.predict([[ 1200 ]]))

```

#

#### Save Model with Joblib:

```python
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

# save model: joblib
from sklearn.externals import joblib
joblib.dump(model, '2_modelJoblib')
```

#### Load Model with Joblib:

```python
# load model: joblib
from sklearn.externals import joblib
model = joblib.load('2_modelJoblib')

# prediction: harga terbaik untuk 300m2, 672m2, 2102m2
print(model.predict([[ 300 ]]))
print(model.predict([[ 672 ]]))
print(model.predict([[ 2102 ]]))
```

#### Lintang Wisesa :love_letter: _lintangwisesa@ymail.com_

[Facebook](https://www.facebook.com/lintangbagus) | 
[Twitter](https://twitter.com/Lintang_Wisesa) |
[Google+](https://plus.google.com/u/0/+LintangWisesa1) |
[Youtube](https://www.youtube.com/user/lintangbagus) | 
:octocat: [GitHub](https://github.com/LintangWisesa) |
[Hackster](https://www.hackster.io/lintangwisesa)