![simplinnovation](https://4.bp.blogspot.com/-f7YxPyqHAzY/WJ6VnkvE0SI/AAAAAAAADTQ/0tDQPTrVrtMAFT-q-1-3ktUQT5Il9FGdQCLcB/s350/simpLINnovation1a.png)

# Basic Machine Learning Using Sklearn Tutorial

### __2. Multivariate Regression__ (📂[_click to go to its repo_](https://github.com/LintangWisesa/Sklearn_Tutorial_Youtube/tree/master/2%20Multivariate%20Regression))

[![lintang](https://img.youtube.com/vi/pwIFAiqk6TU/0.jpg)](https://www.youtube.com/watch?v=pwIFAiqk6TU)

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataku = pd.read_excel('data.xlsx')
print(dataku.head())

# split dataset: 90% training & 10% test
from sklearn.model_selection import train_test_split
a, b, c, d = train_test_split(
    dataku[['ukuran', 'kamar', 'AC']],
    dataku['harga'],
    test_size = .1
)
# print(a)
# print(b)
# print(c)
# print(d)

# linear regression multi variable: multivariate linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# training
model.fit(a,c)

# slope/gradient (m)
print(model.coef_)

# intercept (b)
print(model.intercept_)

# akurasi (score)
print(model.score(b,d) * 100, '%')

# prediksi:
# ukuran: 60, kamar: 3, AC: 2
print(model.predict([[60, 3, 2]]))  # 605jt => 598.8jt
# ukuran: 100, kamar: 3, AC: 3 
print(model.predict([[100, 3, 3]]))  # 1005jt => 1011jt

```

#

#### Lintang Wisesa :love_letter: _lintangwisesa@ymail.com_

[Facebook](https://www.facebook.com/lintangbagus) | 
[Twitter](https://twitter.com/Lintang_Wisesa) |
[Google+](https://plus.google.com/u/0/+LintangWisesa1) |
[Youtube](https://www.youtube.com/user/lintangbagus) | 
:octocat: [GitHub](https://github.com/LintangWisesa) |
[Hackster](https://www.hackster.io/lintangwisesa)