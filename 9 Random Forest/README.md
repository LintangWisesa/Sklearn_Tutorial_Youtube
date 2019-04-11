![simplinnovation](https://4.bp.blogspot.com/-f7YxPyqHAzY/WJ6VnkvE0SI/AAAAAAAADTQ/0tDQPTrVrtMAFT-q-1-3ktUQT5Il9FGdQCLcB/s350/simpLINnovation1a.png)

# Basic Machine Learning Using Sklearn Tutorial

### __9. Random Forest__ (📂[_click to go to its repo_](https://github.com/LintangWisesa/Sklearn_Tutorial_Youtube/tree/master/7%20Dummy%20Var%20%26%20Label%20Encoder))

[![lintang](https://img.youtube.com/vi/Sp5mqaH4FE0/0.jpg)](https://www.youtube.com/watch?v=Sp5mqaH4FE0)

```python
import pandas as pd
import numpy as np

df = pd.read_csv('0_data.csv')

# convert nominal data => ordinal data: label encoder
from sklearn.preprocessing import LabelEncoder

LabelKantor = LabelEncoder()
df['kantorLE'] = LabelKantor.fit_transform(df['kantor'])
LabelJabat = LabelEncoder()
df['jabatLE'] = LabelJabat.fit_transform(df['jabatan'])
LabelTitel = LabelEncoder()
df['titelLE'] = LabelTitel.fit_transform(df['titel'])

df = df.drop(['kantor', 'jabatan', 'titel'], axis = 'columns')

print(df)

# '''
# kantor: Bukalapak 0, Gojek 1, Tokopedia 2
# jabatan: GM 0, Manager 1, Staff IT 2
# titel: S1 0, S2 1
# '''

# split dataset: training 90% & testing 10%

from sklearn.model_selection import train_test_split

x_train, x_tes, y_train, y_tes = train_test_split(
    df[['kantorLE', 'jabatLE', 'titelLE']],
    df['gaji>50jt'],
    test_size = .1
)
print(len(x_train))
print(len(x_tes))

# random forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100)

# training
model.fit(x_train, y_train)

# '''
# kantor: Bukalapak 0, Gojek 1, Tokopedia 2
# jabatan: GM 0, Manager 1, Staff IT 2
# titel: S1 0, S2 1
# '''

# classification
print(model.predict([[2, 2, 0]]))

# accuracy
print(model.score(x_train, y_train) * 100, '%')
```

#

#### Lintang Wisesa :love_letter: _lintangwisesa@ymail.com_

[Facebook](https://www.facebook.com/lintangbagus) | 
[Twitter](https://twitter.com/Lintang_Wisesa) |
[Google+](https://plus.google.com/u/0/+LintangWisesa1) |
[Youtube](https://www.youtube.com/user/lintangbagus) | 
:octocat: [GitHub](https://github.com/LintangWisesa) |
[Hackster](https://www.hackster.io/lintangwisesa)