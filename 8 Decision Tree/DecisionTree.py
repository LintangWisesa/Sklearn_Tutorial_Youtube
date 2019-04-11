import pandas as pd
import numpy as np

df = pd.read_csv('0_data.csv')

# print(df)

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

# split dataset: training 90% & testing 10%

from sklearn.model_selection import train_test_split

x_train, x_tes, y_train, y_tes = train_test_split(
    df[['kantorLE', 'jabatLE', 'titelLE']],
    df['gaji>50jt'],
    test_size = .1
)
print(len(x_train))
print(len(x_tes))

# decision tree
from sklearn import tree
model = tree.DecisionTreeClassifier()

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
print(model.score(x_tes, y_tes) * 100, '%')