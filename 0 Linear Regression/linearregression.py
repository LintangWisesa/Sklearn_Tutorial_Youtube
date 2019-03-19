import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

myData = pd.read_excel('data.xlsx')
# print(myData)

# split datasets: 90% training data & 10% test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    myData['ukuran'],
    myData['harga'],
    test_size = .1
)
print(len(x_train))
print(len(x_test))

# linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# training
model.fit(myData[['ukuran']], myData['harga'])

# slope/gradient/m best fit line:
print(model.coef_[0])

# intercept/b best fit line:
print(model.intercept_)


# plot best fit line
# y = mx + b
plt.plot(
    myData['ukuran'],
    model.coef_[0] * myData['ukuran'] + model.intercept_,
    'r-'
)

# plot dataframe
plt.scatter(
    myData['ukuran'],
    myData['harga'],
    marker = 'o',
    color = 'g'
)
plt.title('Harga Rumah vs Ukuran Tanah')
plt.xlabel('Ukuran (m2)')
plt.ylabel('Harga (juta Rupiah)')
plt.grid(True)

plt.show()

# prediction: harga terbaik: 320m2, 670m2 & 3000m2 ?
print('Prediksi')
print('Prediksi harga rumah 320m2 =', model.predict([[320]]))
print('Prediksi harga rumah 670m2 =', model.predict([[670]]))
print('Prediksi harga rumah 3000m2 =', model.predict([[3000]]))