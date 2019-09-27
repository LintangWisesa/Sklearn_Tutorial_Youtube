# OLD WAY
# load model: joblib
# from sklearn.externals import joblib
# model = joblib.load('2_modelJoblib')

# NEW WAY
# $ pip install joblib
import joblib
model = joblib.load('2_modelJoblib')

# prediction: harga terbaik untuk 300m2, 672m2, 2102m2
print(model.predict([[ 300 ]]))
print(model.predict([[ 672 ]]))
print(model.predict([[ 2102 ]]))