# load model: pickle
import pickle

with open('1_modelPickle', 'rb') as modelku:
    model = pickle.load(modelku)

# prediction: harga terbaik untuk rumah ukuran: 300m2, 450m2, 1200m2?
print(model.predict([[ 300 ]]))
print(model.predict([[ 450 ]]))
print(model.predict([[ 1200 ]]))