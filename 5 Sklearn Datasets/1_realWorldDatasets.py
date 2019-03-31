import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()
print(dir(faces))
# print(faces['data'][0])
# print(len(faces['data'][0]))
# print(faces['images'][0])
# print(len(faces['images'][0]))
# print(len(faces['images'][0][0]))
# print(faces['target'][0])
# 40 org @10 wajah

# plot
# plt.imshow(faces['images'][9])
# plt.show()

# plot orang ke-x
fig = plt.figure('Wajah', figsize=(14,7))
for x in range(10):
    orangke = 28
    plt.subplot(2,5,x+1)
    plt.imshow(faces['images'][x + (10 * (orangke-1))], cmap='gray')
    plt.suptitle('Wajah orang ke-{}'.format(orangke))

plt.show()