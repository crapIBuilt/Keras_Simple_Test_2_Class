import numpy as np
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
dataset = loadtxt('data.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8:10]
model = keras.models.load_model("model")
model.predict(X)
inputData = [[5, 121, 72, 23, 112, 26.2, 0.245, 30]]
prediction = model.predict(inputData)
model.summary()
print("X=%s, Predicted=%s" % (inputData[0], prediction[0]))
