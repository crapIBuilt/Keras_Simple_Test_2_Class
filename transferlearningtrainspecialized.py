import numpy as np
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
datasetold = loadtxt('data.csv', delimiter=',')
dataset = loadtxt('datanew.csv', delimiter=',')
Xevalmaintainedaccu = datasetold[:,0:8]
yevalmaintainedaccu = datasetold[:,8]
Xnewdata = dataset[:,0:8]
globb = 0
ynewdata = dataset[:,8]
model = keras.models.load_model("model")
model.summary()
def train():
    model.fit(Xnewdata, ynewdata, epochs=5000, batch_size=5000)
    accuracy1 = model.evaluate(Xevalmaintainedaccu, yevalmaintainedaccu)
    globb = accuracy1
_, accuracy = model.evaluate(Xevalmaintainedaccu, yevalmaintainedaccu)
train()
print('---------------------------')
print(accuracy)
print(model.evaluate(Xevalmaintainedaccu, yevalmaintainedaccu))
model.save('transfermodel')
