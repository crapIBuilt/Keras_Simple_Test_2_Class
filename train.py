from numpy import loadtxt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.regularizers import l2
import tensorflow as tf
import shutil
import os
dataset = loadtxt('data.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8:10]
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        u = logs.get('accuracy')

        if logs.get('accuracy') >= 99e-2:
            self.model.stop_training = True
callback = CustomCallback()
model = Sequential()
model.add(Dense(24, input_shape=(8,), activation='sigmoid', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model.add(Dense(14, activation='gelu'))
model.add(Dense(12, activation='gelu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=20000, batch_size=5000, callbacks=[callback])
_, accuracy = model.evaluate(X, y)
print(accuracy)
model.save('model')
shutil.make_archive('baseDataModel.zip', 'zip', 'model')
