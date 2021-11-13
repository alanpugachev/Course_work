# imports
import numpy as np
from numpy.core.fromnumeric import shape
from pandas import read_csv
from numpy.lib.npyio import genfromtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils

#TODO
#def normalization(data, x, y):

# main
training_data = read_csv('training_data.csv')
training_data = training_data.values

#—prepare data
answers = training_data[:,0]
answers = utils.to_categorical(answers)
data = training_data[:,1:]

#——creating AI-model
model = Sequential()
model.add(Dense(128, activation='sigmoid', input_dim=7))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
model.fit(data, answers, batch_size=1, epochs=100, verbose=1)
