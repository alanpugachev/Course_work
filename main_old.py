# imports
import numpy as np
from numpy.core.fromnumeric import shape
from pandas import read_csv
from numpy.lib.npyio import genfromtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.python.keras.layers.core import Dropout

def normalization(data):
    least_element = np.min(data)
    max_element = np.max(data)
    dim = data.shape
    x = dim[0]
    y = dim[1]

    for i in range(x):
        for j in range(y):
            data[i][j] = ((data[i][j] - least_element) / (max_element - least_element))

    return data

# main
data = read_csv('training_data.csv')
data = data.values

#—prepare data
answers = data[:,0]
answers = utils.to_categorical(answers)
training_data = data[:55,1:]
training_data = normalization(training_data)
testing_data = data[55:63,1:]
testing_data = normalization(testing_data) #batchnormalization
control_data = data[63:,1:]
control_data = normalization(control_data)

#——creating AI-model
model = Sequential()
model.add(Dense(64, activation='linear', input_dim=7))
model.add(Dense(64, activation='selu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data, answers, batch_size=1, epochs=250, verbose=1)
