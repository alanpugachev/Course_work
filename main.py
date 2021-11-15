# imports
import numpy as np
from numpy.core.fromnumeric import shape
from pandas import read_csv
from numpy.lib.npyio import genfromtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils

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
training_data = read_csv('training_data.csv')
training_data = training_data.values

#—prepare data
answers = training_data[:,0]
answers = utils.to_categorical(answers)
data = training_data[:,1:]
data = normalization(data)
print(data)

#——creating AI-model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=7))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(data, answers, batch_size=1, epochs=100, verbose=1)
