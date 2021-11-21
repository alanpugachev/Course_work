# imports
import numpy as np
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import utils
from tensorflow.python.keras.layers.core import Activation, Dropout


# main
data = read_csv('training_data.csv')
data = data.values


#—prepare data
#----------training----------#
y_train = data[:55,0] #prepare answers for training (0-54 elements)
y_train = utils.to_categorical(y_train)
x_train = data[:55,1:] #prepare dataset for training

#----------testing----------#
y_test = data[55:63,0] #prepare answers for testing (55-62 elements)
y_test = utils.to_categorical(y_test)
x_test = data[55:63,1:] #prepare dataset for testing

#----------validation----------#
y_val = data[63:,0]
y_val = utils.to_categorical(y_val)
x_val = data[63:,1:]


#——creating NN-Model
model = Sequential()

model.add(Dense(32, input_dim=7))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.25))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=11, epochs=100, verbose=1, validation_data=(x_val, y_val))
