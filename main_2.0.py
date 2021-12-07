# imports
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import utils
from tensorflow.python.keras.layers.core import Activation, Dropout
from keras_tuner import RandomSearch, Hyperband, BayesianOptimization


# functions
def build_model(hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])

    model.add(Dense(units=hp.Int('units_input', min_value=512, max_value=1024, step=32), input_dim=7, activation=activation_choice))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
        min_value=128, max_value=1024, step=32), activation='elu'))

    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=hp.Choice('optimizer', values=['adam', 'SGD', 'rmsprop']), metrics=['accuracy'])

    return model


# main
