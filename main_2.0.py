# imports
import numpy as np
import matplotlib.pyplot as plt

from pandas import read_csv
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
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu', 'linear'])

    model.add(Dense(units=hp.Int('units_input', min_value=128, max_value=1024, step=32), input_dim=29, activation=activation_choice))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
        min_value=128, max_value=1024, step=32), activation='elu'))

    model.add(Dense(19, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=hp.Choice('optimizer', values=['adam', 'SGD', 'rmsprop']), metrics=['accuracy'])

    return model


# main
data = read_csv('dataset.csv')
data = data.values

#—prepare data
#----------training----------#
y_train = data[:90,2] #prepare answers for training
y_train = utils.to_categorical(y_train)
x_train = data[:90,3:] #prepare dataset for training
x_train = np.asarray(x_train).astype('float32')

#----------validation----------#
y_val = data[90:100,2]
y_val = utils.to_categorical(y_val)
x_val = data[90:100,3:]
x_val = np.asarray(x_val).astype('float32')

#----------testing----------#
y_test = data[100:,2] #prepare answers for testing
y_test = utils.to_categorical(y_test)
x_test = data[100:,3:] #prepare dataset for testing
x_test = np.asarray(x_test).astype('float32')


#——creating NN-Model
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    directory='models'
)
tuner.search(x_train, y_train, batch_size=9, epochs=10, validation_split=0.2 ,verbose=1, validation_data=(x_val, y_val))
print(tuner.get_best_models(num_models=1))
models = tuner.get_best_models(num_models=1)

for model in models:
    model.summary()
    model.evaluate(x_test, y_test)
    print()


# #———graphics
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']

# epochs = range(1, len(loss_values) + 1)

# #----------loss graphic----------#
# plt.plot(epochs, loss_values, 'bo', label='Training loss') 
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# #----------accuracy graphic----------#
# plt.clf()
# acc_values = history_dict['accuracy']
# val_acc_values = history_dict['val_accuracy']

# plt.plot(epochs, acc_values, 'bo', label='Training acc')
# plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
