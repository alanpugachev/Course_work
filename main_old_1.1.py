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


def build_model(hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])

    model.add(Dense(units=hp.Int('units_input', min_value=512, max_value=1024, step=32), input_dim=7, activation=activation_choice))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    for i in range(hp.Int('num_layers', 2, 4)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
        min_value=128, max_value=1024, step=32), activation='elu'))

    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=hp.Choice('optimizer', values=['adam', 'SGD', 'rmsprop']), metrics=['accuracy'])

    return model


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
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    directory='models/rating'
)
tuner.search(x_train, y_train, batch_size=10, epochs=50, validation_split=0.2 ,verbose=1, validation_data=(x_val, y_val))
print(tuner.get_best_models(num_models=3))
models = tuner.get_best_models(num_models=3)

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