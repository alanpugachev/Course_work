# imports
import numpy as np
import matplotlib.pyplot as plt
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
model.add(Activation('selu'))
model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=5, epochs=50, verbose=1, validation_data=(x_val, y_val))


#———graphics
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

#----------loss graphic----------#
plt.plot(epochs, loss_values, 'bo', label='Training loss') 
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#----------accuracy graphic----------#
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()