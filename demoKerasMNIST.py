import os
import GPy, GPyOpt
import numpy as np
import pandas as pds
import random

from keras.layers import Activation, Dropout, BatchNormalization, Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from bayesHyperSearch import BAYESOPT

'''INSTRUCTION
Step 1: Import data to train/validate on
Step 2: Define Bounds
Step 3: modify the setHyperParameter Function as per the naming convention in Bounds
Step 4: modify the ml_model(model definition), model_fit(training), evalauate(evaluation criterian-return scalar)
'''


def get_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    return X_train, X_test, Y_train, Y_test

x_train, x_test, y_train, y_test = get_mnist_data()


class ffnn(BAYESOPT):

    def __init__(self, optimize=True, runBestModel=False):
        self.optimize = optimize
        self.runBestModel = runBestModel
        self.patienceSearch = 1
        self.patienceFinal = 2
        self.bounds = [{'id':0, 'name': 'validation_split', 'type': 'continuous', 'domain': (0.0, 0.3)},
                  {'id':1, 'name': 'l1_drop', 'type': 'continuous', 'domain': (0.0, 0.3)},
                  {'id':2, 'name': 'l2_drop', 'type': 'continuous', 'domain': (0.0, 0.3)},
                  {'id':3, 'name': 'l1_out', 'type': 'discrete', 'domain': (64, 128, 256, 512, 1024)},
                  {'id':4, 'name': 'l2_out', 'type': 'discrete', 'domain': (64, 128, 256, 512, 1024)},
                  {'id':5, 'name': 'batch_size', 'type': 'discrete', 'domain': (10, 100, 500)},
                  {'id':6, 'name': 'epochs', 'type': 'discrete', 'domain': (5, 10, 20)}]

        self.max_iter = 5
        self.filename = 'trial'
        if self.optimize:
            self._runBayesOpt()

        if runBestModel:
            self.mean, self.std = self.final_evaluation(self.opt, num_runs=2)


    def setHyperParameter(self, variable, patience=2):
        self.validation_split = variable[self.map['validation_split']]
        self.l1_drop = variable[self.map['l1_drop']]
        self.l2_drop = variable[self.map['l2_drop']]
        self.l1_out = int(variable[self.map['l1_out']])
        self.l2_out = int(variable[self.map['l2_out']])
        self.batch_size = int(variable[self.map['batch_size']])
        self.epochs = int(variable[self.map['epochs']])
        self.patience = patience

        self.__model = self.ml_model()


    def ml_model(self):
        # Build Model

        model = Sequential()
        model.add(Dense(self.l1_out, input_shape=(x_train.shape[1],)))
        model.add(Activation('relu'))
        model.add(Dropout(self.l1_drop))
        model.add(Dense(self.l2_out))
        model.add(Activation('relu'))
        model.add(Dropout(self.l2_drop))
        model.add(Dense(y_train.shape[1]))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        return model

    # fit mnist model
    def model_fit(self):
        early_stopping = EarlyStopping(patience=self.patienceSearch, verbose=1)
        filepath = os.getcwd() + "/output/weights/" + self.filename + ".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=True,
                                     mode='min')
        callbacks_list = [checkpoint, early_stopping]

        self.__model.fit(x_train, y_train,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         verbose=1,
                         validation_split=self.validation_split,
                         callbacks=callbacks_list)

    # evaluate mnist model
    def evaluation(self):
        self.model_fit()

        evaluation = self.__model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=1)
        return evaluation[0]

########### initialize the standard model
model = ffnn(optimize=True,runBestModel=True)

print('Mean Over runs with Best Model:', model.mean)
print('Std Over runs with Best Model:', model.std)
