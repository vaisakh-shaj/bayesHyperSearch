import os
import GPy, GPyOpt
import numpy as np
import pandas as pds
import random

from sklearn import datasets, svm, metrics
from keras.datasets import mnist

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

    Y_train = y_train
    Y_test = y_test
    return X_train, X_test, Y_train, Y_test

x_train, x_test, y_train, y_test = get_mnist_data()

class supportVectorMachine(BAYESOPT):

    def __init__(self, optimize=True, runBestModel=False):
        self.optimize = optimize
        self.runBestModel = runBestModel
        self.patienceSearch = 1
        self.patienceFinal = 2

        self.bounds = [{'id':0, 'name': 'C', 'type': 'continuous', 'domain': (0,100)},
                  {'id':1, 'name': 'kernel', 'type': 'discrete', 'domain': (0,1,2,3)}]

        self.max_iter = 5
        self.filename = 'sklearn'
        if self.optimize:
            self._runBayesOpt()

        if runBestModel:
            self.mean, self.std = self.final_evaluation(self.opt, num_runs=2)


    def setHyperParameter(self, variable, patience=2):
        self.C = float(variable[self.map['C']])
        self.kernel= int(variable[self.map['kernel']])
        if self.kernel == 0:
            self.kernel = 'linear'
        if self.kernel == 1:
            self.kernel = 'rbf'
        if self.kernel == 2:
            self.kernel = 'poly'
        if self.kernel == 3:
            self.kernel = 'sigmoid'


        self.__model = self.ml_model()


    def ml_model(self):
        # Build Model
        model = svm.SVC(kernel=self.kernel,C=self.C,gamma=0.001,max_iter=500)

        return model

    # fit mnist model
    def model_fit(self):

        self.__model.fit(x_train,y_train)

    # evaluate mnist model
    def evaluation(self):
        self.model_fit()

        evaluation = -metrics.accuracy_score(y_test,self.__model.predict(x_test))
        return evaluation

########### initialize the standard model
model = supportVectorMachine(optimize=True,runBestModel=True)

print('Mean Over runs with Best Model:', model.mean)
print('Std Over runs with Best Model:', model.std)
