import os
import pickle
import random
from collections import OrderedDict

import GPy
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pds

"""
Parent Class For the hyperparameter search whose methods are model aganostic 
Correspondence to: Vaisakh Shaj (vaisakhs.shaj@gmail.com)
"""

class BAYESOPT():

    def __init__(self, optimize=True, runBestModel=False):
        self.optimize = optimize
        self.runBestModel = runBestModel
        self.patienceSearch = 5
        self.patienceFinal = 10
        self.bounds = []
        self.max_iter = 5
        self.filename='trial'
        if self.optimize:
            self._runBayesOpt()

        if runBestModel:
            self.mean, self.std = self.final_evaluation(self.opt, num_runs=2)


    def _map(self):
        '''
        Maps the variable names in bounds to the ids so that
        you don't have to keep track of which varible corresponds
        to which id
        :return a dictionary of mapping id -> variable name
        '''
        map = {}
        for i in range(len(self.bounds)):
            map[self.bounds[i]['name']]=self.bounds[i]['id']
        return map

    def _runBayesOpt(self):
        '''
        :return:
        '''
        self.map = self._map()
        self.opt = GPyOpt.methods.BayesianOptimization(f=self.objective, domain=self.bounds)
        # #### Running optimization
        self.opt.run_optimization(max_iter=self.max_iter,report_file=os.getcwd() + '/output/hyper/'+self.filename+'_report.txt', evaluations_file=os.getcwd() + '/output/hyper/'+self.filename+'_evaluations.txt', models_file=os.getcwd() + '/output/hyper/'+self.filename+'_models.txt')
        self.opt.plot_convergence(os.getcwd() + '/output/hyper/'+self.filename+'.pdf')



    def setHyperParameter(self, variable, patience=2):
        '''
        :param variable: list of variable values in the order as in self.bounds
        :param patience:
        :return:
        '''
        self.__model = self.ml_model()
        self.patience = patience

    # mnist model
    def ml_model(self):
        # Build Model

        model = None

        return model

    # fit mnist model
    def model_fit(self):
        early_stopping = EarlyStopping(patience=self.patience, verbose=1)
        filepath = os.getcwd() + "/output/weights/"+self.filename+".hdf5"
        checkpoint = None
        callbacks_list = [checkpoint, early_stopping]

        #self.__model.load_weights(filepath)

    # evaluate mnist model
    def evaluation(self):
        self.model_fit()

        evaluation = None
        return evaluation

    def objective(self, x):
        print(x)
        x = np.squeeze(x, 0)
        self.setHyperParameter(x, patience=self.patienceSearch)
        evaluation = self.evaluation()
        print("Accuracy", evaluation)
        return evaluation

    def final_evaluation(self, opt, num_runs):
        '''
        :param savedOpt: saved GpyOptimzer object
        :param num_runs: num of runs to have
        :return: mean and standard deviation over runs
        '''
        evaluations = []
        for run in range(num_runs):
            self.setHyperParameter(opt.x_opt, patience=self.patienceFinal)

            evaluations.append(self.evaluation())
        print(evaluations)

        return np.mean(evaluations), np.std(evaluations)
