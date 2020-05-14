import os
import pickle
import random
from collections import OrderedDict
from abc import ABC, abstractmethod

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

class BAYESOPT(ABC):

    def __init__(self, optimize=True, runBestModel=False):
        '''
        :param optimize:
        :param runBestModel:
        '''
        self.optimize = optimize
        self.runBestModel = runBestModel
        self.patienceSearch = 5
        self.patienceFinal = 10
        self.bounds = []
        self.max_iter = 5 #Number of  maximum models to run Bayesian Search Over
        self.num_runs = 2 #Number of runs to run the best model to calulate mean & std
        self.filename='trial'
        if self.optimize:
            self._runBayesOpt()

        if runBestModel:
            self.mean, self.std = self.final_evaluation(self.opt, num_runs=self.num_runs)


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


    @abstractmethod
    def setHyperParameter(self, variable, patience=2):
        '''
        :param variable: list of variable values in the order as in self.bounds
        :param patience: patience for early stopping
        :return:
        '''
        pass

    # mnist model
    @abstractmethod
    def ml_model(self):
        # Build Model and return model
        pass

    # fit mnist model
    @abstractmethod
    def model_fit(self):
        pass


        #self.__model.load_weights(filepath)

    # evaluate mnist model
    def evaluation(self):
        pass

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
