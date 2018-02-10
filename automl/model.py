import ML_algorithms as ml
import util
import numpy as np

from smac.configspace import ConfigurationSpace
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

class Model:
    
    def __init__(self, settings={'algorithm':None, 'hyperparameters':None}, num_folds=10, verbose=True, train_features=None,
                train_labels=None):
        """instantiates a model object given an algorithm type, hyperparameter settings and
        a number of folds for cross validation"""
        
        self.algorithm = settings['algorithm']
        """algorithm type, a string"""
        
        self.hyperparameters = settings['hyperparameters']
        """hyperparameter settings, a dict (keys=hyperparameter name, values=hyperparameter values)"""
        
        self.classifier = None
        """sklearn classifier associated with this model"""
        
        self.num_folds = num_folds
        """the number of folds for k-fold cross validation"""
        
        self.fitted = False
        """whether or not the model has been trained"""
        
        self.bayesian_optimized = False
        """whether or not the model's hyperparameters have been tuned"""
        
        self.train_features = train_features
        self.train_labels = train_labels
        """training dataset"""
        
        self.error = None
        """k-fold cross validation error for a given dataset"""
        
        self.cv_predictions = None
        """k-fold predictions for a given dataset"""
        
        self.verbose = verbose
        """whether to generate print statements"""
        
    def fit(self, train_features, train_labels):        
        """fit the model to given training features and labels"""
        self.train_features = train_features
        self.train_labels = train_labels
        self.error, self.cv_predictions, self.classifier = getattr(ml, self.algorithm)(self.train_features, self.train_labels,                                                                                            verbose=self.verbose, **self.hyperparameters)
        self.fitted = True
        return self
    
    def predict(self, test_features):
        """return predictions of ensemble on newly provided test set"""
        return self.classifier.predict(test_features)
        
    def bayesian_optimize(self):    
        """conduct Bayesian optimization on the hyperparameters, starting at current values""" 
        if self.algorithm in ['GNB','Perceptron']:
            return self
        else:
            cs = ConfigurationSpace()
            cs.add_hyperparameters(list(getattr(util, self.algorithm + '_range')(self.hyperparameters).values()))
            #set runcount-limit in Bayesian optimization
            if self.algorithm == 'kNN':
                if self.hyperparameters['k'] == 1: num = 3
                else: num = 5
            else: num = 10
            scenario = Scenario({'run_obj': 'quality', 'runcount-limit': num, 'cs': cs, 'deterministic': 'true',  'memory_limit': None})
            smac = SMAC(scenario=scenario, rng=np.random.RandomState(100), tae_runner=self.error_function)
            try:
                incumbent = smac.optimize()
            finally:
                incumbent = smac.solver.incumbent
            self.error = smac.get_tae_runner().run(incumbent, 1)[1]
            self.hyperparameters = incumbent.get_dictionary()
            self.bayesian_optimized = True
            return self
        
    def error_function(self, hyperparameters):
        """function on which to conduct Bayesian optimization"""
        return getattr(ml, self.algorithm)(self.train_features, self.train_labels, num_splits=3, verbose=self.verbose, **hyperparameters)[0]
    
    def add_training_data(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels
    
