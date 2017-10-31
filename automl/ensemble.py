from model import Model

import multiprocessing as mp
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

class Ensemble:
    
    def __init__(self, ensemble_size, ensemble_method, verbose, n_cores):
        """instantiates an ensemble object given a list of model objects to ensemble"""
    
        self.current_size = 0
        """current size of the ensemble, an int"""
        
        self.fitted = False
        """whether Ensemble has finished fitting"""
        
        self.second_layer_matrix = None
        """the second layer matrix that will be used to fit stacking method"""
        
        self.ensemble_method = ensemble_method
        """the stacking algorithm used in the second layer"""
        
        # self.model = Model({'algorithm':ensemble_method, 'hyperparameters':{'C': 1.0}}, verbose=verbose)
        self.model = Model({'algorithm':'GB', 'hyperparameters':{'learning_rate': 0.1}}, verbose=verbose)
        
        self.base_learners = []        
        """a list of base learner candidates for constructing the ensemble"""   
        
        self.n_cores = n_cores
        """the number of cores to use per autolearner object"""
        
    def add_learner(self, model):
        """method to add a model object as an ensemble"""
        self.base_learners.append(model)
        self.current_size += 1
    
    def get_base_learner_params(self):
        base_params = {model.algorithm:model.hyperparameters for model in self.base_learners}
        return base_params
    
    def bayesian_optimize(self):
        """conduct bayesian optimization on each individual model within the ensemble"""
        p = Pool(self.n_cores)
        optimized_models = p.map(Model.bayesian_optimize, self.base_learners)
        for i in range(self.current_size):
            self.base_learners[i] = optimized_models[i]
    
    def fit_base_learners(self, train_features, train_labels):        
        """fit the base learners to given features and labels"""
        p = mp.Pool(self.n_cores)
        a = [p.apply_async(Model.fit, args=(model, train_features, train_labels)) for model in self.base_learners]
        p.close()
        p.join()
        for i in range(len(a)):
            self.base_learners[i] = a[i].get()
        predictions = tuple(model.cv_predictions for model in self.base_learners)
        self.second_layer_matrix = np.matrix.transpose(np.stack(predictions))            
        
    def fit_stacked_learner(self, train_features, train_labels):
        """fit the stacked learner of the ensemble"""
        self.model.add_training_data(self.second_layer_matrix, train_labels)
        self.model = self.model.bayesian_optimize()
        self.model.fit(self.second_layer_matrix, train_labels)
        self.fitted = True
        
    def predict(self, test_features):
        """returns predictions of ensemble on newly provided test set"""
        return self.model.predict(test_features)