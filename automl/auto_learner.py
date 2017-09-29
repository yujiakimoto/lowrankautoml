from error_matrix import ErrorMatrix
from ensemble import Ensemble
from model import Model
import multiprocessing as mp
import numpy as np
import time

class AutoLearner:
    
    def __init__(self, selected_algorithms='all', selected_hyperparameters='default', ensemble_size=3, 
                 ensemble_method='Logit', error_matrix_values='default', verbose=True):
        """instantiates an AutoLearner object """

        self.error_matrix = ErrorMatrix(selected_algorithms, selected_hyperparameters, ensemble_size, error_matrix_values, verbose)
        """error matrix defined for specific dataset"""
        
        self.ensemble = Ensemble(ensemble_size=ensemble_size, ensemble_method=ensemble_method, verbose=verbose)
        """instantiate empty ensemble object"""
        
    def fit(self, train_features, train_labels, categorical):        
        """fit the model to a given training feature and label"""
        # preprocessing
        self.error_matrix.add_dataset(train_features, train_labels)
        for model in self.error_matrix.best_algorithms(train_features, train_labels):
            self.ensemble.add_learner(model)
        self.ensemble.bayesian_optimize()
        self.ensemble.fit_base_learners(train_features, train_labels)
        self.ensemble.fit_stacked_learner(train_features, train_labels)
    
    def refit(self, train_features, train_labels, categorical):
        """refits the autolearner object to a newly provided training set"""
        # preprocessing
        self.ensemble.fit_base_learners(train_features, train_labels)
        self.ensemble.fit_stacked_learner(train_features, train_labels)

    def predict(self, test_features):
        """returns predictions of the autolearner object on newly provided test set"""
        p = mp.Pool()
        a = [p.apply_async(Model.predict, args=(model, test_features)) for model in self.ensemble.base_learners]
        p.close()
        p.join()
        predictions = ()
        for i in a:
            predictions += (i.get(),)
        test_second_layer_matrix = np.matrix.transpose(np.stack(predictions))     
        return self.ensemble.model.predict(test_second_layer_matrix)