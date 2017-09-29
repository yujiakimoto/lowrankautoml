from ensemble import Ensemble
from model import Model

import ML_algorithms as ml
import low_rank_models as lrm
import util
import time

import pandas as pd
import numpy as np
import multiprocessing as mp
from scipy.linalg import qr

class ErrorMatrix:
    
    def __init__(self, selected_algorithms, selected_hyperparameters, ensemble_size, error_matrix_values, verbose):
        """instantiates an error matrix object given values"""
        
        self.values_default = (error_matrix_values == 'default')
        """whether the default values of the error matrix were used"""
        
        all_algs = ['kNN', 'CART', 'GB', 'lSVM', 'kSVM', 'Logit', 'Perceptron', 'Adaboost', 'GNB', 'RF']
        if selected_algorithms == 'all':
            self.selected_algorithms = all_algs
        
        if self.values_default == False:
            self.values = error_matrix_values
            """if error matrix values are given, use given values"""
        else:
            em = pd.read_csv('default/error_matrix.csv', index_col=0).values
            self.values = em[:, np.in1d(getattr(util, 'generate_headings')(all_algs, 'default')[0], np.array(self.selected_algorithms))]
            """if error matrix values are not given, use default"""   
            
        self.headings = getattr(util, 'generate_headings')(self.selected_algorithms, selected_hyperparameters)
        """"generate a list of 3 lists each of length equal to the number of columns in the dataset: the first 
        containing the algorithm type (string), the second containing the hyperparameter names (tuple), and the 
        third containing the hyperparameter values (tuple)"""
 
        self.rank = lrm.approx_rank(self.values)
        """calculate the approximate rank of the error matrix: defined as the number of singular values
        that are at least 3/100th of the largest singular value"""       
        
        self.index_qr = lrm.pivoted_qr(self.values)
        """indices of error matrix columns, sorted in order from most orthogonal to least"""        
        
        self.computed_indices = self.index_qr[0:self.rank]
        """which indices of the error matrix we computed - identified using pivoted QR factorization of 
        the error matrix values"""      
        
        self.new_row = np.zeros((1, self.values.shape[1]))
        """new row of error matrix corresponding to user data"""
        
        self.ensemble_size = ensemble_size
        """number of the algorithms to select for the ensemble"""
        
        self.verbose = verbose
        """whether to generate print statements"""  
    
    def generate_settings(self, i):
        settings = {'algorithm':self.headings[0][i], 
                        'hyperparameters':{self.headings[1][i][j]:self.headings[2][i][j] for j in range(len(self.headings[1][i]))}}
        if list(settings['hyperparameters'].keys())[0] == '':
                settings['hyperparameters'] = {}
        return settings
    
    def add_dataset(self, train_features, train_labels):
        """compute error values for r entries as identified by pivoted QR factorization"""
        p1 = mp.Pool()
        a1 = [p1.apply_async(self.compute_entry, args=[train_features, train_labels, i]) for i in self.computed_indices]
        p1.close()
        p1.join()
        for i in range(len(self.computed_indices)):
            self.new_row[0, self.computed_indices[i]] = a1[i].get().error
        approx_row = lrm.low_rank_approximation(self.values, self.new_row, self.computed_indices)
        unknown = np.setdiff1d(np.arange(self.values.shape[1]), self.computed_indices)
        self.new_row[:,unknown] = approx_row[:,unknown]
        candidate_indices = self.new_row.argsort()[0][:5]
        p2 = mp.Pool()
        a2 = [p2.apply_async(self.compute_entry, args=(train_features, train_labels, i)) for i in candidate_indices if
             i not in self.computed_indices]
        for i in range(len(a2)):
            self.new_row[0, candidate_indices[i]] = a2[i].get().error
        approx_row = lrm.low_rank_approximation(self.values, self.new_row, np.union1d(self.computed_indices, candidate_indices))
        unknown = np.setdiff1d(np.arange(self.values.shape[1]), np.union1d(self.computed_indices, candidate_indices))
        self.new_row[:,unknown] = approx_row[:,unknown]
        p2.close()
        p2.join()
            
    def compute_entry(self, train_features, train_labels, index):
        """compute error value for individual entry in row corresponding to new dataset"""
        settings = self.generate_settings(index)              
        model = Model(settings=settings, verbose=self.verbose)
        model.fit(train_features, train_labels) 
        return model
    
    def best_algorithms(self, train_features, train_labels):
        """returns the best algorithms and corresponding hyperparameter settings: a list of dicts"""       
        lowest_indices = self.new_row.argsort()[0][:self.ensemble_size]
        ensemble_settings = [self.generate_settings(i) for i in lowest_indices]
        best_models = []
        for i in range(self.ensemble_size):
            model = Model(settings=ensemble_settings[i], verbose=self.verbose)
            model.add_training_data(train_features, train_labels)
            best_models.append(model)
        return best_models