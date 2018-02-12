from ensemble import Ensemble
from model import Model

import ML_algorithms as ml
import low_rank_models as lrm
import util
import time

import pandas as pd
import numpy as np
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool
from scipy.linalg import qr
import util

class ErrorMatrix:
    
    def __init__(self, selected_algorithms, selected_hyperparameters, ensemble_size, error_matrix_values, verbose, n_cores, num_folds=10, fit_best_predicted_models=False):
        """instantiates an error matrix object given values"""
        
        self.values_default = (error_matrix_values == 'default')
        """whether the default values of the error matrix were used"""
        
        self.num_folds = num_folds
        """ the number of folds when calculating entries in the error matrix"""
        
        self.headings = util.generate_headings(selected_algorithms, selected_hyperparameters)
        
        self.fit_best_predicted_models = fit_best_predicted_models
        
        util.HEADINGS = self.headings
        """"generate a list of 3 lists each of length equal to the number of columns in the dataset: the first
            containing the algorithm type (string), the second containing the hyperparameter names (tuple), and the
            third containing the hyperparameter values (tuple)"""
        

        if not self.values_default:
            self.values = error_matrix_values
            """if error matrix values are given, use given values"""
        else:
            self.values = pd.read_csv('../automl/default/error_matrix.csv', index_col=0).values
#            self.values = em[:, np.in1d(getattr(util, 'generate_headings')(all_algs, 'default')[0], np.array(self.selected_algorithms))]
            """if error matrix values are not given, use default"""
        
 
        self.rank = lrm.approx_rank(self.values)
        """calculate the approximate rank of the error matrix: defined as the number of singular values
        that are at least 3/100th of the largest singular value"""       

        X,Y,Vt = lrm.pca(self.values)
        
        self.indices_qr = lrm.pivoted_qr(Vt)[:self.rank]
        """which indices of the error matrix we computed - identified using pivoted QR factorization of
        the error matrix values"""      
        
        self.new_row = np.zeros((1, self.values.shape[1]))
        """new row of error matrix corresponding to user data"""
        
        self.ensemble_size = ensemble_size
        """number of the algorithms to select for the ensemble"""
        
        self.verbose = verbose
        """whether to generate print statements"""  
        
        self.n_cores = n_cores
        """the number of cores to use per autolearner object"""
    
#    def generate_settings(self, i):
#        settings = {'algorithm': self.headings[0][i],
#                    'hyperparameters': {self.headings[1][i][j]: self.headings[2][i][j] for j in range(len(self.headings[1][i]))}}
#        if list(settings['hyperparameters'].keys())[0] == '':
#                settings['hyperparameters'] = {}
#        return settings

    """compute error value for individual entry in row corresponding to new dataset"""
    def compute_entry(self, features, labels, index, num_folds, verbose=False):
        settings = util.generate_setting_single_model(index)
        model = Model(settings=settings, num_folds=num_folds, verbose=verbose)
        model.fit(features, labels)
        return model


    def add_dataset(self, train_features, train_labels):
        """compute the row corresponding to a new dataset, in which entries in the pivot columns are computed by actually fitting the corresponding model on the new dataset, and the other entries are computed through low rank approximation"""
        p1 = mp.Pool(self.n_cores)
        QR_models = [p1.apply_async(self.compute_entry, args=[train_features, train_labels, i, self.num_folds]) for i in self.indices_qr]
        p1.close()
        p1.join()
    
        for i, model in enumerate(QR_models):
            self.new_row[:, self.indices_qr[i]] = model.get().error

        if self.fit_best_predicted_models:
    
            candidate_idx = self.new_row[0].argsort()[:5] #the best models in low rank approx
    
    
            p2 = mp.Pool(N_CORES)
            candidate_models = [p2.apply_async(self.compute_entry, args=[x_train, y_train, i, self.num_folds]) for i in candidate_idx if i not in self.indices_qr] #compute true errors of the best models in low rank approx
            p2.close()
            p2.join()
                            
            for i, model in enumerate(candidate_models):
                self.new_row[:, candidate_idx[i]] = model.get().error
#                secondlayercolumns += (model.get().cv_predictions, )
#                base_learners.append(model.get())

        approx = lrm.low_rank_approximation(self.values, self.new_row, self.indices_qr)
        self.new_row = np.copy(approx)

            
    def best_algorithms(self, train_features, train_labels):
        """returns the best algorithms and corresponding hyperparameter settings: a list of dicts"""
        lowest_indices = self.new_row.argsort()[0][:self.ensemble_size]
        ensemble_settings = [util.generate_setting_single_model(i) for i in lowest_indices]
        best_models = []
        for i in range(self.ensemble_size):
            model = Model(settings=ensemble_settings[i], verbose=self.verbose)
            model.add_training_data(train_features, train_labels)
            best_models.append(model)
        return best_models

