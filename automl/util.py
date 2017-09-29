import numpy as np
from itertools import product
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

def generate_headings(selected_algorithms, selected_hyperparameters):
        """generate headings of error matrix"""
        
        alg_types, hyperparameter_names, hyperparameter_values = [], [], []
        all_algorithms = ['kNN', 'CART', 'GB', 'lSVM', 'kSVM', 'Logit', 'Perceptron', 'Adaboost', 'GNB', 'RF']
        all_hyperparameters = [{'k':[1,3,5,7,9,11]}, {'min_samples_split':[0.01,0.001,0.0001,0.00001]},
                               {'learning_rate':[0.1,0.01,0.001]}, {'C':[0.25,0.5,0.75,1.0,2.0]}, 
                               {'C':[0.25,0.5,0.75,1.0,2.0]}, {'C':[0.25,0.5,0.75,1.0,2.0]}, 
                               {}, {'n_estimators':[50,100], 'learning_rate':[1.0,2.0]}, {},
                               {'min_samples_split':[0.1,0.01,0.001,0.0001,0.00001]}]
               
        if selected_hyperparameters == 'default':
            selected_hyperparameters = np.array(all_hyperparameters)[np.in1d(np.array(all_algorithms), np.array(selected_algorithms))]
            
        num_alg_types = len(selected_algorithms)
        num_hyperparams_alg = [len(list(selected_hyperparameters[i].keys())) for i in range(num_alg_types)]
        num_settings_hyp_alg = [[len(selected_hyperparameters[i][list(selected_hyperparameters[i].keys())[j]]) 
                                 for j in range(num_hyperparams_alg[i])] for i in range(num_alg_types)]

        for i in range(num_alg_types):
            if num_hyperparams_alg[i] == 0:
                alg_types.append(selected_algorithms[i])
                hyperparameter_names.append(tuple(('',)))
                hyperparameter_values.append(tuple((None,)))
            elif num_hyperparams_alg[i] == 1:
                for k in range(num_settings_hyp_alg[i][0]):
                    alg_types.append(selected_algorithms[i])
                    hyperparameter_names.append(tuple((list(selected_hyperparameters[i].keys())[0],)))
                    hyperparameter_values.append(tuple((list(selected_hyperparameters[i].values())[0][k],)))
            else:
                for k in range(np.prod(num_settings_hyp_alg[i])):
                    alg_types.append(selected_algorithms[i])
                    hyperparameter_names.append(tuple(selected_hyperparameters[i].keys()))
                    hyperparameter_values.append(tuple((list(product(*list(selected_hyperparameters[i].values())))[k])))
                    
        return [alg_types, hyperparameter_names, hyperparameter_values]

def kNN_range(hyperparameters):
    return {'k': UniformIntegerHyperparameter('k', max(1, hyperparameters['k']-2), hyperparameters['k']+2, default=hyperparameters['k'])}

def CART_range(hyperparameters):
    return {'min_samples_split': UniformFloatHyperparameter('min_samples_split', 0.1*hyperparameters['min_samples_split'], 10*hyperparameters['min_samples_split'], default=hyperparameters['min_samples_split'])}

def RF_range(hyperparameters):
     return {'min_samples_split': UniformFloatHyperparameter('min_samples_split', 0.1*hyperparameters['min_samples_split'], 10*hyperparameters['min_samples_split'], default=hyperparameters['min_samples_split'])}

def GB_range(hyperparameters):
    return {'learning_rate': UniformFloatHyperparameter('learning_rate', 0.1*hyperparameters['learning_rate'], 10*hyperparameters['learning_rate'], default=hyperparameters['learning_rate'])}

def lSVM_range(hyperparameters):
    return {'C': UniformFloatHyperparameter('C', 0.5*hyperparameters['C'], 2*hyperparameters['C'], default=hyperparameters['C'])}

def kSVM_range(hyperparameters):
    return {'C': UniformFloatHyperparameter('C', 0.5*hyperparameters['C'], 2*hyperparameters['C'], default=hyperparameters['C'])}

def Logit_range(hyperparameters):
    return {'C': UniformFloatHyperparameter('C', 0.5*hyperparameters['C'], 2*hyperparameters['C'], default=hyperparameters['C'])}

def Perceptron_range(hyperparameters):
    return {}

def Adaboost_range(hyperparameters):
    return {'n_estimators': UniformIntegerHyperparameter('n_estimators', int(0.5*hyperparameters['n_estimators']), int(2*hyperparameters['n_estimators']), default=int(hyperparameters['n_estimators'])),
           'learning_rate': UniformFloatHyperparameter('learning_rate', 0.5*hyperparameters['learning_rate'], 2*hyperparameters['learning_rate'], default=hyperparameters['learning_rate'])}

def GNB_range(hyperparameters):
    return {}
