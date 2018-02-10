import numpy as np
from itertools import product
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

#whether to optimize the models' hyperparameters over the entire ranges given by the columns
WIDE_HYPERPARAMETER_RANGE = True


def generate_headings(selected_algorithms, selected_hyperparameters):
        """generate headings of error matrix

        Returns:
            alg_types: a list containing algorithms corresponding to each column, e.g. ['kNN', 'kNN', 'CART'];
            hyperparameter_names: a list containing tuples, with first entries of each tuple being the
            hyperparameter name corresponding to that column;
            hyperparameter_values: a list containing tuples, with first entries of each tuple
            being the hyperparameter value corresponding to that column.

        """

        alg_types, hyperparameter_names, hyperparameter_values = [], [], []

        all_algorithms = ['kNN', 'CART', 'GB', 'lSVM', 'kSVM', 'Logit', 'Percep', 'GNB', 'MNB', 'RF', 'ExtraTrees', 'ABT', 'MLP']
        all_hyperparameters = [{'k': [1, 3, 5, 7, 9, 11, 13, 15], 'P':[1, 2]},
                               {'min_samples_split': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 0.01, 0.001, 0.0001, 0.00001]},
                               {'learning_rate': [0.001, 0.01, 0.025, 0.05, 0.1, 0.25,  0.5], 'max_depth':[3, 6], 'max_features': [None, 'log2']},
                               {'C': [0.125, 0.25, 0.5, 0.75, 1, 2, 4, 8, 16]},
                               {'C': [0.125, 0.25, 0.5, 0.75, 1, 2, 4, 8, 16], 'kernel': ['rbf', 'poly'], 'coef0': [0, 10]},
                               {'C': [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4], 'solver': ['liblinear', 'saga'], 'penalty': ['l1', 'l2']},
                               {},
                               {},
                               {},
                               {'min_samples_split':[2,4,8,16,32,64,128,256,512,1024, 0.1, 0.01, 0.001, 0.0001, 0.00001], 'criterion': ['gini', 'entropy']},
                               {'min_samples_split':[2,4,8,16,32,64,128,256,512,1024,0.1,0.01,0.001,0.0001,0.00001], 'criterion':['gini', 'entropy']},
                               {'n_estimators':[50,100], 'learning_rate':[1.0,1.5,2.0,2.5,3.0]}, {'learning_rate':[0.0001,0.001,0.01], 'solver':['sgd', 'adam'], 'alpha':[0.0001, 0.01]}]

        less_algorithms = ['kNN', 'CART', 'GB', 'lSVM', 'kSVM', 'Logit', 'Percep', 'ABT', 'GNB', 'RF']
        less_hyperparameters = [{'k': [1, 3, 5, 7, 9, 11, 13, 15], 'P': [2]},
                                {'min_samples_split': [0.01, 0.001, 0.0001, 0.00001]},
                                {'learning_rate': [0.1, 0.01, 0.001]},
                                {'C': [0.25, 0.5, 0.75, 1.0, 2.0]},
                                {'C': [0.25, 0.5, 0.75, 1.0, 2.0]},
                                {'C': [0.25, 0.5, 0.75, 1.0, 2.0]},
                                {},
                                {'n_estimators': [50, 100], 'learning_rate': [1.0, 2.0]},
                                {},
                                {'min_samples_split': [0.1, 0.01, 0.001, 0.0001, 0.00001]}]

        if selected_algorithms == 'all':
            selected_algorithms = all_algorithms        # large error matrix
        elif selected_algorithms == 'less':
            selected_algorithms = less_algorithms       # small error matrix

        if selected_hyperparameters == 'default':       # hyperparameters of large error matrix
            selected_hyperparameters = np.array(all_hyperparameters)[np.in1d(np.array(all_algorithms), np.array(selected_algorithms))]
        elif selected_hyperparameters == 'less':
            selected_hyperparameters = np.array(less_hyperparameters)[np.in1d(np.array(less_algorithms), np.array(selected_algorithms))]


        # how many algorithms are selected
        num_alg_types = len(selected_algorithms)
        # how many hyperparameters does each of the selected algorithm have
        num_hyperparams_alg = [len(list(selected_hyperparameters[i].keys())) for i in range(num_alg_types)]
        # how many cases does each of the hyperparameter settings have

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

#generate the setting of the ith model in headings list
def generate_setting_single_model(HEADINGS, i):
    
    single_model_setting = {'algorithm':HEADINGS[0][i], 'hyperparameters':{HEADINGS[1][i][j]:HEADINGS[2][i][j] for j in range(len(HEADINGS[1][i]))}}
    
    if list(single_model_setting['hyperparameters'].keys())[0] == '':
        single_model_setting['hyperparameters'] = {}
    
    return single_model_setting


#generate the setting of the ith model in headings list
def generate_settings(HEADINGS):
    
    settings = []
    
    for i in range(len(HEADINGS[0][i])):
        
        single_model_setting = {'algorithm':HEADINGS[0][i], 'hyperparameters':{HEADINGS[1][i][j]:HEADINGS[2][i][j] for j in range(len(HEADINGS[1][i]))}}
        
        if list(single_model_setting['hyperparameters'].keys())[0] == '':
            single_model_setting['hyperparameters'] = {}
        
        settings.append(single_model_setting)
    
    return settings


# generate the setting of the ith model in headings list
def generate_setting_single_model(HEADINGS, i):
    
    single_model_setting = {'algorithm':HEADINGS[0][i], 'hyperparameters':{HEADINGS[1][i][j]:HEADINGS[2][i][j] for j in range(len(HEADINGS[1][i]))}}
        
    if list(single_model_setting['hyperparameters'].keys())[0] == '':
        single_model_setting['hyperparameters'] = {}
        
    return single_model_setting


# generate the setting of the ith model in headings list
def generate_settings(HEADINGS):
    
    settings = []
    
    for i in range(len(HEADINGS[0][i])):
    
        single_model_setting = {'algorithm':HEADINGS[0][i], 'hyperparameters':{HEADINGS[1][i][j]:HEADINGS[2][i][j] for j in range(len(HEADINGS[1][i]))}}
    
        if list(single_model_setting['hyperparameters'].keys())[0] == '':
            single_model_setting['hyperparameters'] = {}

        settings.append(single_model_setting)
    
    return settings


if WIDE_HYPERPARAMETER_RANGE:
    def kNN_range(hyperparameters):        
        return {'k': UniformIntegerHyperparameter('k', max(1, hyperparameters['k']-2), max(all_hyperparameters[all_algorithms.index('kNN')]['k']), default_value=hyperparameters['k'])}


    def CART_range(hyperparameters):
        return {'min_samples_split': UniformFloatHyperparameter('min_samples_split', min(all_hyperparameters[all_algorithms.index('CART')]['min_samples_split']), max(all_hyperparameters[all_algorithms.index('CART')]['min_samples_split']), default_value=hyperparameters['min_samples_split'])}

    def RF_range(hyperparameters):
        return {'min_samples_split': UniformFloatHyperparameter('min_samples_split', min(all_hyperparameters[all_algorithms.index('RF')]['min_samples_split']), max(all_hyperparameters[all_algorithms.index('RF')]['min_samples_split']), default_value=hyperparameters['min_samples_split'])}

    def ExtraTrees_range(hyperparameters):
        return {'min_samples_split': UniformFloatHyperparameter('min_samples_split', min(all_hyperparameters[all_algorithms.index('RF')]['min_samples_split']), max(all_hyperparameters[all_algorithms.index('RF')]['min_samples_split']), default_value=hyperparameters['min_samples_split'])}


    def GB_range(hyperparameters): 
        return {'learning_rate': UniformFloatHyperparameter('learning_rate', min(all_hyperparameters[all_algorithms.index('GB')]['learning_rate']), max(all_hyperparameters[all_algorithms.index('GB')]['learning_rate']), default_value=hyperparameters['learning_rate'])}

    def lSVM_range(hyperparameters):  
        return {'C': UniformFloatHyperparameter('C', min(all_hyperparameters[all_algorithms.index('lSVM')]['C']), max(all_hyperparameters[all_algorithms.index('lSVM')]['C']), default_value=hyperparameters['C'])}

    def kSVM_range(hyperparameters):  
        return {'C': UniformFloatHyperparameter('C', min(all_hyperparameters[all_algorithms.index('kSVM')]['C']), max(all_hyperparameters[all_algorithms.index('kSVM')]['C']), default_value=hyperparameters['C'])}

    def Logit_range(hyperparameters):
        return {'C': UniformFloatHyperparameter('C', min(all_hyperparameters[all_algorithms.index('Logit')]['C']), max(all_hyperparameters[all_algorithms.index('Logit')]['C']), default_value=hyperparameters['C'])}

    def Perceptron_range(hyperparameters):
        return {}

    def Adaboost_range(hyperparameters):
        return {'n_estimators': UniformIntegerHyperparameter('n_estimators', int(min(all_hyperparameters[all_algorithms.index('Adaboost')]['n_estimators'])), int(max(all_hyperparameters[all_algorithms.index('Adaboost')]['n_estimators'])), default_value=int(hyperparameters['n_estimators'])),
               'learning_rate': UniformFloatHyperparameter('learning_rate', min(all_hyperparameters[all_algorithms.index('Adaboost')]['learning_rate']), max(all_hyperparameters[all_algorithms.index('Adaboost')]['learning_rate']), default_value=hyperparameters['learning_rate'])}

    def GNB_range(hyperparameters):
        return {}


else:
    def kNN_range(hyperparameters):
        return {'k': UniformIntegerHyperparameter('k', max(1, hyperparameters['k']-2), hyperparameters['k']+2, default=hyperparameters['k'])}

    def CART_range(hyperparameters):
        return {'min_samples_split': UniformFloatHyperparameter('min_samples_split', 0.1*hyperparameters['min_samples_split'], 10*hyperparameters['min_samples_split'], default=hyperparameters['min_samples_split'])}

    def RF_range(hyperparameters):
         return {'min_samples_split': UniformFloatHyperparameter('min_samples_split', 0.1*hyperparameters['min_samples_split'], 10*hyperparameters['min_samples_split'], default=hyperparameters['min_samples_split'])}

    def ExtraTrees_range(hyperparameters):
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
