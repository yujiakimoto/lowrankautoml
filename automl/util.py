import numpy as np

from itertools import product
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

# whether to optimize the models' hyperparameters over the entire ranges given by the columns
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

        all_algorithms = ['kNN', 'CART', 'GB', 'lSVM', 'kSVM', 'Logit', 'Perceptron', 'GNB', 'MNB', 'RF', 'ExtraTrees', 'Adaboost', 'MLP']
        all_hyperparameters = [{'k': [1, 3, 5, 7, 9, 11, 13, 15], 'p':[1, 2]},
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

        less_algorithms = ['kNN', 'CART', 'GB', 'lSVM', 'kSVM', 'Logit', 'Perceptron', 'Adaboost', 'GNB', 'RF']
        less_hyperparameters = [{'k': [1, 3, 5, 7, 9, 11, 13, 15], 'p': [2]},
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


        if selected_hyperparameters == 'default':       # hyperparameters of the large error matrix
            selected_hyperparameters = np.array(all_hyperparameters)[np.in1d(np.array(all_algorithms), np.array(selected_algorithms))]
        elif selected_hyperparameters == 'less':        # hyperparameters of the small error matrix
            selected_hyperparameters = np.array(less_hyperparameters)[np.in1d(np.array(less_algorithms), np.array(selected_algorithms))]


        # how many algorithms are selected
        num_alg_types = len(selected_algorithms)
        # how many hyperparameters does each of the selected algorithm have
        num_hyperparams_alg = [len(list(selected_hyperparameters[i].keys())) for i in range(num_alg_types)]
        # how many cases does each of the hyperparameter settings have
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


# generate the setting of the ith model in the list HEADINGS
def generate_setting_single_model(i):
    
    single_model_setting = {'algorithm':HEADINGS[0][i], 'hyperparameters':{HEADINGS[1][i][j]:HEADINGS[2][i][j] for j in range(len(HEADINGS[1][i]))}}
    
    if list(single_model_setting['hyperparameters'].keys())[0] == '':
        single_model_setting['hyperparameters'] = {}
    
    return single_model_setting

# generate all settings of the models in the list HEADINGS
def generate_settings(HEADINGS):
    
    settings = []
    
    for i in range(len(HEADINGS[0][i])):
        
        single_model_setting = {'algorithm':HEADINGS[0][i], 'hyperparameters':{HEADINGS[1][i][j]:HEADINGS[2][i][j] for j in range(len(HEADINGS[1][i]))}}
        
        if list(single_model_setting['hyperparameters'].keys())[0] == '':
            single_model_setting['hyperparameters'] = {}
        
        settings.append(single_model_setting)
    
    return settings


#given a model with its algorithm and hyperparameter names specified, find the range of this hyperparameter within all the models at hand
def get_hyperparameter_range(HEADINGS, algorithm_name, hyperparameter_name):
    algorithm_indices = np.in1d(HEADINGS[0], algorithm_name).nonzero()[0]
    hyperparameter_position = np.array(HEADINGS[1])[algorithm_indices][0].index(hyperparameter_name)
    hyperparameter_range = [hyperparameter_values[hyperparameter_position] for hyperparameter_values in np.array(HEADINGS[2])[algorithm_indices]]
    return hyperparameter_range

#Whether to tune hyperparameters in the whole ranges corresponding to all the relevant columns. For example, if all the k values corresponding to kNN are {1, 3, 5, 7, 9}, then "wide" means we will tune k within the range of 1 to 9; "not wide" means we will only tune k within ±2 of k's initial value.
if WIDE_HYPERPARAMETER_RANGE:
    def kNN_range(hyperparameters):
        hyperparameter_range = get_hyperparameter_range(HEADINGS, 'kNN', 'k')
        return {'k': UniformIntegerHyperparameter('k', min(hyperparameter_range), max(hyperparameter_range), default_value=hyperparameters['k'])}


    def CART_range(hyperparameters):
        global HEADINGS
        hyperparameter_range = get_hyperparameter_range(HEADINGS, 'CART', 'min_samples_split')
        return {'min_samples_split': UniformFloatHyperparameter('min_samples_split', min(hyperparameter_range), max(hyperparameter_range), default_value=hyperparameters['min_samples_split'])}

    def RF_range(hyperparameters):
        hyperparameter_range = get_hyperparameter_range(HEADINGS, 'RF', 'min_samples_split')
        return {'min_samples_split': UniformFloatHyperparameter('min_samples_split', min(hyperparameter_range), max(hyperparameter_range), default_value=hyperparameters['min_samples_split'])}

    def ExtraTrees_range(hyperparameters):
        hyperparameter_range = get_hyperparameter_range(HEADINGS, 'ExtraTrees', 'min_samples_split')
        return {'min_samples_split': UniformFloatHyperparameter('min_samples_split', min(hyperparameter_range), max(hyperparameter_range), default_value=hyperparameters['min_samples_split'])}


    def GB_range(hyperparameters):
        hyperparameter_range = get_hyperparameter_range(HEADINGS, 'GB', 'learning_rate')
        return {'learning_rate': UniformFloatHyperparameter('learning_rate', min(hyperparameter_range), max(hyperparameter_range), default_value=hyperparameters['learning_rate'])}

    def lSVM_range(hyperparameters):
        hyperparameter_range = get_hyperparameter_range(HEADINGS, 'lSVM', 'C')
        return {'C': UniformFloatHyperparameter('C', min(hyperparameter_range), max(hyperparameter_range), default_value=hyperparameters['C'])}

    def kSVM_range(hyperparameters):
        hyperparameter_range = get_hyperparameter_range(HEADINGS, 'kSVM', 'C')
        return {'C': UniformFloatHyperparameter('C', min(hyperparameter_range), max(hyperparameter_range), default_value=hyperparameters['C'])}

    def Logit_range(hyperparameters):
        hyperparameter_range = get_hyperparameter_range(HEADINGS, 'Logit', 'C')
        return {'C': UniformFloatHyperparameter('C', min(hyperparameter_range), max(hyperparameter_range), default_value=hyperparameters['C'])}

    def Perceptron_range(hyperparameters):
        return {}

    def Adaboost_range(hyperparameters):
        hyperparameter_range_n_estimators = get_hyperparameter_range(HEADINGS, 'Adaboost', 'n_estimators')
        hyperparameter_range_learning_rate = get_hyperparameter_range(HEADINGS, 'Adaboost', 'learning_rate')
        
        return {'n_estimators': UniformIntegerHyperparameter('n_estimators', int(min(hyperparameter_range_n_estimators)), int(max(hyperparameter_range_n_estimators)), default_value=int(hyperparameters['n_estimators'])),
               'learning_rate': UniformFloatHyperparameter('learning_rate', int(min(hyperparameter_range_learning_rate)), int(max(hyperparameter_range_learning_rate)), default_value=hyperparameters['learning_rate'])}

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
