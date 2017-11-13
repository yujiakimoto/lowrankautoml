# Testing with larger ensemble size
import sys
import numpy as np
import multiprocessing as mp
import pandas as pd
import low_rank_models as lrm
import ML_algorithms as ml
import util
import time
from model import Model
from pathos.multiprocessing import ProcessingPool
from sklearn.model_selection import KFold

N_CORES = None
OUTER_FOLDS = 3
RANK = 9
ENSEMBLE_CANDIDATES = 0
ENSEMBLE_MODELS = 6
VERBOSE = False

ERROR_MATRIX = pd.read_csv('default/error_matrix.csv', index_col=0).values
N_DATASETS,N_MODELS = ERROR_MATRIX.shape
X,Y,Vt = lrm.pca(ERROR_MATRIX)
INDICES_QR = lrm.pivoted_qr(Vt)[:RANK]

HEADINGS = util.generate_headings(['kNN','CART','GB','lSVM','kSVM','Logit','Perceptron','Adaboost','GNB','RF'], 'default')

ID_LIST = [5,8,9,10,13,15,21,41,42,61,150,183,187,287,310,313,1219,1457,1461,1464,1471,1492,1500,1510,1567,4134,4153,40664]
ID = ID_LIST[int(sys.argv[1])]
DATASET = pd.read_csv('Dataset' + str(ID) + '_Preprocessed.csv', header=None).values
FEATURES = DATASET[:,:-1]
LABELS = DATASET[:,-1]

def generate_settings(i):
        settings = {'algorithm':HEADINGS[0][i], 
                        'hyperparameters':{HEADINGS[1][i][j]:HEADINGS[2][i][j] for j in range(len(HEADINGS[1][i]))}}
        if list(settings['hyperparameters'].keys())[0] == '':
                settings['hyperparameters'] = {}
        return settings
    
def compute_entry(features, labels, index, num_folds, verbose):
    settings = generate_settings(index)              
    model = Model(settings=settings, num_folds=num_folds, verbose=verbose)
    model.fit(features, labels) 
    return model

kf = KFold(OUTER_FOLDS, shuffle=True)
total_error = 0
total_time = 0

for train, test in kf.split(DATASET):
    t_start = time.time()
    
    x_train = FEATURES[train, :]
    y_train = LABELS[train]
    
    x_test = FEATURES[test, :]
    y_test = LABELS[test]
    
    newrow = np.zeros((1, N_MODELS))
    secondlayercolumns = ()
    base_learners = []

    # print('Fitting QR models...')
    p1 = mp.Pool(N_CORES)
    QR_models = [p1.apply_async(compute_entry, args=[x_train, y_train, i, 10, VERBOSE]) for i in INDICES_QR]
    p1.close()
    p1.join()

    for i, model in enumerate(QR_models):
        newrow[:, INDICES_QR[i]] = model.get().error
        secondlayercolumns += (model.get().cv_predictions, )
        base_learners.append(model.get())

    approx = lrm.low_rank_approximation(ERROR_MATRIX, newrow, INDICES_QR)
    # if true value is computed, replace with low rank approximation or not?
    # approx[:, INDICES_QR] = newrow[:, INDICES_QR]
    newrow = np.copy(approx)
    candidate_idx = newrow[0].argsort()[:ENSEMBLE_CANDIDATES]

    # print('\nFitting ensemble candidates...')
    p2 = mp.Pool(N_CORES)
    candidate_models = [p2.apply_async(compute_entry, args=[x_train, y_train, i, 10, VERBOSE]) 
                        for i in candidate_idx if i not in INDICES_QR]
    p2.close()
    p2.join()

    for i, model in enumerate(candidate_models):
        newrow[:, candidate_idx[i]] = model.get().error
        secondlayercolumns += (model.get().cv_predictions, )
        base_learners.append(model.get())

    ensemble_idx = newrow[0].argsort()[:ENSEMBLE_MODELS]
    optimized_models = [Model(generate_settings(i), 10, VERBOSE, x_train, y_train)
                        for i in ensemble_idx]

    # print('\nConducting Bayesian Optimization...')
    p3 = ProcessingPool(N_CORES)
    optimized_models = p3.map(Model.bayesian_optimize, optimized_models)

    p4 = mp.Pool(N_CORES)
    boptimized_models = [p4.apply_async(Model.fit, args=(model, x_train, y_train)) 
                         for model in base_learners[-ENSEMBLE_MODELS:]]
    p4.close()
    p4.join()

    # print('\nFitting optimized models...')
    for model in boptimized_models:
        secondlayercolumns += (model.get().cv_predictions, )
        base_learners.append(model.get())

    # print('\nFitting stacked learner...')
    secondlayermatrix = np.matrix.transpose(np.stack(secondlayercolumns))
    stacked_learner = Model({'algorithm':'Logit', 'hyperparameters':{'C': 1.0, 'penalty': 'l1'}}, 
                            verbose=VERBOSE)
    stacked_learner = stacked_learner.fit(secondlayermatrix, y_train)
    optimized_stackedlearner = stacked_learner.bayesian_optimize()

    # print('\nGenerating predictions...')
    test_secondlayercolumns = ()
    p5 = mp.Pool(N_CORES)
    weak_predictions = [p5.apply_async(Model.predict, args=(model, x_test)) for model in base_learners]
    p5.close()
    p5.join()

    for column in weak_predictions:
        test_secondlayercolumns += (column.get(), )
    test_secondlayermatrix = np.matrix.transpose(np.stack(test_secondlayercolumns))
    predictions = optimized_stackedlearner.predict(test_secondlayermatrix)

    error = ml.error_calc(y_test, predictions)
    elapsed = time.time() - t_start

    total_error += error
    total_time += elapsed

    # print('\nDone!')
    # print('Error:', error)

avg_error = total_error/OUTER_FOLDS
avg_time = total_time/OUTER_FOLDS

pd.Series([ID, avg_error, avg_time]).to_csv('lrmresults2/dataset' + str(ID) + '.csv')