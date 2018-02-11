import sys
import re
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

N_CORES = 2
OUTER_FOLDS = 3
RANK = 9 #which is also the number of models to calculate in order to get low rank approximation of the new row
ENSEMBLE_CANDIDATES = 0 #the number of best-predicted models to actually calculate
ENSEMBLE_MODELS = 6 #the number of models to Bayesian optimize and add to the final ensemble
NUM_BOPT_ROUNDS = 10 #the default number of Bayesian optimization rounds
NUM_BOPT_ROUNDS_IN_STACKING = 10
VERBOSE = False
FIT_BEST_PREDICTED_MODELS = False #whether to fit the best predicted models given by low rank approximation
ADD_PIVOTS_TO_BASE_LEARNERS = False
GET_TRAINING_ERROR = True
RANDOM_STATE = 0

ERROR_MATRIX = pd.read_csv('default/error_matrix.csv', index_col=0).values
N_DATASETS,N_MODELS = ERROR_MATRIX.shape
X,Y,Vt = lrm.pca(ERROR_MATRIX)
INDICES_QR = lrm.pivoted_qr(Vt)[:RANK]

HEADINGS = util.generate_headings(['kNN','CART','GB','lSVM','kSVM','Logit','Perceptron','Adaboost','GNB','RF'], 'default')

#ID_LIST = [5,8,9,10,13,15,21,41,42,61,150,183,187,287,310,313,1219,1457,1461,1464,1471,1492,1500,1510,1567,4134,4153,40664]
#ID = ID_LIST[int(sys.argv[1])]
#DATASET = pd.read_csv('Dataset' + str(ID) + '_Preprocessed.csv', header=None).values
filename=sys.argv[1]
ID = int(re.findall('\d+', filename)[-1])

DATASET = pd.read_csv(filename, header=None).values

# print(filename)

# print(DATASET.shape)

FEATURES = DATASET[:,:-1]
LABELS = DATASET[:,-1]

#added the number of Bayesian optimization rounds into settings
def generate_settings(i):
        settings = {'algorithm':HEADINGS[0][i], 
                        'hyperparameters':{HEADINGS[1][i][j]:HEADINGS[2][i][j] for j in range(len(HEADINGS[1][i]))}, 'num_bopt_rounds':NUM_BOPT_ROUNDS}
        if list(settings['hyperparameters'].keys())[0] == '':
                settings['hyperparameters'] = {}
        return settings
    
def compute_entry(features, labels, index, num_folds, verbose):
    settings = generate_settings(index)              
    model = Model(settings=settings, num_folds=num_folds, verbose=verbose)
    model.fit(features, labels) 
    return model

for NUM_BOPT_ROUNDS in range(0, 105, 5):
    kf = KFold(OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    total_error = 0
    total_error_base_learners = np.zeros(ENSEMBLE_MODELS)
    total_error_training = 0
    total_error_base_learners_training = np.zeros(ENSEMBLE_MODELS)
    kfold_total_error_base_learners_training = np.zeros(ENSEMBLE_MODELS)
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
            if ADD_PIVOTS_TO_BASE_LEARNERS:
                secondlayercolumns += (model.get().cv_predictions, )
                base_learners.append(model.get())

        approx = lrm.low_rank_approximation(ERROR_MATRIX, newrow, INDICES_QR)
        # if true value is computed, replace with low rank approximation or not?
        # approx[:, INDICES_QR] = newrow[:, INDICES_QR]
        newrow = np.copy(approx)
        
        #START OPTIONAL PART:compute the true values of best models in low rank approx

        if FIT_BEST_PREDICTED_MODELS:

            candidate_idx = newrow[0].argsort()[:ENSEMBLE_CANDIDATES] #the best models in low rank approx

            # print('\nFitting ensemble candidates...')
            p2 = mp.Pool(N_CORES)
            candidate_models = [p2.apply_async(compute_entry, args=[x_train, y_train, i, 10, VERBOSE]) 
                                for i in candidate_idx if i not in INDICES_QR] #compute true errors of the best models in low rank approx
            p2.close()
            p2.join()

            for i, model in enumerate(candidate_models):
                newrow[:, candidate_idx[i]] = model.get().error
                secondlayercolumns += (model.get().cv_predictions, )
                base_learners.append(model.get())

        #END OPTIONAL PART

        ensemble_idx = newrow[0].argsort()[:ENSEMBLE_MODELS]


        # print([generate_settings(i)['num_bopt_rounds'] for i in ensemble_idx])

        # print([Model(generate_settings(i), 10, VERBOSE, x_train, y_train).num_bopt_rounds for i in ensemble_idx])



        optimized_models = [Model(generate_settings(i), 10, VERBOSE, x_train, y_train) for i in ensemble_idx] #bayesian-optimize the best models in low rank approx


        if NUM_BOPT_ROUNDS>0:

            print('\nConducting Bayesian Optimization on Dataset ' + str(ID) + '...')

            print('\nNumber of Bayesian Optimization Rounds:' + str(NUM_BOPT_ROUNDS))

            p3 = ProcessingPool(N_CORES)
            optimized_models = p3.map(Model.bayesian_optimize, optimized_models)

        if GET_TRAINING_ERROR:
            p7 = mp.Pool(N_CORES)
            kfold_error_base_learners_training = [p7.apply_async(ml.kfolderror, args=(x_train, y_train, model, 10)).get()[0] for model in optimized_models]
            
            p7.close()
            p7.join()

            # kfold_error_base_learners_training = [item[0] for item in kfold_error_base_learners_training]



        p4 = mp.Pool(N_CORES)
        # boptimized_models = [p4.apply_async(Model.fit, args=(model, x_train, y_train)) 
        #                      for model in base_learners[-ENSEMBLE_MODELS:]] #fit the best base learners
        optimized_models_fitted = [p4.apply_async(Model.fit, args=(model, x_train, y_train)) for model in optimized_models]

        p4.close()
        p4.join()

        # print('\nFitting optimized models...')
        # for model in boptimized_models:
        for model in optimized_models_fitted:
            secondlayercolumns += (model.get().cv_predictions, )
            base_learners.append(model.get())

        # print('\nFitting stacked learner...')
        secondlayermatrix = np.matrix.transpose(np.stack(secondlayercolumns))
        stacked_learner = Model({'algorithm':'Logit', 'hyperparameters':{'C': 1.0, 'penalty': 'l1'}, 'num_bopt_rounds': NUM_BOPT_ROUNDS}, verbose=VERBOSE)
        stacked_learner = stacked_learner.fit(secondlayermatrix, y_train)
        if NUM_BOPT_ROUNDS>0: #TODO: change to individual control mode
            stacked_learner = stacked_learner.bayesian_optimize()

        # print('\nGenerating predictions...')
        test_secondlayercolumns = ()
        p5 = mp.Pool(N_CORES)
        weak_predictions = [p5.apply_async(Model.predict, args=(model, x_test)) for model in base_learners]
        p5.close()
        p5.join()

        if GET_TRAINING_ERROR:
            p6 = mp.Pool(N_CORES)
            weak_predictions_training = [p6.apply_async(Model.predict, args=(model, x_train)) for model in base_learners]
            p6.close()
            p6.join()


        for column in weak_predictions:
            test_secondlayercolumns += (column.get(), )
        test_secondlayermatrix = np.matrix.transpose(np.stack(test_secondlayercolumns))
        predictions = stacked_learner.predict(test_secondlayermatrix)


        error_base_learners = np.array([ml.error_calc(y_test, column.get()) for column in weak_predictions])
        # print(error_base_learners)
        error = ml.error_calc(y_test, predictions)
        elapsed = time.time() - t_start

        if GET_TRAINING_ERROR:

            test_secondlayercolumns_training = ()

            for column in weak_predictions_training:
                test_secondlayercolumns_training += (column.get(), )
            test_secondlayermatrix_training = np.matrix.transpose(np.stack(test_secondlayercolumns_training))
            predictions_training = stacked_learner.predict(test_secondlayermatrix_training)

            error_base_learners_training = np.array([ml.error_calc(y_train, column.get()) for column in weak_predictions_training])
            #print(error_base_learners)
            error_training = ml.error_calc(y_train, predictions_training)

            total_error_base_learners_training += error_base_learners_training
            total_error_training += error_training
            kfold_total_error_base_learners_training += kfold_error_base_learners_training



        total_error_base_learners += error_base_learners
        total_error += error
        total_time += elapsed

        # print('\nDone!')
        # print('Error:', error)

    avg_error_base_learners = total_error_base_learners/OUTER_FOLDS
    avg_error = total_error/OUTER_FOLDS
    avg_time = total_time/OUTER_FOLDS

    if GET_TRAINING_ERROR:
        avg_error_base_learners_training = total_error_base_learners_training/OUTER_FOLDS
        avg_error_training = total_error_training/OUTER_FOLDS
        avg_kfold_error_base_learners_training = kfold_total_error_base_learners_training/OUTER_FOLDS
     

    print('FINISHED on Dataset ' + str(ID) + 'with Number of Bayesian Optimization Rounds =' + str(NUM_BOPT_ROUNDS))

    
    pd.Series([ID, avg_error, avg_time]).to_csv('our_results/'+str(NUM_BOPT_ROUNDS)+'/dataset' + str(ID) + '.csv')



    result_base_learners = []
    result_base_learners.append(ID)
    for item in avg_error_base_learners:
        result_base_learners.append(item)
    result_base_learners.append(avg_time)

    index_csv = []
    index_csv.append("ID")
    for i in ensemble_idx:
        index_csv.append(generate_settings(i))
    index_csv.append("time")

    pd.DataFrame(result_base_learners, index=index_csv).to_csv('our_results/'+str(NUM_BOPT_ROUNDS)+'/dataset' + str(ID) + '_base_learners.csv')



    if GET_TRAINING_ERROR:

        pd.Series([ID, avg_error_training, avg_time]).to_csv('our_results/'+str(NUM_BOPT_ROUNDS)+'/dataset' + str(ID) + '_training.csv')



        result_base_learners_training = []
        result_base_learners_training.append(ID)
        for item in avg_error_base_learners_training:
            result_base_learners_training.append(item)
        result_base_learners_training.append(avg_time)

        index_csv = []
        index_csv.append("ID")
        for i in ensemble_idx:
            index_csv.append(generate_settings(i))
        index_csv.append("time")

        pd.DataFrame(result_base_learners_training, index=index_csv).to_csv('our_results/'+str(NUM_BOPT_ROUNDS)+'/dataset' + str(ID) + '_base_learners_training.csv')


        result_kfold_base_learners_training = []
        result_kfold_base_learners_training.append(ID)
        for item in avg_kfold_error_base_learners_training:
            result_kfold_base_learners_training.append(item)
        result_kfold_base_learners_training.append(avg_time)

        index_csv = []
        index_csv.append("ID")
        for i in ensemble_idx:
            index_csv.append(generate_settings(i))
        index_csv.append("time")

        pd.DataFrame(result_kfold_base_learners_training, index=index_csv).to_csv('our_results/'+str(NUM_BOPT_ROUNDS)+'/dataset' + str(ID) + '_base_learners_training_kfold.csv')




           
