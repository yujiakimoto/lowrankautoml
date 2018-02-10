import numpy as np
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron as PerceptronClf
from sklearn.model_selection import KFold

RANDOM_STATE = 0

# Error calculator for class average for each fold
def error_calc(test_labels, predictions):
    error = []
    
    eps = 1e-15
    
    # Calculate the custom metric 1- 0.5(Specificity + Sensitivity)
    for i in np.unique(test_labels):
        #true positives
        tp=((test_labels == i) & (predictions==i)).sum()

        #true negatives
        tn=((test_labels != i) & (predictions!=i)).sum()

        #false positives
        fp=((test_labels != i) & (predictions==i)).sum()

        #false negatives
        fn=((test_labels == i) & (predictions!=i)).sum()
        
        tp_new = sp.maximum(eps, tp)
        pos_new = sp.maximum(eps, tp+fn)
        tn_new = sp.maximum(eps, tn)
        neg_new = sp.maximum(eps, tn+fp)
        
        error.append(1- 0.5*(tp_new/pos_new) - 0.5*(tn_new/neg_new))
    
    #convert the error list into numpy array
    error_np = np.array(error)
    return(np.mean(error_np))

#returns the kfold CV error given data and classifier
def kfolderror(data_numeric, data_labels, clf, num_splits):
    Error_Aver = 0
    Predictions=np.zeros(shape=data_labels.shape)
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=RANDOM_STATE)
    for train, test in kf.split(data_numeric):
        X_train=data_numeric[train,:]
        Y_train=data_labels[train]
        X_test=data_numeric[test,:]
        Y_test= data_labels[test]
        
        clf.fit(X_train, Y_train)
        Predictions[test] = clf.predict(X_test)        
        Error_Aver += error_calc(Y_test, Predictions[test])
    
    Error_Aver = Error_Aver/float(num_splits)
    return Error_Aver, Predictions
    
def kNN(data_numeric, data_labels, num_splits=10, k=5, Error_type='ClassAverage', verbose=True):
    clf = KNeighborsClassifier(n_neighbors=k)
    Error_Aver, Predictions = kfolderror(data_numeric, data_labels, clf, num_splits)
    if verbose:
        print("kNN finished (", k,")"+ str(data_numeric.shape))
    return Error_Aver, Predictions, clf
    
def CART(data_numeric, data_labels, min_samples_split=2, num_splits=10, Error_type='ClassAverage', verbose=True):
    clf = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=RANDOM_STATE)
    Error_Aver, Predictions = kfolderror(data_numeric, data_labels, clf, num_splits)
    if verbose:
        print("CART finished (", min_samples_split, ")" + str(data_numeric.shape))
    return Error_Aver, Predictions, clf

def RF(data_numeric, data_labels, min_samples_split=2, num_splits=10,\
                 num_trees=100,num_jobs=1,type_num_features='sqrt',Error_type='ClassAverage', verbose=True):

    clf = RandomForestClassifier(n_estimators=num_trees, min_samples_split=min_samples_split,  max_features=type_num_features, n_jobs=num_jobs, random_state=RANDOM_STATE)
     
    Error_Aver,Predictions = kfolderror(data_numeric, data_labels, clf, num_splits)
    if verbose:
        print("RF finished (", min_samples_split, ")" + str(data_numeric.shape))
    return Error_Aver, Predictions, clf
    

def Perceptron(data_numeric, data_labels, num_splits=10, num_iter=5, penalty=None, Error_type='ClassAverage', verbose=True):
    clf = PerceptronClf(penalty=penalty, n_iter=num_iter, random_state=RANDOM_STATE)
    Error_Aver, Predictions = kfolderror(data_numeric, data_labels, clf, num_splits)
    if verbose:
        print("Perceptron finished " +  str(data_numeric.shape))
    return Error_Aver, Predictions, clf

def Adaboost(data_numeric, data_labels, n_estimators=50, learning_rate=1.0, num_splits=10, Error_type='ClassAverage', verbose=True):

    clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=RANDOM_STATE)
    Error_Aver, Predictions = kfolderror(data_numeric, data_labels, clf, num_splits)
    if verbose:
        print("Adaboost finished (", n_estimators, learning_rate, ")"+  str(data_numeric.shape))
    return Error_Aver, Predictions, clf

def GNB(data_numeric, data_labels, num_splits=10, Error_type='ClassAverage', verbose=True):

    clf = GaussianNB()
    Error_Aver, Predictions = kfolderror(data_numeric, data_labels, clf, num_splits)
    if verbose:
        print("Gaussian Naive Bayes finished "+  str(data_numeric.shape))
    return Error_Aver, Predictions, clf

def GB(data_numeric, data_labels, learning_rate=0.1, num_splits=10, Error_type='ClassAverage', verbose=True): 
    
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=learning_rate, random_state=RANDOM_STATE)
    Error_Aver, Predictions = kfolderror(data_numeric, data_labels, clf, num_splits)
    if verbose:
        print("GradientBoosting finished (", learning_rate, ")" +str(data_numeric.shape))
    return Error_Aver, Predictions, clf

def lSVM(data_numeric, data_labels, C=1.0, num_splits=10, Error_type='ClassAverage', verbose=True):

    clf = LinearSVC(C=C,multi_class='ovr', random_state=RANDOM_STATE)
    Error_Aver, Predictions = kfolderror(data_numeric, data_labels, clf, num_splits)
    if verbose:
        print("LinearSVM finished (", C, ")"+  str(data_numeric.shape))
    return Error_Aver, Predictions, clf

def kSVM(data_numeric, data_labels, C=1.0, num_splits=10, Kernel='rbf', Error_type='ClassAverage', verbose=True):

    clf = SVC(C=C,decision_function_shape='ovo',kernel=Kernel, random_state=RANDOM_STATE)
    Error_Aver, Predictions = kfolderror(data_numeric, data_labels, clf, num_splits)
    if verbose:
        print("KernelSVM finished (", C, ")"+  str(data_numeric.shape))
    return Error_Aver, Predictions, clf

def Logit(data_numeric, data_labels, C=1.0, num_splits=10, penalty='l2', Error_type='ClassAverage', verbose=True):
 
    clf = LogisticRegression(C=C, penalty=penalty, random_state=RANDOM_STATE)
    Error_Aver, Predictions = kfolderror(data_numeric, data_labels, clf, num_splits)
    if verbose:
        print("Logit finished (", C, ")"+  str(data_numeric.shape))
    return Error_Aver, Predictions, clf

def NeuralNet(data_numeric, data_labels, num_splits=10, Solver='sgd',\
                  Alpha=1e-5,Hidden_layer_sizes=(5, 2), Random_state=1,Error_type='ClassAverage', verbose=True):

    clf = MLPClassifier(solver=Solver, alpha=Alpha,hidden_layer_sizes=Hidden_layer_sizes, random_state=RANDOM_STATE)
    Error_Aver, Predictions = kfolderror(data_numeric, data_labels, clf, num_splits)
    if verbose:
        print("Neural Network finished"+  str(data_numeric.shape))
    return Error_Aver, Predictions, clf
