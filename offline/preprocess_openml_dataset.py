#Caution: specify your OpenML apikey in line 20 before running this file
#usage: in terminal, type `python preprocess_openml_dataset.py $(OPENML ID)`

import numpy as np
import scipy as sp
import scipy.sparse as sps

import openml
import random
import pandas as pd
import sys

# Preprocessing modules from sklearn
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer


apikey_in_use = YOUR_APIKEY
openml.config.apikey = apikey_in_use

dataset_id = sys.argv[1]

#Arguments:
#Data_numeric: an n-by-d numpy array corresponding to n datapoints and d features
#Categorical: a d-dimensional boolean array indicating whether each feature is categorical
#bool_Imputer: whether to do imputation
#bool_Standardization: whether to do standardization
#bool_OneHotEncoder: whether to do one-hot-encoding

def DataPreprocessing(Data_numeric, Categorical, bool_Imputer=True, bool_Standardization=True, bool_OneHotEncoder=True):
    
    #whether to impute missing entries
    if bool_Imputer:
        # whether there exist categorical features
        bool_cat = bool(np.sum(np.isfinite(np.where(np.asarray(Categorical)==True))))
        # whether there exist noncategorical features
        bool_noncat = bool(np.sum(np.isfinite(np.where(np.asarray(Categorical)==False))))
        
        
        if bool_cat:
            # categorical features
            Data_numeric_cat = Data_numeric[:, Categorical]
            # impute missing entries in categorical features using the most frequent number
            imp_cat = Imputer(missing_values='NaN', strategy='most_frequent', axis=0, copy=False)
            imp_cat.fit(Data_numeric_cat)
            Data_numeric_cat = imp_cat.transform(Data_numeric_cat)
            # number of categorical features
            num_cat = Data_numeric_cat.shape[1]
        
        
        if bool_noncat:            
            #noncategorical features
            Data_numeric_noncat = Data_numeric[:,np.invert(Categorical)]
            #impute missing entries in non-categorical features using mean
            imp_noncat = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)
            imp_noncat.fit(Data_numeric_noncat)
            Data_numeric_noncat = imp_noncat.transform(Data_numeric_noncat)
            #number of noncategorical features
            num_noncat = Data_numeric_noncat.shape[1]
        
        #whether there exist both categorical and noncategorical features
        if bool_cat*bool_noncat:
            
            Data_numeric = np.concatenate((Data_numeric_cat, Data_numeric_noncat), axis=1)
            Categorical = [True for i in range(num_cat)] + [False for i in range(num_noncat)]
        
        #whether there are only categorical features
        elif bool_cat*(not bool_noncat):
            Data_numeric = Data_numeric_cat
            Categorical = [True for i in range(num_cat)]
        
        #whether there are only noncategorical features
        elif (not bool_cat)*bool_noncat:
            Data_numeric = Data_numeric_noncat
            Categorical = [False for i in range(num_noncat)]

    # OneHotEncoding for categorical features
    if bool_OneHotEncoder:
        
        #check if there exist categorical features
        if np.sum(np.isfinite(np.where(np.asarray(Categorical) == True))):
            enc=OneHotEncoder(categorical_features = Categorical)
            enc.fit(Data_numeric)
            Data_numeric = enc.transform(Data_numeric).toarray()
            
    # Standardization of all features
    if bool_Standardization:
        if bool_OneHotEncoder:
            Data_numeric = scale(Data_numeric)
        
        #check if there exist numerical features
        elif np.sum(np.isfinite(np.where(np.asarray(Categorical) == False))):
            Data_numeric[:,np.invert(Categorical)] = scale(Data_numeric[:,np.invert(Categorical)])

    print("DataPreprocessing finished")
    return Data_numeric, Categorical


dataset=openml.datasets.get_dataset(dataset_id)
data_numeric,data_labels,categorical = dataset.get_data(target=dataset.default_target_attribute,return_categorical_indicator=True)

if sps.issparse(data_numeric):
    data_numeric=data_numeric.todense()

#doing imputation and standardization and not doing one-hot-encoding achieves optimal empirical performances (smallest classification error) on a bunch of OpenML datasets
data_numeric, categorical = DataPreprocessing(Data_numeric=data_numeric, Categorical=categorical, bool_Imputer=True, bool_Standardization=True, bool_OneHotEncoder=False)

#the output is a preprocessed dataset with all the columns except the last one being preprocessed features, and the last column being labels
dat = np.append(data_numeric, np.array(data_labels, ndmin=2).T,axis=1)

np.savetxt(fname='dataset_'+str(dataset_id)+'_preprocessed.csv', X=dat, delimiter=',')
