import scipy.sparse as sps
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer


#Categorical: a boolean array indicating which feature is categorical
#bool_Imputer, bool_Standardization and bool_OneHotEncoder: boolean variables indicating whether to perform this type of data preprocessing method or not


def DataPreprocessing(Data_numeric, Categorical, bool_Imputer=True, bool_Standardization=True, bool_OneHotEncoder=True):
    
    if sps.issparse(Data_numeric):
        Data_numeric = Data_numeric.todense()

    if bool_Imputer:
        # whether there exist categorical features
        bool_cat = bool(np.sum(np.isfinite(np.where(np.asarray(Categorical)==True))))
        # whether there exist noncategorical features
        bool_noncat = bool(np.sum(np.isfinite(np.where(np.asarray(Categorical)==False))))
        
        
        if bool_cat:
            # categorical features
            Data_numeric_cat = Data_numeric[:,Categorical]
            # imputer for missing entries
            imp_cat = Imputer(missing_values='NaN', strategy='most_frequent', axis=0, copy=False)
            imp_cat.fit(Data_numeric_cat)
            Data_numeric_cat = imp_cat.transform(Data_numeric_cat)
            # number of categorical features
            num_cat = Data_numeric_cat.shape[1]
        
        
        if bool_noncat:
            
            #noncategorical features
            Data_numeric_noncat = Data_numeric[:,np.invert(Categorical)]
            imp_noncat = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)
            imp_noncat.fit(Data_numeric_noncat)
            Data_numeric_noncat = imp_noncat.transform(Data_numeric_noncat)
            #number of noncategorical features
            num_noncat = Data_numeric_noncat.shape[1]

        #true if there exist both categorical and noncategorical features
        if bool_cat*bool_noncat:
            
            Data_numeric = np.concatenate((Data_numeric_cat, Data_numeric_noncat), axis=1)
            Categorical = [True for i in range(num_cat)] + [False for i in range(num_noncat)]

        #true if there only exist categorical features
        elif bool_cat*(not bool_noncat):
            Data_numeric = Data_numeric_cat
            Categorical = [True for i in range(num_cat)]
            
        #true if there only exist noncategorical features
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
